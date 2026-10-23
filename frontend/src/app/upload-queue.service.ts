import { HttpEventType } from '@angular/common/http';
import { Injectable, inject, signal } from '@angular/core';
import { ApiErrorInfo, ResumeUploadResponse } from './models';
import { ResumeApiService } from './resume-api.service';

export type UploadStatus = 'queued' | 'uploading' | 'indexed' | 'pending' | 'failed';

export interface QueueItem {
  id: number;
  file: File;
  name: string;
  status: UploadStatus;
  /** 0-100 while uploading, when the browser reports it. */
  progress: number | null;
  error: ApiErrorInfo | null;
  /** Failed, but the server may have stored the resume (network drop, timeout, unconfirmed index). Retry is safe. */
  uncertain: boolean;
  pageCount: number | null;
}

/** Errors after which the resume may already be stored. Re-uploading the same file is idempotent. */
const UNCERTAIN_CODES = new Set(['NETWORK', 'PROCESSING_TIMEOUT', 'READINESS_UNCONFIRMED', 'STORAGE_UNAVAILABLE']);

/**
 * Upload queue shared by the Resume Upload page. Files go one at a time because the backend ingests on a single
 * worker. Lives in a root service so uploads keep running (and their state survives) when HR switches pages.
 */
@Injectable({ providedIn: 'root' })
export class UploadQueueService {
  private readonly api = inject(ResumeApiService);
  private nextId = 1;
  private running = false;

  readonly items = signal<QueueItem[]>([]);

  add(files: File[]): void {
    const added = files.map<QueueItem>((file) => {
      const isPdf = file.name.toLowerCase().endsWith('.pdf');
      return {
        id: this.nextId++, file, name: file.name,
        status: isPdf ? 'queued' : 'failed', progress: null, uncertain: false, pageCount: null,
        error: isPdf ? null : { status: 0, code: 'NOT_PDF', message: 'Only PDF files are accepted.' },
      };
    });
    this.items.update((list) => [...list, ...added]);
    void this.run();
  }

  /** Retry a failed or pending resume. Safe for uncertain uploads: re-ingesting the same PDF replaces its data. */
  retry(id: number): void {
    const item = this.items().find((i) => i.id === id);
    if (!item || (item.status !== 'failed' && item.status !== 'pending') || !item.file.name.toLowerCase().endsWith('.pdf')) {
      return;
    }
    this.patch(id, { status: 'queued', error: null, uncertain: false, progress: null });
    void this.run();
  }

  remove(id: number): void {
    this.items.update((list) => list.filter((i) => i.id !== id || i.status === 'uploading'));
  }

  /** Starts (or resumes) processing queued files. */
  async run(): Promise<void> {
    if (this.running) return;
    this.running = true;
    try {
      for (let next = this.firstQueued(); next; next = this.firstQueued()) {
        await this.uploadOne(next);
      }
    } finally {
      this.running = false;
    }
  }

  private firstQueued(): QueueItem | undefined {
    return this.items().find((i) => i.status === 'queued');
  }

  private patch(id: number, changes: Partial<QueueItem>): void {
    this.items.update((list) => list.map((i) => (i.id === id ? { ...i, ...changes } : i)));
  }

  private uploadOne(item: QueueItem): Promise<void> {
    this.patch(item.id, { status: 'uploading', progress: 0, error: null });
    return new Promise<void>((resolve) => {
      this.api.upload(item.file).subscribe({
        next: (event) => {
          if (event.type === HttpEventType.UploadProgress && event.total) {
            this.patch(item.id, { progress: Math.round((100 * event.loaded) / event.total) });
          } else if (event.type === HttpEventType.Response && event.body) {
            this.finish(item.id, event.body);
          }
        },
        error: (error: ApiErrorInfo) => {
          this.patch(item.id, {
            status: 'failed', progress: null, error,
            uncertain: UNCERTAIN_CODES.has(error.code) || error.status === 504,
          });
          resolve();
        },
        complete: () => resolve(),
      });
    });
  }

  private finish(id: number, body: ResumeUploadResponse): void {
    // Only an explicit ready=true is "indexed". Anything else (including 202 PENDING_INDEXING) stays pending.
    const indexed = body.ready === true && body.status === 'INGESTED';
    this.patch(id, { status: indexed ? 'indexed' : 'pending', progress: 100, pageCount: body.page_count, error: null });
  }
}
