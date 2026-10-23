import { Component, computed, inject } from '@angular/core';
import { QueueItem, UploadQueueService } from './upload-queue.service';

@Component({
  selector: 'app-upload-page',
  templateUrl: './upload.page.html',
})
export class UploadPage {
  protected readonly queue = inject(UploadQueueService);

  protected readonly counts = computed(() => {
    const items = this.queue.items();
    const count = (s: QueueItem['status']) => items.filter((i) => i.status === s).length;
    return { queued: count('queued'), uploading: count('uploading'), indexed: count('indexed'),
             pending: count('pending'), failed: count('failed') };
  });

  protected readonly labels: Record<QueueItem['status'], string> = {
    queued: 'Queued', uploading: 'Uploading', indexed: 'Indexed', pending: 'Pending indexing', failed: 'Failed',
  };

  onFiles(event: Event): void {
    const input = event.target as HTMLInputElement;
    const files = Array.from(input.files ?? []);
    input.value = '';
    if (files.length) this.queue.add(files);
  }
}
