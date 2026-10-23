import { HttpClient, HttpErrorResponse, HttpEvent } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, catchError, throwError } from 'rxjs';
import { ApiErrorInfo, JdRankRequest, PoolRankResponse, ResumeUploadResponse } from './models';

export function toApiError(err: unknown): ApiErrorInfo {
  if (err instanceof HttpErrorResponse) {
    const body = err.error?.error;
    if (body?.code) {
      return { status: err.status, code: body.code, message: body.message ?? 'Request failed.' };
    }
    if (err.status === 0) {
      return { status: 0, code: 'NETWORK', message: 'Cannot reach the API. Is the backend running?' };
    }
    return { status: err.status, code: 'HTTP_ERROR', message: `Request failed (${err.status}).` };
  }
  return { status: 0, code: 'UNKNOWN', message: 'Unexpected error.' };
}

@Injectable({ providedIn: 'root' })
export class ResumeApiService {
  private readonly http = inject(HttpClient);
  private readonly base = '/api/v1';

  /** POST /resume/upload as an event stream so the caller can show upload progress. */
  upload(file: File): Observable<HttpEvent<ResumeUploadResponse>> {
    const form = new FormData();
    form.append('file', file, file.name);
    return this.http
      .post<ResumeUploadResponse>(`${this.base}/resume/upload`, form, {
        observe: 'events',
        reportProgress: true,
      })
      .pipe(catchError((e) => throwError(() => toApiError(e))));
  }

  /** POST /jd/rank: the server discovers and ranks the shared pool. */
  rank(request: JdRankRequest): Observable<PoolRankResponse> {
    return this.http
      .post<PoolRankResponse>(`${this.base}/jd/rank`, request)
      .pipe(catchError((e) => throwError(() => toApiError(e))));
  }
}
