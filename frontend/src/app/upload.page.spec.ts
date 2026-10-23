import { HttpEventType } from '@angular/common/http';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { UploadPage } from './upload.page';
import { UploadQueueService } from './upload-queue.service';
import { HEX_B, pdf, tick, uploadResponse } from './testing';

const URL = '/api/v1/resume/upload';

describe('Resume Upload page', () => {
  let fixture: ComponentFixture<UploadPage>;
  let http: HttpTestingController;
  let queue: UploadQueueService;

  const text = () => fixture.nativeElement.textContent as string;
  const statuses = () => queue.items().map((i) => i.status);

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [provideHttpClient(), provideHttpClientTesting()] });
    http = TestBed.inject(HttpTestingController);
    queue = TestBed.inject(UploadQueueService);
    fixture = TestBed.createComponent(UploadPage);
    fixture.detectChanges();
  });

  afterEach(() => http.verify());

  it('uploads several files one at a time, showing queued then uploading then indexed', async () => {
    queue.add([pdf('a.pdf'), pdf('b.pdf'), pdf('c.pdf')]);
    expect(statuses()).toEqual(['uploading', 'queued', 'queued']);
    fixture.detectChanges();
    expect(text()).toContain('a.pdf');
    expect(text()).toContain('Queued');

    const first = http.expectOne(URL);
    expect(first.request.headers.has('Authorization')).toBeFalse();      // no API key is sent
    expect((first.request.body as FormData).get('file')).toEqual(jasmine.any(File));
    first.event({ type: HttpEventType.UploadProgress, loaded: 50, total: 100 });
    expect(queue.items()[0].progress).toBe(50);
    fixture.detectChanges();
    expect(text()).toContain('50%');
    first.flush(uploadResponse());
    await tick();
    expect(statuses()).toEqual(['indexed', 'uploading', 'queued']);

    http.expectOne(URL).flush(uploadResponse({ document_id: HEX_B }));
    await tick();
    http.expectOne(URL).flush(uploadResponse({ document_id: 'c'.repeat(64) }));
    await tick();
    expect(statuses()).toEqual(['indexed', 'indexed', 'indexed']);
    fixture.detectChanges();
    expect(text()).toContain('3 indexed');
  });

  it('never marks a pending resume as indexed', async () => {
    queue.add([pdf('slow.pdf')]);
    http.expectOne(URL).flush(uploadResponse({ status: 'PENDING_INDEXING', ready: false }), { status: 202, statusText: 'Accepted' });
    await tick();
    expect(statuses()).toEqual(['pending']);
    fixture.detectChanges();
    expect(text()).toContain('Pending indexing');
    expect(text()).not.toContain('Indexed');
  });

  it('shows a safe error for a failed upload and keeps going with the next file', async () => {
    queue.add([pdf('bad.pdf'), pdf('good.pdf')]);
    http.expectOne(URL).flush(
      { error: { code: 'NO_TEXT_CONTENT', message: 'No readable resume text was found.' } },
      { status: 422, statusText: 'Unprocessable' });
    await tick();
    http.expectOne(URL).flush(uploadResponse());
    await tick();
    expect(statuses()).toEqual(['failed', 'indexed']);
    expect(queue.items()[0].uncertain).toBeFalse();
    fixture.detectChanges();
    expect(text()).toContain('No readable resume text was found.');
    expect(text()).toContain('1 failed');
  });

  it('retries a failed upload', async () => {
    queue.add([pdf('x.pdf')]);
    http.expectOne(URL).flush(
      { error: { code: 'EMBEDDING_UNAVAILABLE', message: 'The embedding service is not available right now.' } },
      { status: 503, statusText: 'Unavailable' });
    await tick();
    expect(statuses()).toEqual(['failed']);
    fixture.detectChanges();
    const retry = Array.from(fixture.nativeElement.querySelectorAll('button')).find(
      (b) => (b as HTMLElement).textContent?.trim() === 'Retry') as HTMLButtonElement;
    retry.click();
    expect(statuses()).toEqual(['uploading']);
    http.expectOne(URL).flush(uploadResponse());
    await tick();
    expect(statuses()).toEqual(['indexed']);
    expect(queue.items()[0].error).toBeNull();
  });

  it('treats a network failure as uncertain and allows a safe retry', async () => {
    queue.add([pdf('net.pdf')]);
    http.expectOne(URL).error(new ProgressEvent('error'));
    await tick();
    expect(statuses()).toEqual(['failed']);
    expect(queue.items()[0].uncertain).toBeTrue();
    expect(queue.items()[0].error?.code).toBe('NETWORK');
    fixture.detectChanges();
    expect(text()).toContain('retrying is safe');
  });

  it('retries a pending resume and then shows it indexed', async () => {
    queue.add([pdf('p.pdf')]);
    http.expectOne(URL).flush(uploadResponse({ status: 'PENDING_INDEXING', ready: false }), { status: 202, statusText: 'Accepted' });
    await tick();
    queue.retry(queue.items()[0].id);
    http.expectOne(URL).flush(uploadResponse());
    await tick();
    expect(statuses()).toEqual(['indexed']);
  });

  it('rejects non-PDF files locally without calling the API', () => {
    queue.add([new File(['x'], 'notes.txt')]);
    http.expectNone(URL);
    expect(statuses()).toEqual(['failed']);
    fixture.detectChanges();
    expect(text()).toContain('Only PDF files are accepted.');
    expect(text()).not.toContain('Retry');
  });

});
