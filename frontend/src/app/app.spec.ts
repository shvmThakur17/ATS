import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { provideRouter, Router } from '@angular/router';
import { RouterTestingHarness } from '@angular/router/testing';
import { App } from './app';
import { routes } from './app.routes';
import { SearchPage } from './search.page';
import { UploadPage } from './upload.page';

describe('App shell and routing', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideRouter(routes), provideHttpClient(), provideHttpClientTesting()],
    });
  });

  it('has separate navigation items for the two modules', () => {
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();
    const links = Array.from(fixture.nativeElement.querySelectorAll('nav a')) as HTMLAnchorElement[];
    expect(links.map((a) => a.textContent?.trim())).toEqual(['Resume Upload', 'JD Candidate Search']);
    expect(links.map((a) => a.getAttribute('href'))).toEqual(['/upload', '/search']);
    expect(fixture.nativeElement.querySelector('input[type=password]')).toBeNull();     // no API key field
  });

  it('redirects the root URL to the upload page', async () => {
    const harness = await RouterTestingHarness.create();
    await harness.navigateByUrl('/');
    expect(TestBed.inject(Router).url).toBe('/upload');
    expect(harness.routeDebugElement?.componentInstance).toEqual(jasmine.any(UploadPage));
  });

  it('shows the search page at /search', async () => {
    const harness = await RouterTestingHarness.create();
    await harness.navigateByUrl('/search');
    expect(harness.routeDebugElement?.componentInstance).toEqual(jasmine.any(SearchPage));
  });


});
