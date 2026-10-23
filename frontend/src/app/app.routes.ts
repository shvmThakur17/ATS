import { Routes } from '@angular/router';
import { SearchPage } from './search.page';
import { UploadPage } from './upload.page';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'upload' },
  { path: 'upload', component: UploadPage, title: 'Resume Upload' },
  { path: 'search', component: SearchPage, title: 'JD Candidate Search' },
  { path: '**', redirectTo: 'upload' },
];
