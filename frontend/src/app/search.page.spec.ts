import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { SearchPage } from './search.page';
import { HEX_B, rankResponse } from './testing';

const URL = '/api/v1/jd/rank';

describe('JD Candidate Search page', () => {
  let fixture: ComponentFixture<SearchPage>;
  let page: SearchPage;
  let http: HttpTestingController;

  const text = () => fixture.nativeElement.textContent as string;

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [provideHttpClient(), provideHttpClientTesting()] });
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(SearchPage);
    page = fixture.componentInstance;
    fixture.detectChanges();
  });

  afterEach(() => http.verify());

  it('submits only the job description and shows loading then ranked results with evidence', () => {
    page.jobDescription = '  Need Python  ';
    expect(page.canSearch()).toBeTrue();
    page.search();
    expect(page.loading()).toBeTrue();
    const req = http.expectOne(URL);
    expect(req.request.body).toEqual({ job_description: 'Need Python' });      // no candidate ids or tokens
    expect(req.request.headers.has('Authorization')).toBeFalse();
    req.flush(rankResponse());
    expect(page.loading()).toBeFalse();
    fixture.detectChanges();
    expect(text()).toContain('#1 jane.pdf');
    expect(text()).toContain('Required 100%');
    expect(text()).toContain('Preferred 0%');
    expect(text()).toContain('Python, SQL');                 // skill evidence
    expect(text()).toContain('Missing preferred: Azure');
    expect(text()).toContain('Ranked 2 of 2');
  });

  it('keeps the backend order and does not re-rank', () => {
    page.jobDescription = 'x';
    page.search();
    const body = rankResponse();
    body.candidates = [body.candidates[1], body.candidates[0]];              // deliberately odd order
    http.expectOne(URL).flush(body);
    expect(page.result()!.candidates.map((c) => c.rank)).toEqual([2, 1]);
    fixture.detectChanges();
    const rows = Array.from(fixture.nativeElement.querySelectorAll('.results summary')).map((e) => (e as HTMLElement).textContent);
    expect(rows[0]).toContain('#2');
    expect(rows[1]).toContain('#1');
  });

  it('shows missing required skills, warnings, and a short id when there is no filename', () => {
    page.jobDescription = 'x';
    page.search();
    http.expectOne(URL).flush(rankResponse());
    fixture.detectChanges();
    expect(text()).toContain('Missing required: Python');
    expect(text()).toContain('NO_RESUME_VECTORS');
    expect(text()).toContain(HEX_B.slice(0, 12));
  });

  it('shows skipped resumes and a truncated pool', () => {
    page.jobDescription = 'x';
    page.search();
    http.expectOne(URL).flush(rankResponse({
      pool: { discovered: 150, considered: 100, ranked: 98, skipped: 2, truncated: true, max_considered: 100 },
      skipped: [{ document_id: HEX_B, reason: 'STALE' }, { document_id: 'c'.repeat(64), reason: 'TOO_LARGE' }],
    }));
    fixture.detectChanges();
    expect(text()).toContain('only the first 100 resumes were ranked');
    expect(text()).toContain('2 resume(s) were skipped');
    expect(text()).toContain('not fully indexed yet');
    expect(text()).toContain('too large to match completely');
  });

  it('says so when the pool is empty', () => {
    page.jobDescription = 'x';
    page.search();
    http.expectOne(URL).flush(rankResponse({
      pool: { discovered: 0, considered: 0, ranked: 0, skipped: 0, truncated: false, max_considered: 100 },
      candidates: [],
    }));
    fixture.detectChanges();
    expect(text()).toContain('No resumes available to rank');
  });

  it('shows the API error and clears loading', () => {
    page.jobDescription = 'x';
    page.search();
    http.expectOne(URL).flush(
      { error: { code: 'NO_SKILLS_IN_JOB_DESCRIPTION', message: 'The job description lists no skills.' } },
      { status: 422, statusText: 'Unprocessable' });
    expect(page.loading()).toBeFalse();
    fixture.detectChanges();
    expect(text()).toContain('The job description lists no skills.');
    expect(page.result()).toBeNull();
  });

  it('reports a network failure', () => {
    page.jobDescription = 'x';
    page.search();
    http.expectOne(URL).error(new ProgressEvent('error'));
    expect(page.error()?.code).toBe('NETWORK');
    fixture.detectChanges();
    expect(text()).toContain('Cannot reach the API');
  });

  it('blocks searching without a job description', () => {
    expect(page.canSearch()).toBeFalse();
    page.jobDescription = 'x';
    expect(page.canSearch()).toBeTrue();
    page.jobDescription = '   ';
    expect(page.canSearch()).toBeFalse();
  });

  it('submits structured fields without free text and preserves skill categories', () => {
    page.role = '  Full Stack Developer  ';
    page.requiredSkillInput = ' C# ';
    page.addSkill('required');
    page.preferredSkillInput = ' Angular ';
    page.minimumExperience = 2.5;
    page.maximumExperience = 6;
    page.education = ' BSc ';
    page.educationRequired = true;
    page.responsibilities = 'Build APIs\n\n Maintain UI ';
    page.additionalPreferences = ' Azure experience ';
    page.includeReasoning = true;
    expect(page.canSearch()).toBeTrue();
    page.search();
    const req = http.expectOne(URL);
    expect(req.request.body).toEqual({
      job_description: '',
      include_reasoning: true,
      structured_jd: {
        role: 'Full Stack Developer', required_skills: ['C#'], preferred_skills: ['Angular'],
        experience: { minimum_years: 2.5, maximum_years: 6 },
        education: { required: true, description: 'BSc' },
        responsibilities: ['Build APIs', 'Maintain UI'], additional_preferences: ['Azure experience'],
      },
    });
    req.flush(rankResponse());
  });

  it('prevents duplicates within and across skill categories and supports removal', () => {
    page.requiredSkillInput = 'Python';
    page.addSkill('required');
    page.requiredSkillInput = ' python ';
    page.addSkill('required');
    expect(page.requiredSkills).toEqual(['Python']);
    page.requiredSkillInput = '';
    page.preferredSkillInput = 'PYTHON';
    page.addSkill('preferred');
    expect(page.preferredSkills).toEqual([]);
    page.role = 'Developer';
    expect(page.canSearch()).toBeFalse();
    page.removeSkill('required', 0);
    page.addSkill('preferred');
    expect(page.preferredSkills).toEqual(['PYTHON']);
    expect(page.canSearch()).toBeTrue();
  });

  it('validates structured role, skills, experience range and line limits before sending', () => {
    page.role = 'Developer';
    expect(page.canSearch()).toBeFalse();
    page.preferredSkillInput = 'Angular';
    expect(page.canSearch()).toBeTrue();
    page.minimumExperience = 5;
    page.maximumExperience = 2;
    page.search();
    http.expectNone(URL);
    expect(page.validationMessage()).toContain('Minimum experience');
    page.maximumExperience = 61;
    expect(page.canSearch()).toBeFalse();
    page.maximumExperience = null;
    page.minimumExperience = 0;
    expect(page.canSearch()).toBeTrue();
    page.responsibilities = Array(51).fill('Build APIs').join('\n');
    expect(page.canSearch()).toBeFalse();
  });

  it('shows backend scores and distinguishes confirmed evidence, semantic matches and reasoning states', () => {
    page.jobDescription = 'Need Python';
    page.search();
    const body = rankResponse();
    body.candidates[0].overall_match_score = 83.25;
    body.candidates[0].required_skills[0].status = 'EXPLICIT_TEXT_EVIDENCE';
    body.candidates[0].preferred_skills[0].status = 'SEMANTIC_MATCH';
    body.candidates[0].reasoning = {
      status: 'AVAILABLE', strengths: ['Python evidence'], gaps: ['Azure missing'],
      experience_summary: 'Experience not established', recommendation: 'Good Match',
    };
    body.candidates[1].overall_match_score = 0;
    body.candidates[1].reasoning = {
      status: 'UNAVAILABLE', strengths: [], gaps: [], experience_summary: '',
      recommendation: 'Insufficient Evidence',
    };
    http.expectOne(URL).flush(body);
    fixture.detectChanges();
    expect(text()).toContain('Match 83.25%');
    expect(text()).toContain('Match 0%');
    expect(text()).toContain('Required 100%');
    expect(text()).toContain('Confirmed text');
    expect(text()).toContain('Semantic match');
    expect(text()).toContain('unconfirmed');
    expect(text()).toContain('Good Match');
    expect(text()).toContain('Python evidence');
    expect(text()).toContain('Experience not established');
    expect(text()).toContain('Candidate reasoning unavailable');
  });

  it('shows server failures without retaining stale results', () => {
    page.jobDescription = 'Need Python';
    page.result.set(rankResponse());
    page.search();
    http.expectOne(URL).flush(
      { error: { code: 'SERVICE_UNAVAILABLE', message: 'The service is not available right now.' } },
      { status: 503, statusText: 'Unavailable' });
    expect(page.result()).toBeNull();
    expect(page.loading()).toBeFalse();
    fixture.detectChanges();
    expect(text()).toContain('The service is not available right now.');
  });
});
