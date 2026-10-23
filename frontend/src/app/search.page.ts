import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ApiErrorInfo, JdRankRequest, PoolCandidateResult, PoolRankResponse } from './models';
import { ResumeApiService } from './resume-api.service';

@Component({
  selector: 'app-search-page',
  imports: [FormsModule],
  templateUrl: './search.page.html',
})
export class SearchPage {
  private readonly api = inject(ResumeApiService);

  jobDescription = '';
  role = '';
  requiredSkills: string[] = [];
  preferredSkills: string[] = [];
  requiredSkillInput = '';
  preferredSkillInput = '';
  minimumExperience: number | null = null;
  maximumExperience: number | null = null;
  education = '';
  educationRequired = false;
  responsibilities = '';
  additionalPreferences = '';
  includeReasoning = false;
  skillMessage = '';
  readonly loading = signal(false);
  readonly error = signal<ApiErrorInfo | null>(null);
  readonly result = signal<PoolRankResponse | null>(null);

  canSearch(): boolean {
    return !this.loading() && !this.validationMessage();
  }

  private hasStructuredFields(): boolean {
    return !!(this.role.trim() || this.requiredSkills.length || this.preferredSkills.length
      || this.requiredSkillInput.trim() || this.preferredSkillInput.trim()
      || this.minimumExperience !== null || this.maximumExperience !== null
      || this.education.trim() || this.educationRequired || this.responsibilities.trim()
      || this.additionalPreferences.trim());
  }

  private lines(value: string): string[] {
    return value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  }

  private skillValues(category: 'required' | 'preferred'): string[] {
    const values = category === 'required'
      ? [...this.requiredSkills, this.requiredSkillInput.trim()]
      : [...this.preferredSkills, this.preferredSkillInput.trim()];
    return values.filter((value, index) => value && values.findIndex(
      (item) => item.toLowerCase() === value.toLowerCase()) === index);
  }

  addSkill(category: 'required' | 'preferred'): void {
    const value = (category === 'required' ? this.requiredSkillInput : this.preferredSkillInput).trim();
    if (!value) return;
    const existing = [...this.requiredSkills, ...this.preferredSkills];
    if (existing.some((skill) => skill.toLowerCase() === value.toLowerCase())) {
      this.skillMessage = 'This skill is already added. Remove it before changing its category.';
      return;
    }
    const skills = category === 'required' ? this.requiredSkills : this.preferredSkills;
    if (skills.length >= 100) {
      this.skillMessage = 'Each category supports up to 100 skills.';
      return;
    }
    skills.push(value);
    if (category === 'required') this.requiredSkillInput = '';
    else this.preferredSkillInput = '';
    this.skillMessage = '';
  }

  removeSkill(category: 'required' | 'preferred', index: number): void {
    (category === 'required' ? this.requiredSkills : this.preferredSkills).splice(index, 1);
    this.skillMessage = '';
  }

  validationMessage(): string {
    if (this.jobDescription.length > 20000) return 'Job description must be at most 20,000 characters.';
    if (!this.hasStructuredFields()) return this.jobDescription.trim() ? '' : 'Enter a role and skills, or a job description.';
    if (!this.role.trim() || this.role.trim().length > 200) return 'Enter a job role of up to 200 characters.';
    const required = this.skillValues('required');
    const preferred = this.skillValues('preferred');
    if (!required.length && !preferred.length) return 'Add at least one required or preferred skill.';
    if (required.length > 100 || preferred.length > 100) return 'Each category supports up to 100 skills.';
    if (required.some((skill) => preferred.some((item) => item.toLowerCase() === skill.toLowerCase()))) {
      return 'A skill cannot be both required and preferred. Remove the duplicate.';
    }
    for (const value of [this.minimumExperience, this.maximumExperience]) {
      if (value !== null && (!Number.isFinite(value) || value < 0 || value > 60)) {
        return 'Experience must be between 0 and 60 years.';
      }
    }
    if (this.minimumExperience !== null && this.maximumExperience !== null
      && this.minimumExperience > this.maximumExperience) return 'Minimum experience cannot exceed maximum experience.';
    if (this.education.length > 500) return 'Education must be at most 500 characters.';
    if (this.lines(this.responsibilities).length > 50 || this.lines(this.additionalPreferences).length > 50) {
      return 'Responsibilities and preferences support up to 50 lines each.';
    }
    return '';
  }

  private request(): JdRankRequest {
    const request: JdRankRequest = { job_description: this.jobDescription.trim() };
    if (this.includeReasoning) request.include_reasoning = true;
    if (this.hasStructuredFields()) {
      request.structured_jd = {
        role: this.role.trim(),
        required_skills: this.skillValues('required'),
        preferred_skills: this.skillValues('preferred'),
        experience: { minimum_years: this.minimumExperience, maximum_years: this.maximumExperience },
        education: { required: this.educationRequired, description: this.education.trim() || null },
        responsibilities: this.lines(this.responsibilities),
        additional_preferences: this.lines(this.additionalPreferences),
      };
    }
    return request;
  }

  search(): void {
    if (!this.canSearch()) return;
    this.loading.set(true);
    this.error.set(null);
    this.result.set(null);
    this.api.rank(this.request()).subscribe({
      next: (res) => {
        this.result.set(res);
        this.loading.set(false);
      },
      error: (err: ApiErrorInfo) => {
        this.error.set(err);
        this.loading.set(false);
      },
    });
  }

  name(c: PoolCandidateResult): string {
    return c.source_file || c.document_id.slice(0, 12);
  }

  pct(value: number | null): string {
    return value === null ? 'n/a' : `${Math.round(value * 100)}%`;
  }

  score(value: number | null | undefined): string {
    return value == null ? 'n/a' : `${value}%`;
  }

  evidenceStatus(status: string): string {
    switch (status) {
      case 'EXPLICIT_SKILL': return 'Confirmed skill';
      case 'EXPLICIT_TEXT_EVIDENCE': return 'Confirmed text';
      case 'SEMANTIC_MATCH': return 'Semantic match — unconfirmed';
      case 'NOT_FOUND': return 'Not found';
      default: return status;
    }
  }

  skills(c: PoolCandidateResult) {
    return [...c.required_skills, ...c.preferred_skills];
  }
}
