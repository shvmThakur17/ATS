import { PoolRankResponse, ResumeUploadResponse } from './models';

export const HEX_A = 'a'.repeat(64);
export const HEX_B = 'b'.repeat(64);

export const uploadResponse = (over: Partial<ResumeUploadResponse> = {}): ResumeUploadResponse => ({
  status: 'INGESTED', ready: true, document_id: HEX_A, namespace: 'ns', chunk_count: 5, vectors_upserted: 5,
  page_count: 2, replaced_existing: false, embedding_model: 'm', embedding_dimension: 3, access_token: 't'.repeat(64),
  ...over,
});

export const rankResponse = (over: Partial<PoolRankResponse> = {}): PoolRankResponse => ({
  ranking_basis: 'skills-only',
  jd: { job_id: 'j', title: 'Dev', required_skills: ['Python'], preferred_skills: ['Azure'] },
  retrieval: { effective_top_k: 10, max_retrieval_depth: 200 },
  pool: { discovered: 2, considered: 2, ranked: 2, skipped: 0, truncated: false, max_considered: 100 },
  skipped: [],
  candidates: [
    {
      rank: 1, document_id: HEX_A, source_file: 'jane.pdf', required_skill_coverage: 1, preferred_skill_coverage: 0,
      ranking_key: [1, 0, 0, 0],
      required_skills: [{
        skill: 'Python', requirement: 'REQUIRED', status: 'EXPLICIT_SKILL', covered: true,
        evidence: { kind: 'SKILLS_LIST', canonical_path: 'skills[0]', matched_term: 'Python', excerpt: 'Python, SQL',
                    page_numbers: ['1'], chunk_id: 'c1' },
      }],
      preferred_skills: [{ skill: 'Azure', requirement: 'PREFERRED', status: 'NOT_FOUND', covered: false, evidence: null }],
      missing_skills: { required: [], preferred: ['Azure'] }, warnings: [],
    },
    {
      rank: 2, document_id: HEX_B, source_file: null, required_skill_coverage: 0, preferred_skill_coverage: null,
      ranking_key: [0, 0, 0, 0], required_skills: [], preferred_skills: [],
      missing_skills: { required: ['Python'], preferred: [] }, warnings: ['NO_RESUME_VECTORS'],
    },
  ],
  ...over,
});

export const pdf = (name: string) => new File(['%PDF-1.4'], name, { type: 'application/pdf' });
export const tick = () => new Promise<void>((resolve) => setTimeout(resolve));
