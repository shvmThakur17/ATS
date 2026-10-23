// Mirrors app/schemas/pipeline_api.py
export interface ResumeUploadResponse {
  status: 'INGESTED' | 'PENDING_INDEXING';
  ready: boolean;
  document_id: string;
  namespace: string;
  chunk_count: number;
  vectors_upserted: number;
  page_count: number;
  replaced_existing: boolean | null;
  embedding_model: string;
  embedding_dimension: number;
  access_token: string;
}

export interface JdRankRequest {
  job_description?: string;
  structured_jd?: StructuredJD;
  include_reasoning?: boolean;
}

export interface StructuredJD {
  role: string;
  required_skills: string[];
  preferred_skills: string[];
  experience: { minimum_years: number | null; maximum_years: number | null };
  education: { required: boolean; description: string | null };
  responsibilities: string[];
  additional_preferences: string[];
}

export interface CandidateReasoning {
  status: 'AVAILABLE' | 'UNAVAILABLE' | 'NOT_REQUESTED';
  strengths: string[];
  gaps: string[];
  experience_summary: string;
  recommendation: string;
}

export interface SkillEvidence {
  kind: string | null;
  canonical_path: string | null;
  matched_term: string | null;
  excerpt: string | null;
  page_numbers: string[];
  chunk_id: string | null;
}

export interface SkillResult {
  skill: string;
  requirement: 'REQUIRED' | 'PREFERRED';
  status: string;
  covered: boolean;
  evidence: SkillEvidence | null;
}

export interface CandidateResult {
  rank: number;
  document_id: string;
  required_skill_coverage: number | null;
  preferred_skill_coverage: number | null;
  ranking_key: number[];
  required_skills: SkillResult[];
  preferred_skills: SkillResult[];
  missing_skills: { required: string[]; preferred: string[] };
  warnings: string[];
}

export interface PoolCandidateResult extends CandidateResult {
  source_file: string | null;
  required_skill_points?: number | null;
  preferred_skill_points?: number | null;
  overall_match_score?: number | null;
  reasoning?: CandidateReasoning;
}

export interface SkippedResume {
  document_id: string;
  reason: 'EMPTY' | 'STALE' | 'TOO_LARGE';
}

export interface PoolSummary {
  discovered: number;
  considered: number;
  ranked: number;
  skipped: number;
  truncated: boolean;
  max_considered: number;
}

/** POST /api/v1/jd/rank: the server discovers candidates in the shared pool. */
export interface PoolRankResponse {
  ranking_basis: 'skills-only';
  jd: { job_id: string; title: string | null; required_skills: string[]; preferred_skills: string[] };
  retrieval: { effective_top_k: number; max_retrieval_depth: number };
  pool: PoolSummary;
  skipped: SkippedResume[];
  candidates: PoolCandidateResult[];
}

/** Backend error envelope is {"error": {"code", "message", "details"?}}; this is the normalized form. */
export interface ApiErrorInfo {
  status: number;
  code: string;
  message: string;
}
