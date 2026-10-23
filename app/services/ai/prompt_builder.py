class PromptBuilder:

    @staticmethod
    def build_resume_analysis_prompt(
        job_description: str,
        resume_text: str
    ) -> str:

        return f"""
Analyze the resume against the job description.

JD:
{job_description}

RESUME:
{resume_text}

Return ONLY valid JSON:

{{
  "match_score": 0,
  "strengths": [],
  "missing_skills": [],
  "experience_summary": "",
  "recommendation": ""
}}

Rules:
- Score: 0-100.
- Use only information from the resume.
- strengths: max 3.
- missing_skills: max 3.
- experience_summary: max 15 words.
- recommendation: Strong Match, Good Match, Average Match, or Weak Match.
- No explanation. No markdown. No extra fields.
"""