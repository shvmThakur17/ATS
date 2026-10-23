import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from app.services.ai.prompt_builder import PromptBuilder
from app.services.ai.ollama_service import OllamaService

job_description = """
Senior .NET Developer
Angular
Azure
SQL Server
REST API
"""

resume = """
5 years experience in Angular.
Worked with REST APIs.
Experience in .NET Core.
Knowledge of SQL Server.
"""

prompt = PromptBuilder.build_resume_analysis_prompt(
    job_description,
    resume
)

result = OllamaService.analyze(prompt)

print(result)