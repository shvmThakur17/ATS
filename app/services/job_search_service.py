from app.services.embedding_service import EmbeddingService
from app.services.pinecone_service import PineconeService
from app.services.candidate_ranking_service import CandidateRankingService
from app.services.ai.prompt_builder import PromptBuilder
from app.services.ai.ollama_service import OllamaService


class JobSearchService:

    @staticmethod
    def search_candidates(job_description: str):

        embedding = EmbeddingService.generate(job_description)

        result = PineconeService.search_by_embedding(embedding=embedding,top_k=50)

        ranked_candidates = CandidateRankingService.rank(result["matches"])

        top_candidates = ranked_candidates[:5]

        analyzed_candidates = []

        for candidate in top_candidates:

            resume_text = "\n\n".join(
                chunk["text"]
                for chunk in candidate["chunks"][:3]
            )

            prompt = PromptBuilder.build_resume_analysis_prompt(
                job_description=job_description,
                resume_text=resume_text
            )

            print("\n==============================")
            print("CANDIDATE:", candidate["file_name"])
            print("MATCHED CHUNKS:", candidate["matched_chunks"])

            for i, chunk in enumerate(candidate["chunks"][:3]):
                print(f"\nCHUNK {i + 1}")
                print(chunk["text"])
                print("SCORE:", chunk["score"])

            print("==============================\n")
            analysis = OllamaService.analyze(prompt)

            analyzed_candidates.append({
                "candidate": {
                    "file_name": candidate["file_name"],
                    "average_score": candidate["average_score"],
                    "best_score": candidate["best_score"],
                    "matched_chunks": candidate["matched_chunks"]
                },
                "analysis": analysis
            })

        return {
            "total_candidates": len(analyzed_candidates),
            "candidates": analyzed_candidates
        }