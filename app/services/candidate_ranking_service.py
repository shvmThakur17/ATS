class CandidateRankingService:

    @staticmethod
    def rank(matches):

        candidates = {}

        for match in matches:

            metadata = match.get("metadata")

            if not metadata:
                continue

            file_name = metadata["file_name"]

            candidates.setdefault(
                file_name,
                {
                    "file_name": file_name,
                    "scores": [],
                    "chunks": []
                }
            )

            candidates[file_name]["scores"].append(match["score"])

            candidates[file_name]["chunks"].append(
                {
                    "text": metadata["text"],
                    "score": match["score"]
                }
            )

        results = []

        for candidate in candidates.values():

            candidate["chunks"] = sorted(
                candidate["chunks"],
                key=lambda x: x["score"],
                reverse=True
            )

            avg_score = sum(candidate["scores"]) / len(candidate["scores"])
            best_score = max(candidate["scores"])

            results.append(
                {
                    "file_name": candidate["file_name"],
                    "average_score": round(avg_score * 100, 2),
                    "best_score": round(best_score * 100, 2),
                    "matched_chunks": len(candidate["scores"]),
                    "chunks": candidate["chunks"]
                }
            )

        return sorted(
            results,
            key=lambda x: x["best_score"],
            reverse=True
        )