from app.core.config import settings
from pinecone import Pinecone
from app.services.embedding_service import EmbeddingService
class PineconeService:

    @staticmethod
    def get_index():
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        return pc.Index(settings.PINECONE_INDEX_NAME)   
    @staticmethod
    def health_check():
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)

        index = pc.Index(settings.PINECONE_INDEX_NAME)

        return index.describe_index_stats()
    @staticmethod
    def upsert_test_vector():
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)

        index = pc.Index(settings.PINECONE_INDEX_NAME)

        index.upsert(
            vectors=[
                {
                    "id": "test-1",
                    "values": [0.1] * 768
                }
            ]
        )
    
        index = PineconeService.get_index()
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vectors.append(
                {
                    "id": f"{file_name}_{i}",
                    "values": embedding,
                    "metadata": {
                        "file_name": file_name,
                        "chunk_index": i,
                        "text": chunk
                    }
                }
            )
        index.upsert(vectors=vectors)
        return len(vectors)
    @staticmethod
    def search_chunks(query: str, top_k: int = 5):
        index = PineconeService.get_index()

        query_embedding = EmbeddingService.generate(query)

        result = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        return result
    @staticmethod
    def search_by_embedding(embedding: list[float],top_k: int = 10):
        index = PineconeService.get_index()
        return index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )
    @staticmethod
    def upsert_chunks(
        file_name: str,
        chunks: list[dict],
        embeddings: list[list[float]]
    ):

        index = PineconeService.get_index()

        vectors = []

        for i, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):

            vectors.append({
                "id": f"{file_name}_{i}",
                "values": embedding,
                "metadata": {
                    "file_name": file_name,
                    "chunk_index": i,
                    "section": chunk["section"],
                    "subsection": chunk["subsection"] or "",
                    "text": chunk["text"]
                }
            })
        print("\nPINECONE VECTORS:")
        print(vectors)
        index.upsert(vectors=vectors)

        return len(vectors)