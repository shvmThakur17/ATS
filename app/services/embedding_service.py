from sentence_transformers import SentenceTransformer


class EmbeddingService:

    _model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5",trust_remote_code=True)
    @classmethod
    def generate(cls, text: str):
        return cls._model.encode(text).tolist()