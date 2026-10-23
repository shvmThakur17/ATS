from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

resume = """
7 years experience in .NET Core, Angular, SQL Server, Azure
"""

jd = """
Looking for a Senior .NET Core Developer with Angular experience
"""

resume_embedding = model.encode(resume)
jd_embedding = model.encode(jd)

score = cosine_similarity(
    [resume_embedding],
    [jd_embedding]
)

print(f"Similarity Score: {score[0][0]}")