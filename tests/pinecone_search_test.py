import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.services.pinecone_service import PineconeService

result = PineconeService.search_chunks(
    ".NET Core Angular Developer"
)
for match in result["matches"]:
    print(match["score"])
    print(match["metadata"]["file_name"])
    print(match["metadata"]["text"])
    print("-" * 50)
#print(result)