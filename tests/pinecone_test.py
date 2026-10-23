import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.services.pinecone_service import PineconeService

# PineconeService.upsert_test_vector()
result = PineconeService.health_check()

print(result)
