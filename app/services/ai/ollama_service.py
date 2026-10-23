import json
import requests

from app.core.config import settings


class OllamaService:

    @staticmethod
    def analyze(prompt: str) -> dict:

        url = f"{settings.OLLAMA_BASE_URL}/api/generate"

        payload = {
            "model": settings.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": 150
            }
        }

        try:
            response = requests.post(url=url, json=payload, timeout=(3, 10))

            response.raise_for_status()

            result = response.json()

            raw_response = (
                result.get("response")
                or result.get("thinking")
                or ""
            ).strip()

            if not raw_response:
                raise Exception("Ollama returned empty response.")

            return json.loads(raw_response)

        except json.JSONDecodeError as ex:
            raise Exception(
                f"Invalid JSON returned from Ollama: {ex}"
            )

        except requests.RequestException as ex:
            raise Exception(
                f"Ollama connection failed: {ex}"
            )