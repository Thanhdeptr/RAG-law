from typing import Iterable, List
import httpx


class OllamaEmbeddingsClient:
    def __init__(self, base_url: str, model: str, timeout_seconds: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        inputs = list(texts)
        if not inputs:
            return []
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model, "input": inputs}
        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        # OpenAI-compatible format: data: [{embedding: [...]}]
        return [row["embedding"] for row in data.get("data", [])]


class OllamaChatClient:
    def __init__(self, base_url: str, model: str, timeout_seconds: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def chat(self, messages: List[dict], temperature: float = 0.2, max_tokens: int | None = None) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        # OpenAI-compatible: choices[0].message.content
        choices = data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return message.get("content", "")

