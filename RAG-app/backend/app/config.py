import os
from dotenv import load_dotenv


load_dotenv()


def get_env(name: str, default: str | None = None) -> str | None:
    if name in os.environ:
        return os.environ[name]
    # Allow reading front-end style vars for convenience
    if name == "OLLAMA_BASE":
        return os.environ.get("VITE_OLLAMA_BASE", default)
    if name == "GENERATION_MODEL":
        return os.environ.get("VITE_MODEL_NAME", default)
    return default if default is not None else None


class Settings:
    def __init__(self) -> None:
        self.ollama_base: str = get_env("OLLAMA_BASE", "http://localhost:11434/v1")  # type: ignore[assignment]
        self.generation_model: str = get_env("GENERATION_MODEL", "gpt-oss:20b")  # type: ignore[assignment]
        self.embedding_model: str = get_env("EMBEDDING_MODEL", "nomic-embed-text")  # type: ignore[assignment]
        self.chroma_persist_dir: str = get_env("CHROMA_PERSIST_DIR", "./storage/chroma")  # type: ignore[assignment]


settings = Settings()

