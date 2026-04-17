"""OpenAI client wrapper with cost tracking and batching."""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .models import CostSummary


def _load_dotenv() -> None:
    """Load .env from the data directory, project root, or cwd."""
    candidates = [
        Path(os.environ.get("CODEGRAPH_DIR", "~/.codegraph")).expanduser() / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",  # project root
        Path.cwd() / ".env",
    ]
    for env_file in candidates:
        if not env_file.exists():
            continue
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
        return  # stop after first .env found

# Pricing per 1M tokens (USD)
_PRICING: dict[str, dict[str, float]] = {
    "gpt-5.4-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}


def _get_price(model: str, token_type: str) -> float:
    if model in _PRICING:
        return _PRICING[model].get(token_type, 0.0)
    return 0.0


class LLMClient:
    """Sync OpenAI client with cost tracking and batch parallelism."""

    def __init__(self) -> None:
        _load_dotenv()
        self._api_key = os.environ.get("OPENAI_API_KEY", "")
        self._client: Any = None
        self._input_tokens = 0
        self._output_tokens = 0
        self._embedding_tokens = 0

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise RuntimeError(
                    "openai package not installed. "
                    "Install with: pip install 'codegraph[llm]'"
                )
            if not self._api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not configured. "
                    "Set the environment variable to enable LLM features."
                )
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def is_available(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self._api_key)

    def get_cost_summary(self) -> CostSummary:
        total = (
            self._input_tokens * 0.15 / 1_000_000
            + self._output_tokens * 0.60 / 1_000_000
            + self._embedding_tokens * 0.02 / 1_000_000
        )
        return CostSummary(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            embedding_tokens=self._embedding_tokens,
            estimated_cost_usd=round(total, 6),
        )

    def reset_cost_tracking(self) -> CostSummary:
        summary = self.get_cost_summary()
        self._input_tokens = 0
        self._output_tokens = 0
        self._embedding_tokens = 0
        return summary

    def complete(
        self,
        messages: list[dict[str, str]],
        model: str = "gpt-5.4-mini",
        temperature: float = 0.3,
        max_tokens: int = 1000,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> dict[str, Any]:
        """Single completion call with retry and cost tracking."""
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(**kwargs)
                usage = response.usage
                if usage:
                    self._input_tokens += usage.prompt_tokens
                    self._output_tokens += usage.completion_tokens

                msg = response.choices[0].message
                result: dict[str, Any] = {"content": msg.content or ""}
                if msg.tool_calls:
                    result["tool_calls"] = [
                        {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                return result
            except Exception as exc:
                last_error = exc
                if "rate_limit" in str(exc).lower() or "429" in str(exc):
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise last_error or RuntimeError("LLM call failed after retries")

    def batch_complete(
        self,
        prompts: list[list[dict[str, str]]],
        model: str = "gpt-5.4-mini",
        max_concurrent: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Batched completions with ThreadPoolExecutor."""
        results: list[dict[str, Any] | None] = [None] * len(prompts)

        def _call(idx: int, msgs: list[dict[str, str]]) -> tuple[int, dict]:
            return idx, self.complete(msgs, model=model, **kwargs)

        with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
            futures = {
                pool.submit(_call, i, msgs): i
                for i, msgs in enumerate(prompts)
            }
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as exc:
                    idx = futures[future]
                    results[idx] = {"content": "", "error": str(exc)}

        return [r or {"content": "", "error": "unknown"} for r in results]

    def embed(
        self,
        texts: list[str],
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
    ) -> list[list[float]]:
        """Batch embedding with automatic chunking."""
        client = self._get_client()
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            response = client.embeddings.create(model=model, input=batch)
            if response.usage:
                self._embedding_tokens += response.usage.total_tokens
            for item in response.data:
                all_embeddings.append(item.embedding)

        return all_embeddings


# Module-level lazy singleton
_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Return the shared LLMClient instance."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
