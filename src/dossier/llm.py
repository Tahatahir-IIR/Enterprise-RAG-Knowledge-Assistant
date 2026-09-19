from __future__ import annotations

import re
from abc import ABC, abstractmethod

from .config import Settings, get_settings
from .http_retry import post_with_retry

NOT_FOUND_TOKEN = "NOT_FOUND"
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks some open models emit before the answer."""
    return _THINK_RE.sub("", text).strip()


class LLM(ABC):
    name: str = "llm"

    @abstractmethod
    def complete(self, system: str, user: str, temperature: float = 0.0) -> str: ...


class OllamaLLM(LLM):
    def __init__(self, url: str, model: str, timeout: float = 120.0):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout
        self.name = f"ollama:{model}"

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,  # qwen3 / deepseek style reasoning models: answer only
            "options": {"temperature": temperature},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        r = post_with_retry(f"{self.url}/api/chat", payload, self.timeout)
        return strip_thinking(r.json()["message"]["content"])


class OpenAICompatibleLLM(LLM):
    """Works with OpenAI, Groq, Together, vLLM's OpenAI server, LM Studio..."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0):
        self.base_url, self.api_key, self.model, self.timeout = base_url.rstrip("/"), api_key, model, timeout
        self.name = f"openai:{model}"

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        r = post_with_retry(f"{self.base_url}/chat/completions", payload, self.timeout, headers=headers)
        return strip_thinking(r.json()["choices"][0]["message"]["content"])


class StubLLM(LLM):
    """Offline stand-in for tests and demos without a model server.

    Picks the source whose sentences share the most content words with the
    question, returns its two best sentences with a citation, and answers
    NOT_FOUND when no sentence shares at least two content words.
    """

    name = "stub"

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        from .bm25_index import tokenize

        sources = re.findall(r"\[(\d+)\][^\n]*\n(.*?)(?=\n\n\[\d+\]|\n\nQUESTION:)", user, flags=re.S)
        q = re.search(r"QUESTION:\s*(.*)", user, flags=re.S)
        if not sources or not q:
            return NOT_FOUND_TOKEN
        q_words = set(tokenize(q.group(1)))
        best_n, best_sents, best_max = "1", [], 0
        for n, text in sources:
            scored = []
            for idx, sent in enumerate(re.split(r"(?<=[.!?؟])\s+|\n", text)):
                sent = sent.strip()
                overlap = len(q_words & set(tokenize(sent)))
                if sent and overlap > 0:
                    scored.append((overlap, idx, sent))
            if scored and max(s[0] for s in scored) > best_max:
                best_max = max(s[0] for s in scored)
                top = sorted(scored, key=lambda s: (-s[0], s[1]))[:2]
                best_sents = [s[2] for s in sorted(top, key=lambda s: s[1])]
                best_n = n
        if best_max < 2:
            return NOT_FOUND_TOKEN
        return " ".join(best_sents) + f" [{best_n}]"


def build_llm(settings: Settings | None = None) -> LLM:
    settings = settings or get_settings()
    if settings.llm_backend == "stub":
        return StubLLM()
    if settings.llm_backend == "openai":
        return OpenAICompatibleLLM(
            settings.openai_base_url, settings.openai_api_key, settings.openai_model, settings.llm_timeout
        )
    return OllamaLLM(settings.ollama_url, settings.ollama_model, settings.llm_timeout)
