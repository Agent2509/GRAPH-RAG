"""Groq Cloud LLM implementation."""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Union
from groq import Groq, RateLimitError, APIError

from graph_rag.llm.base import BaseLLM


class GroqLLM(BaseLLM):
    """Groq Cloud LLM provider using high-throughput Llama models."""

    DEFAULT_MODEL = "openai/gpt-oss-20b"
    FALLBACK_MODEL = "openai/gpt-oss-20b"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        raw_key = api_key or os.getenv("GROQ_API_KEY") or ""
        self.api_key = raw_key.strip().strip("'\"")
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Please provide an api_key argument or set GROQ_API_KEY in your environment."
            )
        self.client = Groq(api_key=self.api_key)
        self.model = model or self.DEFAULT_MODEL
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @classmethod
    def get_available_models(cls, api_key: str) -> List[str]:
        """Fetch list of chat-compatible models supported on this Groq account."""
        try:
            c = Groq(api_key=api_key)
            resp = c.models.list()
            chat_models = []
            for m in resp.data:
                mid = m.id.lower()
                # Exclude speech, guards, embeddings, and models requiring special terms
                if not any(x in mid for x in ["whisper", "guard", "embed", "safeguard", "orpheus"]):
                    chat_models.append(m.id)

            def rank_model(name: str) -> int:
                n = name.lower()
                if "3.1-8b" in n or "gpt-oss-20b" in n:
                    return 0
                if "3.3-70b" in n or "gpt-oss-120b" in n:
                    return 1
                if "compound-mini" in n:
                    return 2
                if "compound" in n:
                    return 3
                if "70b" in n:
                    return 4
                if "8b" in n:
                    return 5
                if "allam" in n:
                    return 9
                return 6

            chat_models.sort(key=lambda x: (rank_model(x), x))
            return chat_models or ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.1-8b-instant"]
        except Exception:
            return ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.1-8b-instant"]

    def _call_with_retry(self, **kwargs) -> Any:
        """Call Groq API with automatic retries for rate limits and transient errors."""
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return self.client.chat.completions.create(**kwargs)
            except RateLimitError as e:
                err_msg = str(e).lower()
                # If daily token quota (TPD) was exceeded on this model, immediately try failover models
                if "tokens per day" in err_msg or "tpd" in err_msg:
                    fallback_models = ["openai/gpt-oss-120b", "groq/compound-mini", "groq/compound"]
                    current_m = kwargs.get("model", self.model)
                    for next_m in fallback_models:
                        if next_m != current_m:
                            kwargs["model"] = next_m
                            self.model = next_m
                            try:
                                return self.client.chat.completions.create(**kwargs)
                            except Exception:
                                continue
                last_exception = e
                wait_time = self.retry_delay * (2**attempt)
                time.sleep(wait_time)
            except APIError as e:
                # If model not found or unavailable, automatically try robust fallback
                if "model_not_found" in str(e) or e.status_code == 404:
                    fallback = "openai/gpt-oss-20b" if kwargs.get("model") != "openai/gpt-oss-20b" else "openai/gpt-oss-120b"
                    if kwargs.get("model") != fallback:
                        kwargs["model"] = fallback
                        self.model = fallback
                        try:
                            return self.client.chat.completions.create(**kwargs)
                        except Exception:
                            pass
                last_exception = e
                time.sleep(self.retry_delay)
            except Exception as e:
                last_exception = e
                time.sleep(self.retry_delay)

        raise RuntimeError(f"Groq API call failed after {self.max_retries} attempts: {last_exception}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Generate text completion."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._call_with_retry(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Union[Dict[str, Any], List[Any]]:
        """Generate structured JSON output."""
        messages = []
        full_system = (system_prompt or "") + "\nYou MUST respond with valid JSON only. Do not include extra conversational text."
        messages.append({"role": "system", "content": full_system.strip()})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._call_with_retry(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception:
            # Fallback: request standard completion and extract JSON markdown block
            raw_text = self.generate(prompt=prompt, system_prompt=full_system, temperature=temperature, max_tokens=max_tokens)
            return self._extract_json_from_text(raw_text)

    def _extract_json_from_text(self, text: str) -> Union[Dict[str, Any], List[Any]]:
        """Extract JSON structure from arbitrary markdown text."""
        # Try finding markdown code block ```json ... ```
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            candidate = json_match.group(1).strip()
            try:
                return json.loads(candidate)
            except Exception:
                pass

        # Try finding outermost { ... } or [ ... ]
        curly_match = re.search(r"(\{[\s\S]*\})", text)
        if curly_match:
            try:
                return json.loads(curly_match.group(1))
            except Exception:
                pass

        square_match = re.search(r"(\[[\s\S]*\])", text)
        if square_match:
            try:
                return json.loads(square_match.group(1))
            except Exception:
                pass

        # If all else fails, return empty dictionary
        return {}
