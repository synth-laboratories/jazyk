import json
import os
from typing import Any, Dict, List, Tuple, Type

import pydantic
from mistralai import Mistral  # use Mistral as both sync and async client
from pydantic import BaseModel

from synth_ai.zyk.lms.caching.initialize import get_cache_handler
from synth_ai.zyk.lms.vendors.base import VendorBase
from synth_ai.zyk.lms.vendors.constants import SPECIAL_BASE_TEMPS
from synth_ai.zyk.lms.vendors.core.openai_api import OpenAIStructuredOutputClient
from synth_ai.zyk.lms.vendors.retries import BACKOFF_TOLERANCE, backoff

# Since the mistralai package doesn't expose an exceptions module,
# we fallback to catching all Exceptions for retry.
MISTRAL_EXCEPTIONS_TO_RETRY: Tuple[Type[Exception], ...] = (Exception,)


class MistralAPI(VendorBase):
    used_for_structured_outputs: bool = True
    exceptions_to_retry: Tuple = MISTRAL_EXCEPTIONS_TO_RETRY
    _openai_fallback: Any

    def __init__(
        self,
        exceptions_to_retry: Tuple[Type[Exception], ...] = MISTRAL_EXCEPTIONS_TO_RETRY,
        used_for_structured_outputs: bool = False,
    ):
        self.used_for_structured_outputs = used_for_structured_outputs
        self.exceptions_to_retry = exceptions_to_retry
        self._openai_fallback = None

    @backoff.on_exception(
        backoff.expo,
        MISTRAL_EXCEPTIONS_TO_RETRY,
        max_tries=BACKOFF_TOLERANCE,
        on_giveup=lambda e: print(e),
    )
    async def _hit_api_async(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        lm_config: Dict[str, Any],
        use_ephemeral_cache_only: bool = False,
    ) -> str:
        assert (
            lm_config.get("response_model", None) is None
        ), "response_model is not supported for standard calls"
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only)
        cache_result = used_cache_handler.hit_managed_cache(
            model, messages, lm_config=lm_config
        )
        if cache_result:
            return (
                cache_result["response"]
                if isinstance(cache_result, dict)
                else cache_result
            )

        mistral_messages = [
            {"role": msg["role"], "content": msg["content"]} for msg in messages
        ]
        async with Mistral(api_key=os.getenv("MISTRAL_API_KEY", "")) as client:
            response = await client.chat.complete_async(
                model=model,
                messages=mistral_messages,
                max_tokens=lm_config.get("max_tokens", 4096),
                temperature=lm_config.get(
                    "temperature", SPECIAL_BASE_TEMPS.get(model, 0)
                ),
                stream=False,
            )
        api_result = response.choices[0].message.content
        used_cache_handler.add_to_managed_cache(
            model, messages, lm_config=lm_config, output=api_result
        )
        return api_result

    @backoff.on_exception(
        backoff.expo,
        MISTRAL_EXCEPTIONS_TO_RETRY,
        max_tries=BACKOFF_TOLERANCE,
        on_giveup=lambda e: print(e),
    )
    def _hit_api_sync(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        lm_config: Dict[str, Any],
        use_ephemeral_cache_only: bool = False,
    ) -> str:
        assert (
            lm_config.get("response_model", None) is None
        ), "response_model is not supported for standard calls"
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only)
        cache_result = used_cache_handler.hit_managed_cache(
            model, messages, lm_config=lm_config
        )
        if cache_result:
            return (
                cache_result["response"]
                if isinstance(cache_result, dict)
                else cache_result
            )

        mistral_messages = [
            {"role": msg["role"], "content": msg["content"]} for msg in messages
        ]
        with Mistral(api_key=os.getenv("MISTRAL_API_KEY", "")) as client:
            response = client.chat.complete(
                model=model,
                messages=mistral_messages,
                max_tokens=lm_config.get("max_tokens", 4096),
                temperature=lm_config.get(
                    "temperature", SPECIAL_BASE_TEMPS.get(model, 0)
                ),
                stream=False,
            )
        api_result = response.choices[0].message.content
        used_cache_handler.add_to_managed_cache(
            model, messages, lm_config=lm_config, output=api_result
        )
        return api_result

    async def _hit_api_async_structured_output(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response_model: BaseModel,
        temperature: float,
        use_ephemeral_cache_only: bool = False,
    ) -> Any:
        try:
            mistral_messages = [
                {"role": msg["role"], "content": msg["content"]} for msg in messages
            ]
            async with Mistral(api_key=os.getenv("MISTRAL_API_KEY", "")) as client:
                response = await client.chat.complete_async(
                    model=model,
                    messages=mistral_messages,
                    max_tokens=4096,
                    temperature=temperature,
                    stream=False,
                )
            result = response.choices[0].message.content
            parsed = json.loads(result)
            return response_model(**parsed)
        except (json.JSONDecodeError, pydantic.ValidationError):
            if self._openai_fallback is None:
                self._openai_fallback = OpenAIStructuredOutputClient()
            return await self._openai_fallback._hit_api_async_structured_output(
                model="gpt-4o",
                messages=messages,
                response_model=response_model,
                temperature=temperature,
                use_ephemeral_cache_only=use_ephemeral_cache_only,
            )

    def _hit_api_sync_structured_output(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response_model: BaseModel,
        temperature: float,
        use_ephemeral_cache_only: bool = False,
    ) -> Any:
        try:
            mistral_messages = [
                {"role": msg["role"], "content": msg["content"]} for msg in messages
            ]
            with Mistral(api_key=os.getenv("MISTRAL_API_KEY", "")) as client:
                response = client.chat.complete(
                    model=model,
                    messages=mistral_messages,
                    max_tokens=4096,
                    temperature=temperature,
                    stream=False,
                )
            result = response.choices[0].message.content
            parsed = json.loads(result)
            return response_model(**parsed)
        except (json.JSONDecodeError, pydantic.ValidationError):
            print("WARNING - Falling back to OpenAI - THIS IS SLOW")
            if self._openai_fallback is None:
                self._openai_fallback = OpenAIStructuredOutputClient()
            return self._openai_fallback._hit_api_sync_structured_output(
                model="gpt-4o",
                messages=messages,
                response_model=response_model,
                temperature=temperature,
                use_ephemeral_cache_only=use_ephemeral_cache_only,
            )


if __name__ == "__main__":
    import asyncio

    from pydantic import BaseModel

    class TestModel(BaseModel):
        name: str

    client = MistralAPI(used_for_structured_outputs=True, exceptions_to_retry=[])
    import time

    t = time.time()

    async def run_async():
        response = await client._hit_api_async_structured_output(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "What is the capital of the moon?"}],
            response_model=TestModel,
            temperature=0.0,
        )
        print(response)
        return response

    response = asyncio.run(run_async())
    t2 = time.time()
    print(f"Got {len(response.name)} chars in {t2-t} seconds")
