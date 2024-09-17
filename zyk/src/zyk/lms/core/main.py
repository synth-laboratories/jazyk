from typing import List, Dict, Optional, Literal, Any
from zyk.src.zyk.lms.core.vendor_clients import anthropic_naming_regexes, openai_naming_regexes, get_client
from zyk.src.zyk.lms.caching.handler import CacheHandler
from zyk.src.zyk.lms.structured_outputs.handler import StructuredOutputHandler

from pydantic import BaseModel

def build_messages(
    sys_msg: str,
    user_msg: str,
    images_bytes: List = [],
    model_name: Optional[str] = None,
) -> List[Dict]:
    if len(images_bytes) > 0 and any(regex.match(model_name) for regex in openai_naming_regexes):
        return [
            {"role": "system", "content": sys_msg},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_msg}]
                + [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_bytes}"},
                    }
                    for image_bytes in images_bytes
                ],
            },
        ]
    elif len(images_bytes) > 0 and any(regex.match(model_name) for regex in anthropic_naming_regexes):
        system_info = {"role": "system", "content": sys_msg}
        user_info = {
            "role": "user",
            "content": [{"type": "text", "text": user_msg}]
            + [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_bytes,
                    },
                }
                for image_bytes in images_bytes
            ],
        }
        return [system_info, user_info]
    elif len(images_bytes) > 0:
        raise ValueError("Images are not yet supported for this model")
    else:
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]


class LM:
    # if str
    model_name: str
    structured_output_fallback_model_name: str = 'gpt-4o-mini'
    
    def __init__(self, model_name: str, formatting_model_name: str, temperature: float, max_retries: Literal["None","Few","Many"], structured_output_mode: Literal["stringified_json","forced_json"]):
        self.client = get_client(model_name, with_formatting= structured_output_mode == "forced_json")
        formatting_client = get_client(formatting_model_name, with_formatting=True)

        max_retries_dict = {"None": 0, "Few": 2, "Many": 5}
        self.structured_output_handler = StructuredOutputHandler(self.client, formatting_client, structured_output_mode, {"max_retries": max_retries_dict.get(max_retries, 2)})
        self.lm_config = {"temperature": temperature}
        self.model_name = model_name
    
    def respond_sync(self, system_message: str, user_message: str, images_as_bytes: List[Any], response_model: Optional[BaseModel] = None, use_ephemeral_cache: bool = False):
        messages = build_messages(system_message, user_message, images_as_bytes, self.model_name)
        if response_model:
            return self.structured_output_handler.call_sync(messages, model=self.model_name, response_model=response_model, use_ephemeral_cache=use_ephemeral_cache)
        else:
            return self.client._hit_api_sync(messages=messages, model=self.model_name, lm_config=self.lm_config, use_ephemeral_cache=use_ephemeral_cache)
    
    async def respond_async(self, system_message: str, user_message: str, images_as_bytes: List[Any], response_model: Optional[BaseModel] = None, use_ephemeral_cache: bool = False):
        messages = build_messages(system_message, user_message, images_as_bytes, self.model_name)
        if response_model:
            return await self.structured_output_handler.call_async(messages, model=self.model_name, response_model=response_model, use_ephemeral_cache=use_ephemeral_cache)
        else:
            return await self.client._hit_api_async(messages=messages, model=self.model_name, lm_config=self.lm_config, use_ephemeral_cache=use_ephemeral_cache)
        
if __name__ == "__main__":
    import asyncio
    class TestModel(BaseModel):
        thought: str
        emotion: str
        concern: str
        action: str
    lm = LM(model_name="gpt-4o-mini", formatting_model_name="gpt-4o-mini", temperature=0.0, max_retries="Few", structured_output_mode="stringified_json")
    print(lm.respond_sync(system_message="You are a helpful  assistant", user_message="Hello, how are you?", images_as_bytes=[], response_model=None, use_ephemeral_cache=False))
    print(asyncio.run(lm.respond_async(system_message="You are a  helpful assistant", user_message="Hello, how are you?", images_as_bytes=[], response_model=None, use_ephemeral_cache=False)))
    print(asyncio.run(lm.respond_async(system_message="You are  a  helpful assistant", user_message="Hello, how are you?", images_as_bytes=[], response_model=TestModel, use_ephemeral_cache=False)))
    lm = LM(model_name="o1-mini", formatting_model_name="gpt-4o-mini", temperature=1, max_retries="Few", structured_output_mode="stringified_json")
    print(lm.respond_sync(system_message="You are a helpful assistant", user_message="Hello, how are you?", images_as_bytes=[], response_model=None, use_ephemeral_cache=False))
    print(asyncio.run(lm.respond_async(system_message="You are a helpful assistant", user_message="Hello, how are you?", images_as_bytes=[], response_model=None, use_ephemeral_cache=False)))
    print(asyncio.run(lm.respond_async(system_message="You are a  helpful assistant", user_message="Hello, how are you?", images_as_bytes=[], response_model=TestModel, use_ephemeral_cache=False)))