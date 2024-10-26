from model_from_json_loader import ModelFromJSONLoader
from typing import Dict, Type, TypeVar
from dataclasses import dataclass
from openai import BaseModel
from ai import Prompt

import logging
import json

T = TypeVar('T')

@dataclass
class Personality:
    botname: str
    recent_message_history_length: int
    has_long_term_memory: bool
    prompts: Dict[str, Prompt]

class PersonalityLoader:
    def __init__(self, filename: str):
        self.filename = filename
        try:
            self.loader = ModelFromJSONLoader.from_file(filename)
        except FileNotFoundError as e:
            logging.error(f"Personality file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Personality file contains invalid JSON: {e}")
            raise

    def load_personality(self) -> Personality:
        data = self.loader.data

        bot_name = self.safe_get(
            path=["parameters", "botname"], required_type=str
        )
        recent_message_history_length = self.safe_get(
            path=["parameters", "recent_message_history_length"], required_type=int
        )
        has_long_term_memory = self.safe_get(
            path=["parameters", "has_long_term_memory"], required_type=bool
        )
        prompts: dict[str, Prompt] = ModelFromJSONLoader.from_string(json.dumps(data["prompts"])).get_dict(Prompt)
    
        return Personality(
            botname=bot_name,
            recent_message_history_length=recent_message_history_length,
            has_long_term_memory=has_long_term_memory,
            prompts=prompts
        )

    def _get_raw(self, *, path: list[str], required: bool) -> dict | None:
        current = self.loader.data
        try:
            for part in path:
                if not isinstance(current, dict):
                    raise TypeError(f"Expected dict at path {path}, but got {type(current)}")
                current = current[part]
            return current
        except KeyError as e:
            if required:
                raise KeyError(f"Required key {e.args[0]} was not present in path {str(path)}")
            return None

    def safe_get(self, *, path: list[str], required_type: Type[T]) -> T:
        current = self._get_raw(path=path, required=True)
        if current is None:
            raise KeyError(f"Could not get path {str(path)}")
            
        if not isinstance(current, required_type):
            raise TypeError(f"Expected {required_type} at path {path}, got {type(current)}")
        return current