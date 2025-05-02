import json
import logging

from typing import Type, TypeVar
from dataclasses import dataclass
from ai_apis.providers import ProviderData
from pydantic import BaseModel, model_validator
from ai_apis.types import LLMRequestParams, Prompt
from util.environment_vars import parse_api_key_in_config
from util.model_from_json_loader import ModelFromJSONLoader
from pydantic._internal._model_construction import ModelMetaclass

T = TypeVar('T')

class FalImageGenModuleConfig(BaseModel):
    enabled: bool
    model_name: str
    n_images: int
    allow_nsfw: bool
    api_key: str

    @model_validator(mode='before')
    @classmethod
    def check_and_load_api_key(cls, values):
        if 'api_key' in values:
            values['api_key'] = parse_api_key_in_config(values['api_key'])
        return values

@dataclass
class Profile:
    botname: str
    recent_message_history_length: int
    has_long_term_memory: bool
    prompts: dict[str, Prompt]
    request_params: dict[str, LLMRequestParams]
    lang: dict[str, str]
    providers: dict[str, ProviderData]
    regex_replacements: dict[str, str]
    fal_image_gen_config: FalImageGenModuleConfig
    llm_fallbacks: list[str]

class ProfileLoader:
    def __init__(self, filename: str):
        self.filename = filename
        try:
            self.loader = ModelFromJSONLoader.from_file(filename)
        except FileNotFoundError as e:
            logging.error(f"Profile file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Profile file contains invalid JSON: {e}")
            raise

    def load_profile(self) -> Profile:
        bot_name = self.safe_get(
            path=["profile", "parameters", "botname"], required_type=str
        )
        recent_message_history_length = self.safe_get(
            path=["profile", "parameters", "recent_message_history_length"], required_type=int
        )
        has_long_term_memory = self.safe_get(
            path=["profile", "parameters", "has_long_term_memory"], required_type=bool
        )
        llm_fallbacks = self.safe_get(
            path=["profile", "parameters", "llm_fallbacks"], required_type=list[str]
        )
        lang = self.safe_get(
            path=["profile", "lang"], required_type=dict
        )
        regex_replacements = self.safe_get(
            path=["profile", "regex_replacements"], required_type=dict
        )
        prompts: dict[str, Prompt] = self.safe_get_dict_of_model(
            path=["profile", "prompts"], required_type=Prompt
        )
        request_params: dict[str, LLMRequestParams] = self.safe_get_dict_of_model(
            path=["profile", "request_params"], required_type=LLMRequestParams
        )
        providers: dict[str, ProviderData] = self.safe_get_dict_of_model(
            path=["profile", "providers"], required_type=ProviderData
        )
        fal_image_gen_config: FalImageGenModuleConfig = self.safe_get_model(
            path=["profile", 'fal_image_gen_config'], required_type=FalImageGenModuleConfig
        )
        return Profile(
            botname=bot_name,
            recent_message_history_length=recent_message_history_length,
            has_long_term_memory=has_long_term_memory,
            prompts=prompts,
            lang=lang,
            request_params=request_params,
            providers=providers,
            regex_replacements=regex_replacements,
            fal_image_gen_config=fal_image_gen_config,
            llm_fallbacks=llm_fallbacks
        )

    def _get_raw(self, *, path: list[str], required: bool) -> dict | None:
        """
        Retrieves a dictionary from the loader's JSON data using the provided path.
        
        This method the JSON structure by following the sequence of keys in the path list,
        where each element of the list will access a key from the dictionary, descending deeper.
        
        Args:
            path (list[str]): A list of string keys representing the path to traverse,
                e.g. ['personality', 'lang', 'error'] would access data['personality']['lang']['error']
            required (bool): If True, raises KeyError when path doesn't exist.
                If False, returns None for missing paths.
                
        Returns:
            dict | None: The dictionary found at the specified path, or None if the path
                doesn't exist and required=False
                
        Raises:
            KeyError: If required=True and any key in the path doesn't exist
            TypeError: If any intermediate node in the path is not a dictionary
        """
        current = self.loader.data
        try:
            for part in path:
                if not isinstance(current, dict):
                    raise TypeError(f"Expected dict at path {path}, but got {type(current)}")
                current = current[part]
            return current
        except KeyError:
            if required:
                raise KeyError(f"Error when trying to load key '{path[-1]}', it was not found in the profile JSON. Make sure the key is present under {str(path)}")
            return None

    def safe_get_dict_of_model(self, *, path: list[str], required_type: Type[T]) -> dict[str, T]:
        """
        Safely retrieves and converts a dictionary from the JSON structure into a dictionary of Pydantic models.
        
        Retrieves data at the specified path and converts it into a dictionary where values
        are instances of the specified Pydantic model type.
        
        Args:
            path (list[str]): A list of string keys representing the path to traverse
            required_type (Type[T]): The Pydantic model class that values in the dictionary
                should be converted to. Must be a subclass of BaseModel.
                
        Returns:
            dict[str, T]: A dictionary where all values are instances of required_type
            
        Raises:
            KeyError: If the path doesn't exist in the JSON structure
            ValueError: If required_type is not a subclass of BaseModel
            JSONDecodeError: If the data cannot be converted to JSON string
            
        Example:
            >>> safe_get_dict_of_model(path=['personality', 'prompts'], required_type=PromptConfig)
            {'REWRITER': PromptConfig(...), 'KAMI_CHAN': PromptConfig(...)}
        """
        current = self._get_raw(path=path, required=True)
        if current is None:
            raise KeyError(f"Could not get path {str(path)}")

        if not isinstance(required_type, ModelMetaclass):
            raise ValueError(f"Requested type {required_type} is not a valid Pydantic BaseModel")

        json_as_str = json.dumps(current) # TODO: redundant conversion to string
        return ModelFromJSONLoader.from_string(json_as_str).get_dict(required_type)

    def safe_get(self, *, path: list[str], required_type: Type[T]) -> T:
        """
        Safely retrieves and type-checks a value from the JSON structure.
        
        Gets the value at the specified path and verifies it matches the required type.
        
        Args:
            path (list[str]): A list of string keys representing the path to traverse
            required_type (Type[T]): The expected type of the value at the specified path
                
        Returns:
            T: The value at the specified path, guaranteed to be of type T
            
        Raises:
            KeyError: If the path doesn't exist in the JSON structure
            TypeError: If the value's type doesn't match required_type
            
        Example:
            >>> safe_get(path=['personality', 'parameters', 'botname'], required_type=str)
            'MySmartAIBot'
            >>> safe_get(path=['personality', 'parameters', 'has_long_term_memory'], required_type=bool)
            True
        """
        current = self._get_raw(path=path, required=True)
        if current is None:
            raise KeyError(f"Could not get path {str(path)}")
            
        if not isinstance(current, required_type):
            raise TypeError(f"Expected {required_type} at path {path}, got {type(current)}")
        return current
    
    def safe_get_model(self, *, path: list[str], required_type: Type[T]) -> T:
        """
        Safely retrieves and converts a dictionary from the JSON structure into a Pydantic model.
        
        Retrieves data at the specified path and converts it into an instance of the specified
        Pydantic model type.
        
        Args:
            path (list[str]): A list of string keys representing the path to traverse.
            required_type (Type[T]): The Pydantic model class that the dictionary should be
                converted to. Must be a subclass of BaseModel.
                
        Returns:
            T: An instance of required_type representing the data at the specified path.
            
        Raises:
            KeyError: If the path doesn't exist in the JSON structure.
            ValueError: If required_type is not a subclass of BaseModel.
            JSONDecodeError: If the data cannot be converted to a JSON string.
        """
        current = self._get_raw(path=path, required=True)
        if current is None:
            raise KeyError(f"Could not get path {str(path)}")

        if not isinstance(required_type, ModelMetaclass):
            raise ValueError(f"Requested type {required_type} is not a valid Pydantic BaseModel")

        json_as_str = json.dumps(current) # TODO: redundant conversion to string
        return ModelFromJSONLoader.from_string(json_as_str).from_string(json_as_str).get_model(required_type)