from pydantic import BaseModel, Field, model_validator
from typing import List
from environment_vars import get_environment_var
import json

class Provider(BaseModel):
    provider_name: str
    api_base: str = Field(default='https://api.openai.com/v1')
    api_key: str

    @model_validator(mode='before')
    @classmethod
    def check_and_load_api_key(cls, values):
        api_key = values.get('api_key')
        if api_key and api_key.startswith('[') and api_key.endswith(']'):
            env_var_name = api_key[1:-1]
            loaded_api_key = get_environment_var(env_var_name, required=True)
            if not loaded_api_key:
                raise ValueError(f"Environment variable {env_var_name} not set for API key.")
            values['api_key'] = loaded_api_key
        return values

class ProviderStore:
    def __init__(self, providers: List[Provider]):
        self.providers = providers

    @classmethod
    def from_environment_var(cls, environment_var: str) -> "ProviderStore":
        return ProviderStore.load_from_json(get_environment_var(environment_var, required=True))

    @classmethod
    def load_from_json(cls, providers_json: str):
        return ProviderStore(ProviderStore._load_provder_list_from_json(providers_json))

    @staticmethod
    def _load_provder_list_from_json(providers_json: str) -> List[Provider]:
        try:
            providers_data = json.loads(providers_json)
            if not isinstance(providers_data, list):
                raise ValueError("Providers data must be a list of JSON objects.")
            providers = [Provider(**provider_dict) for provider_dict in providers_data]
            return providers 
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(
                "Invalid providers JSON, must be a list of JSON objects containing provider_name, api_base, api_key."
            ) from e

    def get_provider_by_name(self, name: str) -> Provider:
        for provider in self.providers:
            if provider.provider_name == name:
                return provider
        raise RuntimeError(f"Missing provider named {name}, please add it to the providers list")