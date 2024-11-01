from pydantic import BaseModel, Field, model_validator
from util.environment_vars import get_environment_var
from typing import List

class ProviderData(BaseModel):
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

class ProviderDataStore:
    def __init__(self, *, providers: List[ProviderData], optional_provider_names: List[str] = []):
        self.providers: dict[str, ProviderData] = {p.provider_name: p for p in providers}
        self.optional_provider_names = optional_provider_names

    def get_provider_by_name(self, name: str) -> ProviderData | None:
        provider = self.providers.get(name)
        if not provider and name not in self.optional_provider_names:
            raise ValueError(f"Missing required provider named {name}. Please ensure it is set in the personality JSON.")
        return provider