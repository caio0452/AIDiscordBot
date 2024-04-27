from dataclasses import dataclass
from environment_vars import get_environment_var
import json

ENVIRONMENT_VAR_NAME = "API_PROVIDERS"

@dataclass
class Provider:
    provider_name: str
    api_base: str
    api_key: str

def _parse_providers(providers_json: str) -> list[Provider]:
    providers_data = json.loads(providers_json)
    providers = []
    for provider_dict in providers_data:
        try:
            provider = Provider(
                provider_dict["provider_name"],
                provider_dict["api_base"],
                provider_dict["api_key"],
            )
        except KeyError as e:
            raise RuntimeError(
                f"Invalid {ENVIRONMENT_VAR_NAME} environment variable, must be a list of JSON objects containing provider_name, api_base, api_key") from e

        providers.append(provider)
    return providers

def get_provider_by_name(name: str) -> Provider:
    for provider in _providers_list:
        if provider.provider_name == name:
            return provider 
    raise RuntimeError(
        f"Missing provider named {name}, please add it to the {ENVIRONMENT_VAR_NAME}"
        f"environment variable")

_providers_json = get_environment_var(ENVIRONMENT_VAR_NAME, required=True)
_providers_list = _parse_providers(_providers_json)

