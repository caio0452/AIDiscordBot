import os
import sys
from dotenv import load_dotenv

load_dotenv()

def get_environment_var(var_name: str, *, required: bool) -> str:
    value = os.getenv(var_name)
    if required and (value is None or value == ""):
        print(f"Missing '{var_name}' environment variable", file=sys.stderr)
        exit(1)
    return "" if value is None else value

api_key = get_environment_var('OPENAI_API_KEY', required=True)
api_base = get_environment_var('OPENAI_API_BASE', required=False)
qdrant_url = get_environment_var('QDRANT_URL', required=True)
qdrant_port = get_environment_var('QDRANT_PORT', required=True)
bot_token = get_environment_var('AI_BOT_TOKEN', required=True)