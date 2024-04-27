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