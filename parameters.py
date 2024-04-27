from environment_vars import get_environment_var

QDRANT_URL = get_environment_var('QDRANT_URL', required=True)
QDRANT_PORT = get_environment_var('QDRANT_PORT', required=True)
BOT_TOKEN = get_environment_var('AI_BOT_TOKEN', required=True)
FAL_AI_KEY = get_environment_var('FAL_AI_API_KEY', required=False)