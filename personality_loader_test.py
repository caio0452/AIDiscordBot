from bot_workflow.personality_loader import PersonalityLoader
import json

l = PersonalityLoader("C:\\Users\\caiop\\Documents\\Found files\\dir0005.chk\\Documents\\VSCode\\PaperChanV2\\AIDiscordBot\\bot_workflow\\personality.json")
p = l.load_personality()


print("---------")
print(p.request_params) 