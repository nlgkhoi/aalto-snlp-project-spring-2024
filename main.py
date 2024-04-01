import poe
import json

client = poe.Client("quJ_iX6CvpowFJceRlNtEQ%3D%3D")

print(json.dumps(client.bot_names, indent=2))