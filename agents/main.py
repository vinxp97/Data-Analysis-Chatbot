#from ChatBots import ProvisionBot
from MainBot import MainBot

Bot = MainBot()

t=1

while t>0:
    prompt = input("Type 'end' to end the session or enter a question for the bot: \n")

    if prompt == 'end':
        t = 0

    else:
        intent, response, history, system = Bot.query_bot(prompt) #askthebot(prompt=prompt)
        print(response)