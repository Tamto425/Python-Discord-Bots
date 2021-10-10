# IMPORTANT- THIS CODE WILL NOT RUN ON IT'S OWN

# IT REQURES A PRE-EXISITING FILE ON WHICH YOUR
# BOT ALREADY RUNS ON


# IMPORTS

import requests

# COMMAND CODE

@bot.command(pass_context=True)

async def cat(ctx): 
   
    r = requests.get('http://thecatapi.com/api/images/get?format=xml')

    Arr = []
    for line in r.iter_lines():
        if line:
            Arr.append(line)

    newStr = Arr[5].decode('utf-8')
    url1 = newStr[13:-6]

    #await bot.say(url1)
    
    embed = discord.Embed(color = 0xcc00cc)
    embed.set_image(url=url1)
    await bot.say(embed = embed)
