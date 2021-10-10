# IMPORTANT- THIS CODE WILL NOT RUN ON IT'S OWN

# IT REQURES A PRE-EXISITING FILE ON WHICH YOUR
# BOT ALREADY RUNS ON

# IMPORTS

import random

# IMPORTANT- THIS ALSO REQUIRES A TEXT FILE CONTAINING THE QUOTES
# (ALSO GIVEN)

# COMMAND CODE

@bot.command(pass_context = True)

async def quote(ctx):

    quoteFile = open("quotes.txt","r")
    quotes = quoteFile.readlines()
    quoteFile.close()
    
    rQuote = random.randrange(0,len(quotes) - 2)
    asignedQuote = quotes[rQuote]
    displayQuote = asignedQuote[0:len(asignedQuote)-1]

    embed = discord.Embed(color = 0x00ffff)
    embed.add_field(name='Quote #{}'.format(rQuote), value=displayQuote, inline = True)

    await bot.say(embed=embed)
