# IMPORTANT- THIS CODE WILL NOT RUN ON IT'S OWN

# IT REQURES A PRE-EXISITING FILE ON WHICH YOUR
# BOT ALREADY RUNS ON

# IMPORTS

import requests

# COMMAND CODE

def get_urban_definitions(word):
    
    url = 'http://api.urbandictionary.com/v0/define?term=%s' % (str(word))
    dat = requests.get(url).json()
    
    return dat['list']

def get_urban_random_definitions():

    url = 'http://api.urbandictionary.com/v0/random'
    dat1 = requests.get(url).json()

    return dat1['list']

@bot.command(pass_context=True)

async def urban(ctx, messages: str = None):

    
    
    if messages is None:
        
        embed = discord.Embed(color = 0xff66ff)
        embed.set_author(name='!Must specify something to define!')
        await bot.say(embed = embed)

    else:
                
        word = messages
        define = []
        deflist = '**Definition:**\n'
        
        definition = get_urban_definitions(word)[0]

        embed = discord.Embed(color = 0xff66ff)
            
        for i in range(len(definition['definition'])//2000 + 1):
            define += (definition['definition'][i*2000:i*2000+2000])
            
        examples = ('**examples:**\n%s' % definition['example'])

        for x in range(len(define)):

            deflist = deflist.join('\n'+define[x])
        
        embed.add_field(name = 'Urban: '+ ''.join(messages), value = '{0}\n\n{1}'.format(deflist, examples))

        await bot.say(embed=embed)

@bot.command(pass_context=True,aliases=['randomurban', 'rural'])

async def randomurban(ctx):
     
    define = []
    deflist = '**Definition:**\n'
    
    definition = get_urban_random_definitions()[0]

    embed = discord.Embed(color = 0xff66ff)
    
    for i in range(len(definition['definition'])//2000 + 1):
        define += (definition['definition'][i*2000:i*2000+2000])
        
    examples = ('**examples:**\n%s' % definition['example'])

    for x in range(len(define)):

        deflist = deflist.join('\n'+define[x])
    
    embed.add_field(name = 'Urban: {}'.format(definition['word']), value = '{0}\n\n{1}'.format(deflist, examples))

    await bot.say(embed=embed)
