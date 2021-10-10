# IMPORTANT - THIS CODE WILL NOT WORK ON ITS OWN AND REQUIRES A BOT TOKEN

# THIS IS 100% CHANGEABLE TO YOUR OWN LIKING

# PLEASE LOOK INTO http://discordpy.readthedocs.io/en/latest/api.html
# FOR A FULL RANGE OF THE DISCORD PYTHON API

import discord
from discord.ext.commands import Bot
from discord.ext import commands

from discord.enums import ChannelType
from discord.ext.commands.bot import _get_variable

@bot.event
async def on_ready():
    print('Status : Online')
    print('Name : {}'.format(bot.user.name))
    print('ID: {}'.format(bot.user.id))

@bot.command(pass_context=True)

async def examplecommand(ctx):

    await bot.say('Hello World!')

async def embed(ctx):

    embed = discord.Embed(color=('0x'+str(color)))
    embed.set_author(name='This is an embed')
    embed.add_field(name='This is a field',value='This is a description')

    await bot.say(embed=embed)

bot.run('TOKENHERE')
