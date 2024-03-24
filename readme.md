# How to run this bot?

1. Run `pip install requirements.txt`
2. Get your OpenAI API key
3. Get your Discord bot's token
4. Run a [Qdrant](https://github.com/qdrant/qdrant/blob/master/QUICK_START.md) database
5. See the example.env files for the environment variables you have to set. Or, instead, create a copy of it named `.env`in the same directory, with the variables set
6. Run `python main.py`

The bot does not require "read message" intents, and will only read messages if pinged.

# How to add knowledge?
Create a folder named "knowledge" and add some `.txt` files in it. The text files must contain either `paragraphs` or `chunks` somewhere in the text, and that decides how they will be indexed by AI:
* `paragraphs` will split chunks of knowledge whenever it finds a double-newline
* `chunks`will split chunks of knowledge in pieces of approximately ~24k character

That means that the AI will include whole pieces of knowledge it deems relevant on its prompt

# Caution
The whole knowledge database is re-indexed every time the bot restarts (`knowledge.py`). This is a known issue, and will cost you some money, even though creating text embeddings is very cheap

Also, this code is in general very unfinished. Expect bugs.

# Costs
Experimental data suggests the cost is around ~$2/1000 messages exchanged with the bot (pessimistic estimate).

Indexing knowledge (happens just once on startup) costs around $0.04 per megabyte of text.