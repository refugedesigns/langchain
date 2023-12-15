import os
from dotenv import dotenv_values
import openai



config = dotenv_values(".env")
TOGETHER_API_KEY = config["TOGETHER_API_KEY"]


llm = openai.OpenAI(
    api_key=TOGETHER_API_KEY,
    base_url="https://api.together.xyz/v1",
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
)

result = llm("translate English to German: I love programming.")
print(result)

