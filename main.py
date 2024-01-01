from dotenv import dotenv_values
from langchain.llms.together import Together
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

config = dotenv_values(".env")
TOGETHER_API_KEY = config["TOGETHER_API_KEY"]

llm = Together(
    together_api_key=TOGETHER_API_KEY,
    model="upstage/SOLAR-0-70b-16bit",
    temperature=0.7,
    max_tokens=2000,
    top_k=1
)

prompt = ChatPromptTemplate.from_template("tell me an interesting fact about {subject}")

chain = prompt | llm | StrOutputParser()

result = chain.invoke({"subject": "the sun"})

print(result)