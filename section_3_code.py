from dotenv import dotenv_values
from langchain.chains import LLMChain
from langchain.llms.together import Together
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder


config = dotenv_values(".env")
TOGETHER_API_KEY = config["TOGETHER_API_KEY"]

llm = Together(
    together_api_key=TOGETHER_API_KEY,
    model="upstage/SOLAR-0-70b-16bit",
    temperature=0.7,
    max_tokens=2000,
    top_k=1
)

memory = ConversationSummaryBufferMemory(
    memory_key="chat_history", return_messages=True, llm=llm)

prompt = ChatPromptTemplate(
    input_variables=["content", "chat_history"],
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant helps asnwer users queries, only provide valid answers and dont hallucinate or make up your own answers, if you don't know the answer just say that you don't know."),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

while True:
    content = input(">>> ")

    result = chain({"content": content})

    print(result["text"] + "\n")
