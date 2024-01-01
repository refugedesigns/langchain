from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms.together import Together
from langchain.chains import RetrievalQA
from redundant_filter_retriever import RedundantFilterRetriever
from dotenv import dotenv_values
import langchain

langchain.debug = True

config = dotenv_values(".env")
OPENAI_API_KEY = config["OPENAI_API_KEY"]
TOGETHER_API_KEY = config["TOGETHER_API_KEY"]


llm = Together(
    together_api_key=TOGETHER_API_KEY,
    model="upstage/SOLAR-0-70b-16bit",
    temperature=0.9,
    max_tokens=2000,
)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# create instance of embedding database for documents retrieval
db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings,
)


# # retriever function for retrieving docs from db
# retriever = db.as_retriever()

# custom retriever in action
retriever = RedundantFilterRetriever(embeddings=embeddings, chroma=db)

# RetrievalQA chain does all the hard work
# of calculating embeddings of our question, searching the db for relevant docs
# and sending it to the LLM for answers.
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
)

result = chain.run("What is an interesting fact about the English Language?")

print(result)
