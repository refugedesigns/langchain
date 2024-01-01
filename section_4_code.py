from dotenv import dotenv_values
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter


config = dotenv_values(".env")
OPENAI_API_KEY = config["OPENAI_API_KEY"]

# generating embeddings from openai
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# example of how to split text
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)

# example of how to load a document
loader = TextLoader("facts.txt")
docs = loader.load_and_split(text_splitter=text_splitter)


# generating embedings for all docs
# and saving in chroma db at the same time
db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="emb"
)

# example of how to search the db for docks, k means number of docs to return
results = db.similarity_search(
    "What is an interesting fact about English language?", k=8)

for result in results:
    print("---------------------------\n")
    print(result.page_content)