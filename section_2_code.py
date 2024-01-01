from dotenv import dotenv_values
from langchain.llms.together import Together
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="return list of numbers")
parser.add_argument("--language", type=str, default="python")
args = parser.parse_args()


config = dotenv_values(".env")
TOGETHER_API_KEY = config["TOGETHER_API_KEY"]
MODEL_NAME = config["MODEL_NAME"]

llm = Together(
    together_api_key=TOGETHER_API_KEY,
    model="upstage/SOLAR-0-70b-16bit",
    temperature=0.7,
    max_tokens=2000,
    top_p=0.9
)

code_prompt = PromptTemplate(
    input_variables=["task", "language"],
    template="Write a very short {language} function that {task}",
)

test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test for the following {language} code: \n{code}",
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code",
)

test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test",
)

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["task", "language"],
    output_variables=["code", "test"]
)

result = chain({
    "task": args.task,
    "language": args.language
})

print(">>>>>>>>>>>> GENERATED CODE: <<<<<<<<<<<<<\n")
print(result["code"])

print(">>>>>>>>>>>> GENERATED TEST: <<<<<<<<<<<<<\n")
print(result["test"])
