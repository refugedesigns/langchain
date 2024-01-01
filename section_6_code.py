from langchain.llms.together import Together
from langchain.agents import AgentType, initialize_agent, ConversationalChatAgent
from langchain.memory import ConversationBufferWindowMemory
from dotenv import dotenv_values
from tools.sql import run_sql_query_tool, list_tables, describe_tables_tool
from handlers.chat_model_start_handler import ChatModelStartHandler


config = dotenv_values(".env")
TOGETHER_API_KEY = config["TOGETHER_API_KEY"]

handler = ChatModelStartHandler()

llm = Together(
    together_api_key=TOGETHER_API_KEY,
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0,
    max_tokens=2000,
)

memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True)

tools = [
    run_sql_query_tool,
    describe_tables_tool
]

tool_names = [tool.name for tool in tools]

tables = list_tables()

FORMAT_INSTRUCTIONS = """
The database has tables of:\n{tables}\n 

To use a tool, please use the following format:
'''
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of {tool_names}
Action Input: the input to the action
Observation: the result of the action
'''

Just return the answer after you find it and dont keep up asking unecessary questions that is not related to the question asked.

Always return the answer with: 'Final Answer: ' in front of it.

Do not make any assumptions about what tables exist or what columns exist, 

instead use the 'describe_tables' tool. 

The 'describe_tables' tool accepts a string of table names separated by commas. 

The tool should return a list of SQL queries that describe the tables in the database.

Don't add quotes to the sql query when using the 'run_sql_query' tool.
"""

SUFFIX = '''

Begin!

Previous conversation history:
{chat_history}

Instructions: {input}
{agent_scratchpad}
'''

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={
        "format_instructions": FORMAT_INSTRUCTIONS.format(tables=tables, tool_names=tool_names),
        "suffix": SUFFIX,
    },
    memory=memory,
    callbacks=[handler],
)

# print(agent.agent.llm_chain.prompt.messages[0].prompt.template)
# agent.agent.llm_chain.prompt.template = FORMAT_INSTRUCTIONS

agent("How many orders are there in the orders table?")

agent("Do the same for the users table.")
