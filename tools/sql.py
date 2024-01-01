import sqlite3
from langchain.tools import Tool 
from pydantic.v1 import BaseModel 
from typing import List


class RunSqlQueryArgsSchema(BaseModel):
    query: str
    
class DescribeTablesArgsSchema(BaseModel):
    table_names: List[str]
    

conn = sqlite3.connect("db.sqlite")

def list_tables():
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = c.fetchall()
    return "\n".join([row[0] for row in rows if row[0] is not None])

def run_sql_query(query):
    cur = conn.cursor()
    cur.execute(query)
    return cur.fetchall()

def describe_tables(table_names) -> str:
    c = conn.cursor()
    table_names = table_names.split(",")
    tables = ', '.join("'" + table + "'" for table in table_names)
    rows = c.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name IN ({tables});")
    return "\n".join([row[0] for row in rows if row[0] is not None])

run_sql_query_tool = Tool.from_function(
    name="run_sql_query",
    func=run_sql_query,
    description="useful for running SQL queries. it takes input of a sql query as string with no quotes and returns the result.",
    args_schema=RunSqlQueryArgsSchema,
)

describe_tables_tool = Tool.from_function(
    name="describe_tables",
    func=describe_tables,
    description="useful for describing SQL database tables. input should be a list of table names separated by commas.",
    # args_schema=DescribeTablesArgsSchema,
)
