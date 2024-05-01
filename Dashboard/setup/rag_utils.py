from dash import html
from Dashboard.components.partswise_output import create_table
import pandas as pd
from Dashboard.scripts.db_setup import generate_db_file_path
from sqlalchemy import create_engine, text
from llama_index.core import SQLDatabase
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import NLSQLTableQueryEngine
import os
import openai

db_file_path = generate_db_file_path()

connection_string = f"sqlite:///{db_file_path}"

# Create an engine instance
engine = create_engine(connection_string)

def setup_nlsql_query_engine():
    sql_database = SQLDatabase(engine, sample_rows_in_table_info=2)

    context_preface = (
        """
            As an LLM agent managing the 'parts' and 'sales' tables, focus on:
            - 'inventory_category' contains (essential, non-essential, nearing_obsolete, obsolete)
            - ALWAYS include 'part_number', 'description', 'supplier_name', 'price', 'quantity', 'cost_per_unit' at begining of results
            -Include all relevant columns related to the query in results
            - All text is lowercase.
            - Use 'margin_percentage' for margin analysis, except when dollars are specified.
            - Aggregate 'SUM(cost)' for 'total cost' queries, only as explicitly requested.
            - Match any part of 'supplier_name' for inclusivity (e.g., '%bmw%') for brand-specific queries.
            - Convert percentage queries (e.g., '50%') into decimal format (e.g., 0.5).
            - The 'sales' table is for time-based sales analysis; prefix columns with table names for clarity in joins.
        """
    )

    os.environ["OPENAI_API_KEY"] = "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"
    openai.api_key = os.environ["OPENAI_API_KEY"]


    llm = OpenAI(temperature=0.1, model="gpt-4-turbo-preview")
    Settings.llm = llm
    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        service_context=Settings,
        tables = ['parts'],
        context_str_prefix= context_preface
    )
    return query_engine

query_engine = setup_nlsql_query_engine()

def process_user_input_to_sql(user_input):
    response = query_engine.query(user_input)
    sql_query = response.metadata['sql_query'].replace('\n', ' ')  # Assuming this is the format
    return sql_query

# This function decides the output format based on whether the SQL query contains aggregation functions
def query_output(user_input):
    sql_query = process_user_input_to_sql(user_input)
    with engine.connect() as connection:
        result = connection.execute(text(sql_query))
        result_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        if len(result_df) >= 5:
            # Here, a table will be displayed, hence set 'has_data' to True
            return html.Div(create_table(result_df), className='table-container'), True
        else:
            # In this case, no table data is available, hence set 'has_data' to False
            response = query_engine.query(user_input)
            return html.Div(str(response), className='partswise-output-text'), False