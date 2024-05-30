from dash import html
from Dashboard.components.partswise_output import create_table
import pandas as pd
from Dashboard.scripts.create_db import generate_db_file_path, Parts, create_database_schema
from sqlalchemy import create_engine, text
from llama_index.core import SQLDatabase
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.llms.openai import OpenAI
import os
import openai

db_file_path = generate_db_file_path()
create_database_schema(db_file_path)
connection_string = f"sqlite:///{db_file_path}"
engine = create_engine(connection_string)

# Function to setup NL-SQL Query Engine
def setup_nlsql_query_engine():
    # Function to initialize SQLDatabase and table objects
    def initialize_table_objects():
        sql_database = SQLDatabase(engine, sample_rows_in_table_info=2, include_tables=['sales', 'supplier_parts_summary', 'parts', 'supplier_sales_summary'])
        parts_context = "Provides detailed inventory data for individual parts. Use part-specific queries. Combine with 'sales' tables for temporal financial performance"
        sales_context = "Provides time-based sales data for individual parts. Use for part-specific sales queries."
        supplier_parts_context = "Provides key inventory data for each supplier/brand. Use for supplier/brand-specific financial data queries. Combine with 'supplier_sales_summary' tables for temporal financial performance"
        supplier_sales_context = "Provides time-based sales data for each supplier/brand. Use for supplier/brand-specific sales queries."

        table_node_mapping = SQLTableNodeMapping(sql_database)
        table_schema_objs = [
            SQLTableSchema(table_name='sales', context_str=sales_context),
            SQLTableSchema(table_name='supplier_parts_summary', context_str=supplier_parts_context),
            SQLTableSchema(table_name='parts', context_str=parts_context),
            SQLTableSchema(table_name='supplier_sales_summary', context_str=supplier_sales_context)
        ]
        obj_index = ObjectIndex.from_objects(
            table_schema_objs,
            table_node_mapping,
            VectorStoreIndex,
        )
        return sql_database, table_schema_objs, obj_index


    # Function to generate table context string
    def get_table_context_str(sql_database, table_schema_objs):
        context_strs = []
        for table_schema_obj in table_schema_objs:
            table_info = sql_database.get_single_table_info(table_schema_obj.table_name)
            if table_schema_obj.context_str:
                table_opt_context = " The table description is: "
                table_opt_context += table_schema_obj.context_str
                table_info += table_opt_context
            context_strs.append(table_info)
        return "\n\n".join(context_strs)


    # Initialize table objects and get table context string
    sql_database, table_schema_objs, obj_index = initialize_table_objects()
    table_context_str = get_table_context_str(sql_database, table_schema_objs)

    # General Context String
    context_str = (
    "Inventory categories: essential, non-essential, nearing obsolescence, obsolete. "
    "Ensure detailed, relevant responses, including 'supplier_name', 'price', and 'quantity'. "
    "Always use lowercase for ALL queries. Access 'supplier_name' flexibly e.g., ('%bmw'). "
    "Convert percentages to decimals (e.g., '50%' as '0.5'). "
    "Use JOINs prefaced with table names for combining multiple tables. "
    )

    # Combine Table Contexts
    context_str_combined = context_str + "\n\n" + table_context_str

    openai.api_key = os.environ["OPENAI_API_KEY"]  # Replace with your OpenAI API key
    query_engine = SQLTableRetrieverQueryEngine(
        sql_database=sql_database,
        table_retriever=obj_index.as_retriever(similarity_top_k=1),
        synthesize_response=True,
        llm=OpenAI(temperature=0.1, model="gpt-3.5-turbo-0125"),
        context_str_prefix=context_str_combined
    )
    return query_engine


query_engine = setup_nlsql_query_engine()

def process_user_input_to_sql(user_input):
    response = query_engine.query(user_input)
    # Ensure that sql_query is extracted correctly without extra leading text
    sql_query = response.metadata['sql_query'].replace('\n', ' ').strip()
    if sql_query.lower().startswith('sql'):
        sql_query = sql_query[3:].strip()  # Remove the incorrect 'sql' prefix if it exists
    #logging.info(f"SQL: {sql_query}")
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
        