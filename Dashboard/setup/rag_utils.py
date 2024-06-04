from dash import html
from Dashboard.components.partswise_output import create_table
import pandas as pd
from Dashboard.scripts.create_db import generate_db_file_path, create_database_schema
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

# Generate database file path and create the database schema
db_file_path = generate_db_file_path()
create_database_schema(db_file_path)
connection_string = f"sqlite:///{db_file_path}"
engine = create_engine(connection_string)

def setup_nlsql_query_engine():
    """
    Sets up the NL-SQL Query Engine by initializing the SQL database, creating table objects,
    and combining context strings for table schemas and general context. This function also
    configures the OpenAI API for generating SQL queries from natural language input.

    Returns:
        SQLTableRetrieverQueryEngine: Configured NL-SQL Query Engine.
    """
    def initialize_table_objects():
        """
        Initializes the SQLDatabase and SQLTableSchema objects.

        Returns:
            tuple: A tuple containing the SQLDatabase object, a list of SQLTableSchema objects,
                   and an ObjectIndex object.
        """
        sql_database = SQLDatabase(engine, sample_rows_in_table_info=2, include_tables=['sales', 'supplier_parts_summary', 'parts', 'supplier_sales_summary'])
        parts_context = "Provides detailed inventory data for individual parts. Use part-specific queries. Combine with 'sales' tables for temporal financial performance."
        sales_context = "Provides time-based sales data for individual parts. Use for part-specific sales queries."
        supplier_parts_context = "Provides key inventory data for each supplier/brand. Use for supplier/brand-specific financial data queries. Combine with 'supplier_sales_summary' tables for temporal financial performance."
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

    def get_table_context_str(sql_database, table_schema_objs):
        """
        Generates a combined context string for all the table schemas.

        Args:
            sql_database (SQLDatabase): The SQL database object.
            table_schema_objs (list): List of SQLTableSchema objects.

        Returns:
            str: Combined context string for all table schemas.
        """
        context_strs = []
        for table_schema_obj in table_schema_objs:
            table_info = sql_database.get_single_table_info(table_schema_obj.table_name)
            if table_schema_obj.context_str:
                table_opt_context = " The table description is: "
                table_opt_context += table_schema_obj.context_str
                table_info += table_opt_context
            context_strs.append(table_info)
        return "\n\n".join(context_strs)

    sql_database, table_schema_objs, obj_index = initialize_table_objects()
    table_context_str = get_table_context_str(sql_database, table_schema_objs)

    context_str = (
        "Inventory categories: essential, non-essential, nearing obsolescence, obsolete. "
        "Ensure detailed, relevant responses, including 'supplier_name', 'price', and 'quantity'. "
        "Access 'supplier_name' flexibly e.g., ('%bmw'). "
        "Convert percentages to decimals (e.g., '50%' as '0.5'). "
        "Use JOINs prefaced with table names for combining multiple tables."
    )

    context_str_combined = context_str + "\n\n" + table_context_str
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("No API key found for OpenAI. Please set the OPENAI_API_KEY environment variable.")
    openai.api_key = openai_api_key
    query_engine = SQLTableRetrieverQueryEngine(
        sql_database=sql_database,
        table_retriever=obj_index.as_retriever(similarity_top_k=1),
        synthesize_response=True,
        llm=OpenAI(temperature=0.1, model="gpt-3.5-turbo-0125"),
        context_str_prefix=context_str_combined
    )
    return query_engine

# Set up the query engine
query_engine = setup_nlsql_query_engine()

def process_user_input_to_sql(user_input):
    """
    Processes user input and generates an SQL query using the NL-SQL Query Engine.

    Args:
        user_input (str): Natural language input from the user.

    Returns:
        str: Generated SQL query.
    """
    response = query_engine.query(user_input)
    sql_query = response.metadata.get('sql_query', '').replace('\n', ' ').replace('\r', ' ').strip().lower()
    if sql_query.startswith('sql'):
        sql_query = sql_query[3:].strip()
    return sql_query


def query_output(user_input):
    """
    Executes the generated SQL query and decides the output format based on the query results.

    Args:
        user_input (str): Natural language input from the user.

    Returns:
        tuple: A tuple containing the HTML output and a boolean indicating if data is available.
    """
    sql_query = process_user_input_to_sql(user_input)
    with engine.connect() as connection:
        result = connection.execute(text(sql_query))
        result_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        if len(result_df) >= 5:
            return html.Div(create_table(result_df), className='table-container'), True
        else:
            response = query_engine.query(sql_query)
            return html.Div(str(response), className='partswise-output-text'), False

        