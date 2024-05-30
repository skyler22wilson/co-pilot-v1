import pandas as pd
import base64
import io
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

# Define the engine and session
def get_engine_and_session(db_file_path):
    engine = create_engine(f'sqlite:///{db_file_path}')
    Session = scoped_session(sessionmaker(bind=engine))
    return engine, Session

def parse_contents(contents):
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        # Attempt to load CSV data
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        json_data = df.to_json(orient='split')  # Convert DataFrame to JSON
        return json_data  # Return JSON data
    except ValueError as ve:
        print(f'ValueError during processing: {ve}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')

    return None


def generate_table_columns(dataframe, columns_list):
    # Determine column types
    columns = [
        {
            "name": col,
            "id": col,
            "type": "numeric" if dataframe[col].dtype in ['int64', 'float64'] else "text",
        }
        for col in columns_list
    ]
    return columns

def parts_by_category(conn):
    query = """
    SELECT inventory_category, COUNT(*) as number_of_parts 
    FROM parts 
    GROUP BY inventory_category
    """
    category_data = conn.execute(query).fetchall()

    total_parts_query = "SELECT COUNT(*) FROM parts"
    total_parts = conn.execute(total_parts_query).fetchone()[0]

    # Convert to a list of dictionaries
    parts_by_category = [{'inventory_category': row[0], 'number_of_parts': row[1], 
                          'Percentage': (row[1] / total_parts) * 100} for row in category_data]

    return parts_by_category

def calculate_obsolete_value(conn):
    query = """
    SELECT SUM(cost) 
    FROM parts 
    WHERE inventory_category = 'obsolete'
    """
    obsolete_value = conn.execute(query).fetchone()[0]
    return round(obsolete_value, 2) if obsolete_value else 0

def calculate_obsolete_percentage(conn, target=5.0):
    obsolete_count_query = """
    SELECT COUNT(*) 
    FROM parts 
    WHERE inventory_category = 'obsolete'
    """
    current_obsolete_count = conn.execute(obsolete_count_query).fetchone()[0]

    total_count_query = "SELECT COUNT(*) FROM parts"
    total_count = conn.execute(total_count_query).fetchone()[0]

    obsolete_percentage = (current_obsolete_count / total_count) * 100 if total_count else 0
    color = "green" if obsolete_percentage <= target else "red"
    percentage_str = "{:.2f}%".format(obsolete_percentage)

    return percentage_str, color