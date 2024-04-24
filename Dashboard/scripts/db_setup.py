import pandas as pd
import sqlite3
import logging
import os
from datetime import datetime
import logging
import json

def generate_db_file_path(user_identifier='island_moto'):
    base_directory = 'data/databases/'
    db_file_name = f'partswise_{user_identifier}.db'
    db_file_path = os.path.join(base_directory, db_file_name)

    # Ensure the directory exists before returning the file path
    directory = os.path.dirname(db_file_path)
    print(f"Directory to create: {directory}")
    if not os.path.exists(directory):
        logging.info(f"Creating directory for database at {directory}")
        os.makedirs(directory, exist_ok=True)
    return db_file_path

def create_database_schema(db_file_path):
    logging.info(f"Creating database schema at {db_file_path}")
    schema_statements = [
            '''
                CREATE TABLE IF NOT EXISTS parts (
                    part_number TEXT PRIMARY KEY,
                    description TEXT,
                    supplier_name TEXT,
                    quantity INTEGER,
                    price REAL,
                    margin REAL,
                    margin_percentage REAL,
                    cost_per_unit INTEGER,
                    cost INTEGER,
                    months_no_sale INTEGER,
                    last_sold_date DATE,
                    quantity_ordered_ytd INTEGER,
                    last_received_date DATE,
                    special_orders_ytd INTEGER,
                    sales_revenue REAL,
                    cogs REAL,
                    gross_profit REAL,
                    roi REAL,
                    _90_days_supply REAL,
                    _90_day_turnover REAL,
                    _365_days_supply REAL,
                    _12_month_turnover REAL,
                    sales_to_stock_ratio REAL,
                    order_to_sales_ratio REAL,
                    safety_stock INTEGER,
                    reorder_point INTEGER,
                    demand REAL,
                    inventory_category TEXT
                );
            ''',
            '''
                CREATE TABLE IF NOT EXISTS sales (
                    part_number TEXT,
                    month TEXT,
                    year INTEGER,
                    quantity_sold INTEGER,
                    FOREIGN KEY (part_number) REFERENCES parts(part_number) ON DELETE CASCADE ON UPDATE CASCADE,
                    UNIQUE(part_number, month, year)
                );
            ''',
            'CREATE INDEX IF NOT EXISTS  idx_supplier_name ON parts (supplier_name);',
            'CREATE INDEX IF NOT EXISTS  idx_gross_profit ON parts (gross_profit);',
            'CREATE INDEX IF NOT EXISTS  idx_cost ON parts (cost);',
            'CREATE INDEX IF NOT EXISTS  idx_quantity ON parts (quantity);',
            'CREATE INDEX IF NOT EXISTS  idx_price ON parts (price);',
            'CREATE INDEX IF NOT EXISTS  idx_cost_per_unit ON parts (cost_per_unit);',
            'CREATE INDEX IF NOT EXISTS  idx_margin ON parts (margin);',
            'CREATE INDEX IF NOT EXISTS  idx_margin_perc ON parts (margin_percentage);',
            'CREATE INDEX IF NOT EXISTS  idx_turnover ON parts (_12_month_turnover);',
            'CREATE INDEX IF NOT EXISTS  idx_roi ON parts (roi);',
            'CREATE INDEX IF NOT EXISTS  idx_sales_revenue ON parts (sales_revenue);',
            'CREATE INDEX IF NOT EXISTS  idx_inventory_category ON parts (inventory_category);',
            'CREATE INDEX IF NOT EXISTS  idx_month ON sales (month);',
            'CREATE INDEX IF NOT EXISTS  idx_year ON sales (year);',
        ]
    try:
        with sqlite3.connect(db_file_path) as conn:
            cursor = conn.cursor()
            for statement in schema_statements:
                cursor.execute(statement)
            conn.commit()
        logging.info("Database schema created successfully.")
    except sqlite3.OperationalError as e:
        logging.error(f"Failed to connect to db file: {e}")
        raise
    except sqlite3.Error as e:
        logging.error(f"SQLite error during schema creation: {e}")
        raise

def insert_data_into_db(db_file_path, df, table_name, mode='append'):
    """
    Inserts data from a Pandas DataFrame into an SQLite database.

    :param db_file_path: Path to the SQLite database file.
    :param df: DataFrame to insert into the database.
    :param table_name: Name of the table to insert data into.
    :param mode: 'replace' to replace data, 'append' to add new data without removing existing data.
    """
    logging.info(f"Inserting data into {table_name} table at {db_file_path}")
    logging.info(f"Original column names: {list(df.columns)}")
        
    logging.info("Renaming 'total_cost' column to 'cost'...")
    df.rename(columns={'total_cost': 'cost'}, inplace=True)
    logging.info(f"Renamed column names: {list(df.columns)}")
    try:
        conn = sqlite3.connect(db_file_path)
        df.to_sql(table_name, conn, if_exists=mode, index=False)
        logging.info(f"Data inserted successfully into {table_name} table with mode {mode}.")
    except Exception as e:
        logging.error(f"An error occurred while inserting data into {table_name}: {e}")
    finally:
        conn.close()

def update_parts_data(db_file_path, new_parts_df):
    """
    Updates parts data in the database, replacing existing entries.
    """
    logging.info("Updating parts data in the database.")
    insert_data_into_db(db_file_path, new_parts_df, 'parts', mode='replace')


def bulk_insert_sales_data(db_file_path, data_df):
    """
    Performs a bulk insert of monthly sales data into the SQLite database.
    
    :param db_file_path: The file path of the SQLite database.
    :param data_df: A pandas DataFrame containing the sales data to insert.
    """
    # Ensure the DataFrame matches the table's schema: part_number, month, year, quantity_sold
    data_to_insert = list(data_df[['part_number', 'month', 'year', 'quantity_sold']].itertuples(index=False, name=None))

    # Connect to the SQLite database
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    try:
        # Start a transaction
        conn.execute('BEGIN')
        
        # Log the number of rows about to be inserted
        logging.info(f"Attempting to insert {len(data_to_insert)} rows into the sales table.")

        # Perform bulk insert
        cursor.executemany("""INSERT INTO sales (part_number, month, year, quantity_sold) 
                            VALUES (?, ?, ?, ?)""", data_to_insert)
        
        # Commit the transaction
        conn.commit()
        logging.info("Bulk insert operation completed successfully.")
    except sqlite3.IntegrityError as ie:
        logging.error(f"An integrity error occurred: {ie}")
        conn.rollback()
    except Exception as e:
        logging.error(f"An error occurred during the bulk insert operation: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def insert_current_year_data(db_file_path, data_df):
    # Determine the current year
    current_year = datetime.now().year

    # Filter the DataFrame for the current year
    current_year_df = data_df[data_df['year'] == current_year]

    # Convert DataFrame to a list of tuples for executemany
    data_to_insert = list(current_year_df.itertuples(index=False, name=None))

    # Connect to the SQLite database
    conn = sqlite3.connect(db_file_path)
    
    # Use a transaction for efficiency
    conn.execute('BEGIN')
    try:
        # Perform bulk insert; adjust your SQL based on your table's schema
        conn.executemany("""INSERT INTO sales (part_number, month, year, quantity_sold) 
                    VALUES (?, ?, ?, ?) 
                    ON CONFLICT(part_number, month, year) DO UPDATE SET
                    quantity_sold = excluded.quantity_sold""", data_to_insert)
        
        # Commit the transaction
        conn.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        conn.close()

def main(current_task, input_data):
    print('Entering the main function')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    db_file_path = generate_db_file_path()
    create_database_schema(db_file_path)  # Ensure the database schema is ready
    
    logging.info("Starting main function")
    try:
        logging.info("Loading parts data...")
        combined_data = json.loads(input_data)
        monthly_data = json.loads(combined_data['monthly_sales'])
        non_sales = json.loads(combined_data['non_sales'])

        new_monthly_sales_df = pd.DataFrame(monthly_data['data'], columns=monthly_data['columns'])
        df_non_sales = pd.DataFrame(non_sales['data'], columns=non_sales['columns'])

        df_non_sales.to_feather("/Users/skylerwilson/Desktop/PartsWise/Data/Processed/parts_data_finalized.feather")
        logging.info("Renaming columns...")
        df_non_sales.rename(columns={'total_cost': 'cost', '90_days_supply': '_90_days_supply', '365_days_supply': '_365_days_supply', '12_month_turnover':'_12_month_turnover'}, inplace=True)
        
        update_parts_data(db_file_path, df_non_sales)
        bulk_insert_sales_data(db_file_path, new_monthly_sales_df)
        insert_current_year_data(db_file_path, new_monthly_sales_df)

        return {'status': 'SUCCESS', 'message': 'Database created successfully, now building your dashboard...', 'data': db_file_path}

    except ValueError as e:
        logging.error(f"Invalid JSON format: {e}")
        return {'status': 'FAILURE', 'message': f'Error processing data: Invalid file format.'}
    except Exception as e:
        logging.error(f"Error processing data: {str(e)}")
        return {'status': 'FAILURE', 'message': f'Error processing data: {str(e)}'}


if __name__ == '__main__':
    main()