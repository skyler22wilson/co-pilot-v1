import logging
from dashboard.scripts.create_db import Base
from dashboard.setup.utils import get_engine_and_session

# Utility function to create the schema
def create_database_schema(db_file_path):
    logging.info(f"Creating database schema at {db_file_path}")
    engine, _ = get_engine_and_session(db_file_path)
    Base.metadata.create_all(engine)
    logging.info("Database schema created successfully.")

def preprocess_dataframe(df, rename_map=None, drop_cols=None):
    """ Preprocess the DataFrame by renaming and dropping columns. """
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True, errors='ignore')
    return df

def filter_columns(df, model):
    """Filter DataFrame columns to match the model's table columns."""
    model_columns = {column.name for column in model.__table__.columns}
    df_columns = set(df.columns)
    filtered_columns = model_columns.intersection(df_columns)
    return df[list(filtered_columns)]

def insert_data_using_pandas(engine, table_name, data_df, model, if_exists='append', method='multi', chunksize=7500):
    """
    Inserts data from a Pandas DataFrame into a SQL table using the Pandas .to_sql method.
    
    :param engine: SQLAlchemy engine object
    :param table_name: Name of the table to insert data into
    :param data_df: DataFrame containing data to insert
    :param model: SQLAlchemy model representing the table schema
    :param if_exists: 'fail', 'replace', or 'append'
    :param method: Insert method, e.g., 'multi'
    :param chunksize: Number of rows per batch for insertion
    """
    try:
        data_df = filter_columns(data_df, model)
        # Ensure columns match the schema and have the correct data types
        for column in model.__table__.columns:
            if column.name in data_df.columns:
                data_df[column.name] = data_df[column.name].astype(column.type.python_type)
        
        # Insert data into the SQL table
        data_df.to_sql(table_name, engine, if_exists=if_exists, index=False, method=method, chunksize=chunksize)
        logging.info(f"Data successfully inserted into {table_name}.")
    except Exception as e:
        logging.error(f"Failed to insert data into {table_name}: {e}")






