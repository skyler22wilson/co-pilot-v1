import polars as pl
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
import logging
import json
import os

# Define the engine and session
def get_engine_and_session(db_file_path):
    engine = create_engine(f'sqlite:///{db_file_path}')
    Session = scoped_session(sessionmaker(bind=engine))
    return engine, Session

def convert_to_float(df, column_name):
    # First, remove spaces that might be used as thousand separators
    df = df.with_columns(
        pl.col(column_name)
        .str.replace(",", "")
        .alias("no_comma")
    )

    # Then remove commas which might be used as thousand separators
    df = df.with_columns(
        pl.col("no_comma")
        .str.replace(",", "")
        .cast(pl.Float64, strict=False)
        .alias(column_name)
    )

    # Remove the intermediate 'no_spaces' column if no longer needed
    df = df.drop("no_comma")

    return df

def convert_to_int(df, col_name):
    # Convert to integer after handling invalid or fractional values
    return df.with_columns(
        pl.col(col_name)
        .cast(pl.Float64)  # First cast to float to handle decimals
        .round()  # Round the float values to the nearest integer
        .cast(pl.Int16)  # Finally cast to Int16
        .alias(col_name)
    )

def load_configuration(config_path):
    if not os.path.exists(config_path):
        logging.error(f"Configuration file does not exist at path: {config_path}")
        return None

    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        return config
    except json.JSONDecodeError as json_err:
        logging.error(f"Error decoding JSON from the configuration file: {json_err}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error loading configuration from {config_path}: {e}")
        return None

def define_schema():
    schema = {
        "part_number": pl.Utf8,
        "description": pl.Utf8,
        "supplier_name": pl.Utf8,
        "quantity": pl.Int64,
        "price": pl.Float64,
        "margin": pl.Float64,
        "months_no_sale": pl.Int8,
        "quantity_ordered_ytd": pl.Int16,
        "special_orders_ytd": pl.Int8,
        "sales_last_jan": pl.Int16,
        "sales_last_feb": pl.Int16,
        "sales_last_mar": pl.Int16,
        "sales_last_apr": pl.Int16,
        "sales_last_may": pl.Int16,
        "sales_last_jun": pl.Int16,
        "sales_last_jul": pl.Int16,
        "sales_last_aug": pl.Int16,
        "sales_last_sep": pl.Int16,
        "sales_last_oct": pl.Int16,
        "sales_last_nov": pl.Int16,
        "sales_last_dec": pl.Int16,
        "sales_jan": pl.Int16,
        "sales_feb": pl.Int16,
        "sales_mar": pl.Int16,
        "sales_apr": pl.Int16,
        "sales_may": pl.Int16,
        "sales_jun": pl.Int16,
        "sales_jul": pl.Int16,
        "sales_aug": pl.Int16,
        "sales_sep": pl.Int16,
        "sales_oct": pl.Int16,
        "sales_nov": pl.Int16,
        "sales_dec": pl.Int16
    }
    return schema
