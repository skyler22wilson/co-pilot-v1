import polars as pl
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
import logging
import json
import os 
import calendar
from datetime import datetime

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

def create_schema():
    schema = {
            "part_number": pl.Utf8,
            "description": pl.Utf8,
            "supplier_name": pl.Utf8,
            "quantity": pl.Int16,
            "price": pl.Float64,
            "margin": pl.Float64,
            "months_no_sale": pl.Int16,
            "quantity_ordered_ytd": pl.Int16,
            "special_orders_ytd": pl.Int16,
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

def update_schema(df, schema):
    for col, dtype in schema.items():
        df = df.with_columns(
            pl.col(col).cast(dtype, strict=False)
        )
    return df

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

def create_long_form_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert a wide-format DataFrame containing monthly sales data into a long-format DataFrame.
    """
    current_year = datetime.now().year
    last_year = current_year - 1
    current_month_index = datetime.now().month

    this_year_sales_columns = [f'sales_{calendar.month_abbr[i].lower()}' for i in range(1, current_month_index + 1)]
    last_year_sales_columns = [f'sales_last_{calendar.month_abbr[i].lower()}' for i in range(1, 13)]

    # Helper function to unpivot and clean data
    def unpivot_and_clean(data, columns, year):
        return (data.unpivot(
                    index=["part_number"], 
                    on=columns, 
                    variable_name='month', 
                    value_name='quantity_sold'
                )
                .with_columns([
                    pl.lit(year).alias('year'),
                    pl.col("month").str.replace("sales_", "").str.replace("last_", "").str.replace("_", "")
                ])
        )

    df_this_year = unpivot_and_clean(df, this_year_sales_columns, current_year)
    df_last_year = unpivot_and_clean(df, last_year_sales_columns, last_year)
    df_long = pl.concat([df_this_year, df_last_year])
    logging.debug(f"Long form dataframe shape: {df_long.shape}")
    logging.debug(f"Long form dataframe head:\n{df_long.head()}")

    return df_long
