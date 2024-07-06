import polars as pl
from io import StringIO
import logging
from datetime import datetime
import json
import os
import re
from dashboard.setup.utils import load_configuration

CONFIG_PATH = "dashboard/configuration/InitialConfig.json"
LOGGING_DIR = "Logs"
if not os.path.exists(LOGGING_DIR):
    os.makedirs(LOGGING_DIR)

log_filename = os.path.join(LOGGING_DIR, 'cleaning_script.log')
logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s: %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

def filter_conditions(dataframe, config):
    """
    Drop parts that are more than likely not in inventory anymore using Polars.
    """
    sales_columns = config['SalesColumns']

    # Create a column for summing sales and add conditions
    no_sales_condition = (
        (pl.col('months_no_sale') >= 24) &
        (dataframe.select(sales_columns).sum() == 0) &
        (pl.col('quantity') == 0) &
        (pl.col('quantity_ordered_ytd') == 0)
    )
    
    # Filter using the inverse of the condition
    filtered_df = dataframe.filter(~no_sales_condition)

    return filtered_df

def clean_and_convert_numeric(df, columns):
    """
    Optimized function to clean and convert specified columns to numeric types in Polars.
    
    Parameters:
    - df: Polars DataFrame to process.
    - columns: List of column names to clean and convert.
    
    Returns:
    - Polars DataFrame with cleaned and converted columns.
    """
    # Ensure only valid columns are processed
    valid_columns = [col for col in columns if col in df.columns]
    
    # Define a function to clean and convert columns to numeric
    def clean_numeric(col):
        # Remove non-numeric characters, convert to float
        return col.str.replace_all(r'[^\d.-]', '').cast(pl.Float64).fill_none(0.0)
    
    # Apply cleaning and conversion to each column
    for col in valid_columns:
        df = df.with_column(clean_numeric(pl.col(col)).alias(col))
    
    return df

def fix_price(df):
    """
    Ensure thatprices are non-negative.
    """
    price_mask = pl.col('price') < 0
    filtered_df = df.filter(price_mask)
    return filtered_df


def clean_text(dataframe, config):
    """
    Clean text in specified columns more efficiently.
    """
    # Copy only the text columns that need cleaning to avoid unnecessary data duplication
    columns_to_clean = [col for col in config["TextColumns"] if col in dataframe.columns]

    for column_name in columns_to_clean:
        dataframe.with_columns(
            (pl.col(column_name)
             .str.lower()
             .str.strip()
             .str.replace_all(re.compile(r'\s+'), ' ', regex=True)
             .alias(column_name)
            )
        )

    # Log after cleaning all specified columns
    logging.info(f'Cleaned text in columns: {columns_to_clean}')

    return dataframe

def clean_part_numbers(data_frame):
    """
    Clean and validate part numbers using vectorized operations.
    """

    # Filter DataFrame based on valid part numbers using a pre-compiled pattern
    valid_pattern = r'^(?=.*[0-9]).{3,15}$'
    data_frame = data_frame.filter(pl.col('part_number').str.match(valid_pattern))

    data_frame = data_frame.with_columns(
        pl.col('part_number')
        .str.strip()  # Strip whitespace from both ends
        .str.replace_all(re.compile(r'\s+'), ' ', regex=True)
        .alias('part_number')
    )

    return data_frame

def adjust_part_numbers(data_frame):
    condition = (
        (pl.col('part_number').str.lengths() >= 8) & 
        (pl.col('part_number').str.lower().str.startswith('t'))
    )
    
    # Update 'supplier_name' based on condition
    data_frame = data_frame.with_column(
        pl.when(condition).then("triumph").otherwise(pl.col('supplier_name')).alias('supplier_name')
    )
    
    # Strip whitespace from 'part_number'
    data_frame = data_frame.with_column(
        pl.col('part_number').str.strip().alias('part_number')
    )
    return data_frame

def adjust_brands(data_frame):

    """Renames BMW columns"""

    data_frame = data_frame.with_column(
        pl.when(pl.col('supplier_name').str.lower().str.contains('bmw'))
        .then('bmw')
        .otherwise(pl.col('supplier_name'))
        .alias('supplier_name')
    )
    return data_frame

def merge_duplicate_parts(data_frame, config):
    """
    Merges duplicate part number information into a single unique row
    """
    logging.info(f'Original Shape: {data_frame.shape}')

    # Categorize columns based on their intended aggregation method
    sales_columns = config['SalesColumns']
    specific_columns = ['quantity', 'quantity_ordered_ytd', 'special_orders_ytd']
    aggregation_columns = sales_columns + specific_columns
    non_aggregated_columns = [col for col in data_frame.columns if col not in aggregation_columns + ['part_number']]

    # Define aggregation rules combining all column types
    aggregation_rules = {col: 'sum' for col in aggregation_columns}
    aggregation_rules.update({col: 'first' for col in non_aggregated_columns})
    aggregation_rules['months_no_sale'] = 'min'  # Custom rule for specific columns

    # Perform the aggregation
    aggregated_df = data_frame.groupby('part_number').agg(aggregation_rules).reset_index()
    logging.info(f'Aggregated Shape: {aggregated_df.shape}')

    # Reorder columns to match the original order, ensuring all are present
    reordered_df = aggregated_df[data_frame.columns.intersection(aggregated_df.columns)]

    return reordered_df

def clip_sales_columns(dataframe, config):
    """
    Ensure that sales values are non-negative using Polars.
    """
    sales_columns = config["SalesColumns"]
    for column in sales_columns:
        dataframe = dataframe.with_column(
            pl.when(pl.col(column) < 0).then(0).otherwise(pl.col(column)).alias(column)
        )
    return dataframe

def adjust_quantity(df):
    df = df.with_columns([
        pl.when(df['quantity'] < 0).then(0).otherwise(df['quantity']).alias('quantity'),
        pl.when(df['quantity'] < 0).then(0).otherwise(df['negative_on_hand']).alias('negative_on_hand')
    ])
    return df

def update_months_no_sale(df, sales_columns_this_year, sales_columns_last_year):
    current_month_index = datetime.now().month

    # Select the relevant sales columns based on the current month
    if current_month_index == 1:
        months_to_count = sales_columns_last_year[1:]  # from last year's February to December
    else:
        # from this year's January to current month, and then last year's
        months_to_count = sales_columns_this_year[:current_month_index] + sales_columns_last_year

    # Reverse to start counting from the most recent month backwards
    months_to_count.reverse()

    # Calculate months_no_sale as the count of consecutive months with zero sales from the recent month backwards
    df = df.with_column(
        pl.fold(
            pl.lit(0),  # Start with zero
            months_to_count,
            lambda acc, x: pl.when(pl.col(x) == 0).then(acc + 1).otherwise(acc).keep_name()
        ).alias('months_no_sale_computed')
    )

    # Overwrite months_no_sale only if it's not more than 12 already
    df = df.with_column(
        pl.when(pl.col('months_no_sale') > 12)
        .then(pl.col('months_no_sale'))
        .otherwise(pl.col('months_no_sale_computed'))
        .alias('months_no_sale')
    ).drop('months_no_sale_computed')

    return df


def inventory_check(df, config):
    # Extract sales columns from the configuration
    sales_columns = config["SalesColumns"]
    
    # Calculate the sum of sales across all specified sales columns
    total_sales = df.select(sales_columns).sum()

    # Determine rows where quantity is zero and there have been no sales across all sales columns
    no_inventory_no_sales = (
        (pl.col('quantity') == 0) & 
        (total_sales == 0) &
        (pl.col('negative_on_hand') == 0)
    )
    df = df.filter(~no_inventory_no_sales)
    
    return df

def preprocess_data(data_frame, config):
    logging.info("Starting preprocessing data.")
    data_frame = filter_conditions(data_frame, config)
    logging.debug(f"Shape after filtering conditions: {data_frame.shape}")
    data_frame = clean_part_numbers(data_frame)
    data_frame = merge_duplicate_parts(data_frame, config)
    logging.debug(f"Shape after merging duplicates: {data_frame.shape}")
    
    columns = config['ColumnNames']
    data_frame = clean_and_convert_numeric(data_frame, columns)
    data_frame = fix_price(data_frame)
    data_frame = adjust_part_numbers(data_frame)
    data_frame = adjust_brands(data_frame)
    data_frame = clean_text(data_frame, config)
    data_frame = clip_sales_columns(data_frame, config)

    sales_this_year = config['SalesThisYear']
    sales_last_year = config['SalesLastYear']
    data_frame = update_months_no_sale(data_frame, sales_this_year, sales_last_year)
    data_frame = adjust_quantity(data_frame)

    data_frame = data_frame.with_columns(
        pl.col("description").fill_none("No Description").alias("description")
    )
    
    logging.info(f'Shape before removing parts not in inventory: {data_frame.shape}')
    data_frame = inventory_check(data_frame, config)
    logging.info(f'Shape after removing parts not in inventory: {data_frame.shape}')

    logging.info('Data preprocessing completed successfully.')
    return data_frame

def main(current_task, input_data):
    print('Cleaning script running...')
    config = load_configuration(CONFIG_PATH)
    if config is None:
        logging.error("Failed to load configuration.")
        return False

    if not input_data:
        logging.error('Input JSON data is missing.')
        return False

    try:
        df = pl.read_json(StringIO(input_data), infer_schema_length=100)
        print('data loaded successfully!')
        # Preprocess and clean data as before
        print(df.head(50))
        cleaned_data = preprocess_data(df, config)
        print('Data Cleaned Successfully!')
        # Use a temporary file for output
        cleaned_data_json = cleaned_data.write_json()
            # Now you can use tmp_output.name as the file path to access your data
        # Return success at the end
        return cleaned_data_json

    except ValueError as e:
        logging.error(f"Invalid JSON format: {e}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: Invalid JSON format.'})
        return False
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: {str(e)}'})
        return False

