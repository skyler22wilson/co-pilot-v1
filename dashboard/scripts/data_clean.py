import polars as pl
from io import StringIO
import logging
from datetime import datetime
import re
import os
from dashboard.setup.utils import load_configuration

CONFIG_PATH = "dashboard/configuration/InitialConfig.json"
LOGGING_DIR = "Logs"
if not os.path.exists(LOGGING_DIR):
    os.makedirs(LOGGING_DIR)

log_filename = os.path.join(LOGGING_DIR, 'cleaning_script.log')
logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s: %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

def clean_and_convert_numeric(df, float_cols, int_cols):
    """
    Optimized function to clean and convert specified columns to numeric types in Polars.
    
    Parameters:
    - df: Polars DataFrame to process.
    - float_cols: List of column names to clean and convert to float.
    - int_cols: List of column names to clean and convert to int.
    
    Returns:
    - Polars DataFrame with cleaned and converted columns.
    """
    # Ensure only valid columns are processed
    valid_float_columns = list(set(float_cols).intersection(df.columns))
    valid_int_columns = list(set(int_cols).intersection(df.columns))
    
    # Cleaning and conversion expressions
    float_clean_exprs = [
        pl.col(col).str.replace_all(r'[^\d.-]', '').cast(pl.Float64).fill_nan(0.0).alias(col)
        for col in valid_float_columns
    ]
    
    int_clean_exprs = [
        pl.col(col).str.replace_all(r'[^\d.-]', '').cast(pl.Float64).fill_nan(0.0).cast(pl.Int64).alias(col)
        for col in valid_int_columns
    ]
    
    # Apply all transformations at once using with_columns
    df = df.with_columns(float_clean_exprs + int_clean_exprs)
    
    return df

def fix_price(df):
    """
    Ensure thatprices are non-negative.
    """
    price_mask = pl.col('price') > 0
    filtered_df = df.filter(price_mask)
    return filtered_df


def clean_text(dataframe, config):
    """
    Clean text in specified columns more efficiently.
    """
    # Copy only the text columns that need cleaning to avoid unnecessary data duplication
    columns_to_clean = [col for col in config["TextColumns"] if col in dataframe.columns]

    # Apply cleaning operations to each specified column
    for column_name in columns_to_clean:
        # Apply text transformations directly
        dataframe = dataframe.with_columns(
            pl.col(column_name)
            .str.to_lowercase()
            .str.strip_chars()  # Note: Use .strip() instead of .strip_chars() unless you have specific characters to strip
            .str.replace_all(r'\s+', ' ')  # Use raw string directly
            .alias(column_name)
        )

    # Log after cleaning all specified columns
    logging.info(f'Cleaned text in columns: {columns_to_clean}')

    return dataframe

def clean_part_numbers(data_frame):
    """
    Clean and validate part numbers using vectorized operations.
    """
    # Define a valid pattern for part numbers (alphanumeric and hyphen, 3-15 characters long)
    valid_pattern = r'^[A-Za-z0-9-]{3,15}$'

    # Filter DataFrame based on the basic valid part number pattern
    data_frame = data_frame.filter(pl.col('part_number').str.contains(valid_pattern))

    # Strip whitespace from both ends and replace multiple spaces with a single space
    data_frame = data_frame.with_columns(
        pl.col('part_number')
        .str.strip_chars()  # Strip whitespace from both ends
        .str.replace_all(r'\s+', ' ')  # Replace multiple spaces with a single space
        .alias('part_number')
    )

    return data_frame

def filter_consecutive_letters(data_frame):
    """
    Remove part numbers containing consecutive letters of 3 or more.
    """
    def has_consecutive_letters(part_number):
        return bool(re.search(r'[A-Za-z]{3,}', part_number))

    # Apply the filter function to remove invalid part numbers
    data_frame = data_frame.filter(
        ~pl.col('part_number').map_elements(has_consecutive_letters, return_dtype=pl.Boolean)
    )

    return data_frame


def merge_duplicate_parts(data_frame, config):
    """
    Merges duplicate part number information into a single unique row.
    """
    logging.info(f'Original Shape: {data_frame.shape}')

    # Categorize columns based on their intended aggregation method
    sales_columns = config['SalesColumns']
    specific_columns = ['quantity', 'quantity_ordered_ytd', 'special_orders_ytd']
    aggregation_columns = sales_columns + specific_columns
    non_aggregated_columns = [col for col in data_frame.columns if col not in aggregation_columns + ['part_number']]

    # Define aggregation expressions
    aggregations = [pl.col(col).sum().alias(col) for col in aggregation_columns]
    aggregations += [pl.col(col).first().alias(col) for col in non_aggregated_columns]
    
    # Make sure 'months_no_sale' is not duplicated by not adding it again if it's already handled
    if 'months_no_sale' not in aggregation_columns and 'months_no_sale' not in non_aggregated_columns:
        aggregations.append(pl.col('months_no_sale').min().alias('months_no_sale'))  # Custom rule for specific column

    # Perform the aggregation
    aggregated_df = data_frame.group_by('part_number').agg(aggregations)

    # Reorder columns to match the original order, ensuring all are present
    reordered_df = aggregated_df.select(data_frame.columns)

    logging.info(f'Aggregated Shape: {reordered_df.shape}')
    return reordered_df

def clip_sales_columns(dataframe, config):
    """
    Ensure that sales values are non-negative using Polars.
    """
    sales_columns = config["SalesColumns"]
    for column in sales_columns:
        dataframe = dataframe.with_columns(
            pl.when(pl.col(column) < 0).then(0).otherwise(pl.col(column)).alias(column)
        )
    return dataframe

def adjust_quantity(df):
    """
    Adjusts the quantity to 0 if it's negative and creates a negative_on_hand column to track
    the original negative quantities.
    """
    # Track negative quantities in a new column
    df = df.with_columns(
        pl.when(pl.col('quantity') < 0).then(pl.col('quantity')).otherwise(0).abs().alias('negative_on_hand')
    )
    
    # Set negative quantities to zero
    df = df.with_columns(
        pl.when(pl.col('quantity') < 0).then(0).otherwise(pl.col('quantity')).alias('quantity')
    )
    
    return df

def update_months_no_sale(df, sales_columns):
    logging.info("Updating months_no_sale...")
    current_month_index = datetime.now().month  # Adjust for 0-based index
    current_month_index += 12  # Account for two years of data

    # Select the relevant columns in reversed order up to the current month
    months_to_count = sales_columns[:current_month_index][::-1]

    # Define the function to count consecutive zeros
    def count_consecutive_zeros(series):
        data = series.to_numpy()
        counts = []
        for row in data:
            count = 0
            for sale in row:
                if sale == 0:
                    count += 1
                else:
                    break
            counts.append(count)
        return pl.Series(counts)

    # Use map_batches to apply the counting function on the reversed list of sales columns
    df = df.with_columns(
        pl.struct(months_to_count).map_batches(
            function=count_consecutive_zeros,
            return_dtype=pl.Int64
        ).alias("months_no_sale_computed")
    )

    # Update the months_no_sale column with the computed values
    df = df.with_columns(
        pl.when((pl.col('months_no_sale') <= len(months_to_count)) & (pl.sum_horizontal(pl.col(sales_columns)) == 0))
            .then(len(months_to_count))
            .otherwise(
                pl.when(pl.col('months_no_sale') > len(months_to_count))
                .then(pl.col('months_no_sale'))
                .otherwise(pl.col('months_no_sale_computed'))
            )
            .alias('months_no_sale')
        ).drop('months_no_sale_computed')

    logging.info("Update complete.")
    return df

def inventory_check(df, config):
    # Extract sales columns from the configuration
    sales_columns = config["SalesColumns"]
    
    # Calculate the sum of sales across all specified sales columns
    total_sales = pl.sum_horizontal(sales_columns)

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
    
    # Initial shape of the DataFrame
    logging.debug(f"Initial shape: {data_frame.shape}")
    
    # Clean part numbers
    data_frame = clean_part_numbers(data_frame)
    logging.debug(f"Shape after cleaning part numbers: {data_frame.shape}")

    data_frame = filter_consecutive_letters(data_frame)
    logging.debug(f"Shape after cleaning part numbers for consecutive letters: {data_frame.shape}")
    
    # Merge duplicate parts
    data_frame = merge_duplicate_parts(data_frame, config)
    logging.debug(f"Shape after merging duplicates: {data_frame.shape}")
    
    # Clean and convert numeric columns
    float_cols = config['FloatColumns']
    int_cols = config['IntColumns']
    data_frame = clean_and_convert_numeric(data_frame, float_cols, int_cols)
    logging.debug(f"Shape after cleaning and converting numeric columns: {data_frame.shape}")
    
    # Fix price
    data_frame = fix_price(data_frame)
    logging.debug(f"Shape after fixing price: {data_frame.shape}")
    
    # Clean text
    data_frame = clean_text(data_frame, config)
    logging.debug(f"Shape after cleaning text: {data_frame.shape}")
    
    # Clip sales columns
    data_frame = clip_sales_columns(data_frame, config)
    logging.debug(f"Shape after clipping sales columns: {data_frame.shape}")
    
    # Update months no sale
    sales_cols = config['SalesColumns']
    data_frame = update_months_no_sale(data_frame, sales_cols)
    
    # Adjust quantity
    data_frame = adjust_quantity(data_frame)
    logging.debug(f"Shape after adjusting quantity: {data_frame.shape}")
    
    # Shape before removing parts not in inventory
    logging.info(f"Shape before removing parts not in inventory: {data_frame.shape}")
    
    # Inventory check
    data_frame = inventory_check(data_frame, config)
    logging.info(f"Shape after removing parts not in inventory: {data_frame.shape}")
    
    logging.info("Data preprocessing completed successfully.")
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
        cleaned_data = preprocess_data(df, config)
        print('Data Cleaned Successfully!')
        cleaned_data.write_csv('/Users/skylerwilson/Desktop/PartsWise/Data/Processed/parts_data_cleaned.csv')
        cleaned_data_json = cleaned_data.write_json()

        return cleaned_data_json

    except ValueError as e:
        logging.error(f"Invalid JSON format: {e}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: Invalid JSON format.'})
        return False
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: {str(e)}'})
        return False

