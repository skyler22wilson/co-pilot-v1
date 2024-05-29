import pandas as pd
import logging
from datetime import datetime
import json
import numpy as np
import os
import re

CONFIG_PATH = "Dashboard/configuration/InitialConfig.json"
LOGGING_DIR = "Logs"
if not os.path.exists(LOGGING_DIR):
    os.makedirs(LOGGING_DIR)

log_filename = os.path.join(LOGGING_DIR, 'cleaning_script.log')
logging.basicConfig(filename=log_filename, 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def filter_conditions(dataframe, config):
    """
    drop parts that are more than likely not in inventory anymore.
    """
    data_frame = dataframe.copy()

    sales_columns = config['SalesColumns']

    no_sales_condition = ((data_frame['months_no_sale'] >= 24) & 
                          (data_frame[sales_columns].sum(axis=1) == 0) & 
                          (data_frame['quantity'] == 0) &
                          (data_frame['quantity_ordered_ytd'] == 0))
    
    data_frame = data_frame[~no_sales_condition]

    return data_frame

def clean_and_convert_numeric(df, columns):
    """
    Optimized function to clean and convert specified columns to numeric types.
    
    Parameters:
    - df: DataFrame to process.
    - columns: List of column names to clean and convert.
    
    Returns:
    - DataFrame with cleaned and converted columns.
    """
    # Filter out columns not in DataFrame
    valid_columns = [col for col in columns if col in df.columns]
    
    # Perform cleaning and conversion in a batch
    df[valid_columns] = df[valid_columns].replace('[^\d.-]', '', regex=True)
    df[valid_columns] = df[valid_columns].apply(pd.to_numeric, errors='coerce')
    
    # Separate handling for 'months_no_sale', if needed
    if 'months_no_sale' in valid_columns:
        df['months_no_sale'] = df['months_no_sale'].fillna(0).astype(int)
        valid_columns.remove('months_no_sale')  # Prevent dropping rows based on 'months_no_sale'
    
    # Drop rows where numeric conversion resulted in NaN, except for 'months_no_sale'
    df.dropna(subset=valid_columns, inplace=True)
    
    return df

def fix_price(df):
    price_mask = df['price'] > 0
    return df[price_mask]


def clean_text(dataframe, config):
    """
    Clean text in specified columns more efficiently.
    """
    # Copy only the text columns that need cleaning to avoid unnecessary data duplication
    columns_to_clean = [col for col in config["TextColumns"] if col in dataframe.columns and dataframe[col].dtype == 'object']
    data_frame = dataframe[columns_to_clean].copy()

    # Pre-compile the regular expression used in cleaning (if using complex expressions)
    whitespace_re = re.compile(r'\s+')

    for column_name in columns_to_clean:
        # Perform cleaning operations using vectorized methods
        cleaned_column = data_frame[column_name].str.lower().str.strip()
        cleaned_column = cleaned_column.str.replace(whitespace_re, ' ', regex=True)
        data_frame[column_name] = cleaned_column

    # Log after cleaning all specified columns
    logging.info(f'Cleaned text in columns: {columns_to_clean}')

    # Return the original dataframe with cleaned columns
    dataframe[columns_to_clean] = data_frame
    return dataframe

def clean_part_numbers(data_frame):
    """
    Clean and validate part numbers using vectorized operations.
    """
    # Define the valid part number pattern and compile the regex for space reduction
    valid_pattern = r'^(?=.*[0-9]).{3,15}$'
    space_reducer = re.compile(r'\s+')

    # Ensure part numbers are strings
    data_frame['part_number'] = data_frame['part_number'].astype(str)

    # Filter DataFrame based on valid part numbers using a pre-compiled pattern
    valid_mask = data_frame['part_number'].str.match(valid_pattern)
    data_frame = data_frame[valid_mask]

    # Clean part numbers by stripping leading/trailing spaces and replacing multiple spaces with a single space
    data_frame['part_number'] = data_frame['part_number'].str.strip().str.replace(space_reducer, ' ', regex=True)

    return data_frame

def adjust_part_numbers(data_frame):
    condition = (data_frame['part_number'].str.len() >= 8) & (data_frame['part_number'].str.lower().str.startswith('t'))
    data_frame.loc[condition, 'supplier_name'] = 'triumph'
    # Apply stripping here as well, if needed
    data_frame['part_number'] = data_frame['part_number'].str.strip()
    return data_frame


def merge_duplicate_parts(data_frame, config):
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

def clip_sales_columns(dataframe,config):
    """
    Ensure that sales values are non-negative.
    """
    data_frame = dataframe.copy()
    sales_columns = config["SalesColumns"]
    data_frame.loc[:, sales_columns] = data_frame[sales_columns].clip(lower=0)
    return data_frame

def adjust_quantity(dataframe):
    df = dataframe.copy()
    
    # Use vectorized operations for conditions
    df['negative_on_hand'] = df['quantity'].where(df['quantity'] < 0, 0).astype(int)
    df['quantity'] = df['quantity'].where(df['quantity'] >= 0, 0).astype(int)
    
    return df


def update_months_no_sale(row, sales_columns_this_year, sales_columns_last_year):
    current_month_index = datetime.now().month - 1 #set current month to march, this is for complete feb data

    if current_month_index == 1:
        # If it's currently January, count from last year's February to December
        months_to_count = list(reversed(sales_columns_last_year[1:]))
    else:
        # If it's February or later, count from this year's February up to the current month, then add last year's data
        months_to_count = list(reversed(sales_columns_this_year[:current_month_index - 1])) + list(reversed(sales_columns_last_year))
    
    combined_sales = list(months_to_count)

    if row['months_no_sale'] > 12:
        return row['months_no_sale']
    
    # Start counting from February
    months_no_sale = 0
    for sales in combined_sales:
        if row[sales] == 0:
            months_no_sale += 1
        else:
            break  # Stop counting if there's a sale

    return months_no_sale

def inventory_check(df, config):
    # Extract sales columns from the configuration
    sales_columns = config["SalesColumns"]
    
    # Determine rows where quantity is zero and there have been no sales across all sales columns
    no_inventory_no_sales = (df['quantity'] == 0) & (df[sales_columns].sum(axis=1) == 0) & (df['negative_on_hand'] == 0)
    
    # Drop these rows from the dataframe
    df = df.loc[~no_inventory_no_sales]
    
    return df

def preprocess_data(data_frame, config):
    """
    Apply preprocessing steps to the data.
    """
    data_frame = filter_conditions(data_frame, config)
    data_frame = clean_part_numbers(data_frame)
    data_frame = merge_duplicate_parts(data_frame, config)
    columns = config['IntColumns'] + config['FloatColumns'] + config['SalesColumns']
    data_frame = clean_and_convert_numeric(data_frame, columns)
    data_frame = fix_price(data_frame)
    data_frame = adjust_part_numbers(data_frame)
    data_frame = clean_text(data_frame, config)
    data_frame = clip_sales_columns(data_frame, config)
    sales_this_year = config['SalesThisYear']
    sales_last_year = config['SalesLastYear']
    data_frame['months_no_sale'] = data_frame.apply(
    lambda row: update_months_no_sale(row, sales_this_year, sales_last_year),
    axis=1).astype(int)
    data_frame = adjust_quantity(data_frame)
    data_frame['description'] = data_frame['description'].fillna('No Description')
    logging.info(f'Shape before removing items not in inventory: {data_frame.shape}')
    data_frame = inventory_check(data_frame, config)
    logging.info(f'Shape after removing items not in inventory: {data_frame.shape}')

    logging.info('Data preprocessing completed successfully.')
    return data_frame

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

def read_and_process_data(df, config):
    try:
        original_column_names = df.columns
        columns_config = config.get("columns")
        if not columns_config:
            raise KeyError("columns configuration not found in the config file.")

        # Generate new column names from the config
        new_column_names = list(columns_config.keys())

        # Check if the number of columns in the CSV matches the number in the configuration
        if len(original_column_names) != len(new_column_names):
            raise ValueError("The number of columns in the CSV does not match the configuration.")

        # Map original column names to new column names
        column_mapping = dict(zip(original_column_names, new_column_names))
        df = df.rename(columns=column_mapping)

        # Coerce data types of columns as per configuration
        for col, dtype in list(config["columns"].items()):
            if col in df.columns:
                if dtype == "float":
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
                elif dtype == "int":
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            else:
                print(f"Column '{col}' not found in the data file.")

        return df

    except Exception as e:
        print(f"Error processing data file: {e}")
        return pd.DataFrame()

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
        print('Loading the JSON data...')
        data = json.loads(input_data)  # Correctly load the JSON string into a Python dictionary
        df = pd.DataFrame(data['data'], columns=data['columns'])
        print(f'Check if conversion to data frame worked: {type(df)}')
        print('data loaded successfully!')
        # Preprocess and clean data as before
        preprocessed_data = read_and_process_data(df, config)
        cleaned_data = preprocess_data(preprocessed_data, config)
        # Use a temporary file for output
        cleaned_data_json = cleaned_data.to_json(orient='split')
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

