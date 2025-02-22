import os
import pandas as pd
import calendar
from datetime import datetime
import uuid
import logging
import json

LOGGING_DIR = "Logs"
CONFIG_FILE = "dashboard/configuration/SeasonalConfig.json" 

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

def prepare_and_melt_sales_data_by_supplier(df, config):
    """
    Prepare and melt sales data by combining this year's and last year's sales, grouped by supplier.

    :param df: DataFrame containing the initial parts/sales data
    :param config: Configuration dictionary specifying the column details
    :return: Melted sales DataFrame
    """
    current_year = datetime.now().year
    last_year = current_year - 1
    current_month_index = datetime.now().month

    # Define sales columns for this year and last year based on configuration
    this_year_sales_columns = config.get('SalesThisYear', [f'sales_{calendar.month_abbr[i].lower()}' for i in range(1, current_month_index + 1)])
    last_year_sales_columns = config.get('SalesLastYear', [f'sales_last_{calendar.month_abbr[i].lower()}' for i in range(1, 13)])

    # Melt this year's sales data
    df_this_year = pd.melt(df, id_vars=['supplier_name'], value_vars=this_year_sales_columns, var_name='month', value_name='quantity_sold')
    df_this_year['year'] = current_year

    # Melt last year's sales data
    df_last_year = pd.melt(df, id_vars=['supplier_name'], value_vars=last_year_sales_columns, var_name='month', value_name='quantity_sold')
    df_last_year['year'] = last_year

    # Map month abbreviations to full names
    month_mapping = {calendar.month_abbr[i].lower(): calendar.month_name[i] for i in range(1, 13)}
    df_this_year['month'] = df_this_year['month'].str.replace('sales_', '').map(month_mapping).str.lower()
    df_last_year['month'] = df_last_year['month'].str.replace('sales_last_', '').map(month_mapping).str.lower()

    # Combine the melted data for both years
    df_melted_sales = pd.concat([df_this_year, df_last_year], ignore_index=True)

    return df_melted_sales

def sales_data_by_supplier(df, config):
    """
    Aggregate melted sales data grouped by supplier.

    :param df: DataFrame containing the initial parts/sales data
    :param config: Configuration dictionary specifying the column details
    :return: Aggregated sales summary DataFrame grouped by supplier
    """
    # Prepare and melt the sales data
    df_melted_sales = prepare_and_melt_sales_data_by_supplier(df, config)

    # Group by supplier_name, year, and month to summarize sales
    summary_df = df_melted_sales.groupby(['supplier_name', 'year', 'month']).agg(
        quantity_sold=('quantity_sold', 'sum')
    ).reset_index()

    # Assign a UUID to each supplier
    summary_df['supplier_id'] = summary_df['supplier_name'].apply(lambda x: str(uuid.uuid5(uuid.NAMESPACE_DNS, x)))

    # Map month names to numbers for sorting
    month_to_number = {name.lower(): num for num, name in enumerate(calendar.month_name) if name}
    summary_df['month_number'] = summary_df['month'].map(month_to_number)

    # Sort the DataFrame first by month_number, then by supplier_name
    summary_df = summary_df.sort_values(by=['month_number', 'supplier_name'])
    summary_df.drop(columns=['month_number'], inplace=True)

    return summary_df

# Function to test the data processing
def main(current_task, input_data):
    print('Starting data processing workflow...')
    log_filename = os.path.join(LOGGING_DIR, 'supplier_sales_summary.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = load_configuration(CONFIG_FILE)
    if config is None:
        logging.error("Failed to load configuration.")
        return

    try:
        # Load input data
        original_data = json.loads(input_data)
        dataset = pd.DataFrame(original_data['data'], columns=original_data['columns'])

        # Log dataset details
        logging.info(f"Initial dataset loaded with {len(dataset)} rows.")

        # Generate sales data by supplier
        sales_summary_df = sales_data_by_supplier(dataset, config)
        sales_summary_df_path = "/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/data/processed_data/sales_summary.feather"
        sales_summary_df.to_feather(sales_summary_df_path)
        logging.info("Sales summary table saved successfully.")

        sales_summary_json = sales_summary_df.to_json(orient='split')

        # Log completion
        logging.info("Data processing completed successfully.")

        # Return the combined structure as a JSON string
        return sales_summary_json

    except ValueError as e:
        logging.error(f"Invalid JSON format: {e}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: Invalid file format.'})
        return False
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: {str(e)}'})
        return False
