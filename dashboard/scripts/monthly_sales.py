import pandas as pd
import logging
import os
import json
from datetime import datetime
import calendar

CONFIG_FILE = "dashboard/configuration/SeasonalConfig.json"
LOGGING_DIR = "Logs"

def prepare_and_melt_sales_data(df, config):
    """
    Prepare and melt sales data by combining this year's and last year's sales.

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
    df_this_year = pd.melt(df, id_vars=['part_number'], value_vars=this_year_sales_columns, var_name='month', value_name='quantity_sold')
    df_this_year['year'] = current_year

    # Melt last year's sales data
    df_last_year = pd.melt(df, id_vars=['part_number'], value_vars=last_year_sales_columns, var_name='month', value_name='quantity_sold')
    df_last_year['year'] = last_year

    # Map month abbreviations to full names
    month_mapping = {calendar.month_abbr[i].lower(): calendar.month_name[i] for i in range(1, 13)}
    df_this_year['month'] = df_this_year['month'].str.replace('sales_', '').map(month_mapping).str.lower()
    df_last_year['month'] = df_last_year['month'].str.replace('sales_last_', '').map(month_mapping).str.lower()

    # Combine the melted data for both years
    df_melted_sales = pd.concat([df_this_year, df_last_year], ignore_index=True)

    df_melted_sales['month'] = pd.Categorical(df_melted_sales['month'], categories=[calendar.month_name[i].lower() for i in range(1, 13)], ordered=True)
    df_melted_sales = df_melted_sales.sort_values(['year', 'month'])

    return df_melted_sales


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
    
def main(current_task, input_data):
    print('Monthly sales script running...')
    log_filename = os.path.join(LOGGING_DIR, 'monthly_script.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = load_configuration(CONFIG_FILE)
    if config is None:
        logging.error("Failed to load configuration.")
        return

    try:
        # Correct the line to load `input_data`
        original_data = json.loads(input_data)
        dataset = pd.DataFrame(original_data['data'], columns=original_data['columns'])
        df_melted_sales= prepare_and_melt_sales_data(dataset, config)
        
        output_file_path = "/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/data/output_data/sales_data.feather"
        
        df_melted_sales.to_feather(output_file_path)
        print("Feather file saved successfully.")

        # Serialize both DataFrames to JSON
        sales_json = df_melted_sales.to_json(orient='split')
    

        # Return the combined structure as a JSON string
        return sales_json

    except ValueError as e:
        logging.error(f"Invalid JSON format: {e}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: Invalid file format.'})
        return False
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: {str(e)}'})
        return False