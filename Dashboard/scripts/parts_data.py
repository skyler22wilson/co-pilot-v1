import json
import logging
import pandas as pd
import os

CONFIG_FILE = "Dashboard/configuration/SeasonalConfig.json"
LOGGING_DIR = "Logs"

def load_configuration(config_path):
    """Load the configuration file."""
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

def filter_non_sales_columns(df, config):
    """Filter out sales columns using configuration."""
    sales_last_year = config.get('SalesLastYear', [])
    sales_this_year = config.get('SalesThisYear', [])
    sales_columns = sales_last_year + sales_this_year

    columns_to_keep = [col for col in df.columns if col not in sales_columns]
    non_sales_data = df[columns_to_keep]

    return non_sales_data

def parts_data_task(input_data, config):
    """Task to process and return non-sales (parts) data."""
    print('Parts data script running...')
    log_filename = os.path.join(LOGGING_DIR, 'parts_data_script.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Load and process the input data
        original_data = json.loads(input_data)
        dataset = pd.DataFrame(original_data['data'], columns=original_data['columns'])

        # Filter out the sales columns
        non_sales_data = filter_non_sales_columns(dataset, config)

        # Save to a feather file if needed
        output_file_path = "/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/data/output_data/parts_data.feather"
        non_sales_data.to_feather(output_file_path)
        print("Parts feather file saved successfully.")

        # Return the non-sales data as a JSON string
        parts_data_json = non_sales_data.to_json(orient='split')
        return parts_data_json

    except ValueError as e:
        logging.error(f"Invalid JSON format: {e}")
        return False
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        return False

def main(current_task, input_data):
    """Main function to orchestrate the parts data extraction task."""
    print('Running the parts data task...')
    log_filename = os.path.join(LOGGING_DIR, 'parts_data_script.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = load_configuration(CONFIG_FILE)
    if config is None:
        logging.error("Failed to load configuration.")
        return False

    try:
        parts_data_json = parts_data_task(input_data, config)
        return parts_data_json

    except ValueError as e:
        logging.error(f"Invalid JSON format: {e}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: Invalid file format.'})
        return False
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: {str(e)}'})
        return False