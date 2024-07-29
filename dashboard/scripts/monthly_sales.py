import polars as pl
import logging
import os
import json
from io import StringIO
from dashboard.setup.utils import create_long_form_dataframe, load_configuration

CONFIG_FILE = "dashboard/configuration/SeasonalConfig.json"
LOGGING_DIR = "Logs"

    
def create_monthly_sales(current_task, input_data):
    print('Monthly sales script running...')
    log_filename = os.path.join(LOGGING_DIR, 'monthly_script.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = load_configuration(CONFIG_FILE)
    if config is None:
        logging.error("Failed to load configuration.")
        return

    try:
        dataset = pl.read_json(StringIO(input_data))
        monthly_data = create_long_form_dataframe(dataset)
        monthly_data = monthly_data.drop(['month_number', 'day', 'date'])
        
        # Serialize both DataFrames to JSON
        sales_json = monthly_data.write_json()
    

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