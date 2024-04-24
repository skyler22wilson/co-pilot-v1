import pandas as pd
import logging
import json
import os

CONFIG_FILE = "Dashboard/configuration/SeasonalConfig.json"
LOGGING_DIR = "Logs"

def categorizer(df):
    # Ensure that required columns are present
    required_columns = ['months_no_sale', 'demand']
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns {missing_columns} in the dataframe")

    # Compute the demand threshold for a stricter cutoff
    demand_threshold = df['demand'].quantile(0.95)
    print(demand_threshold)

    # Initialize the 'Inventory Category' column to 'Non-essential'
    df['inventory_category'] = 'non-essential'
    
    # Assign 'Obsolete' category
    df.loc[df['months_no_sale'] >= 12, 'inventory_category'] = 'obsolete'
    
    # Assign 'Nearing Obsolete' category
    df.loc[(df['months_no_sale'] >= 7) & (df['months_no_sale'] < 12), 'inventory_category'] = 'nearing_obsolete'
    
    # Assign 'Essential' category based on stricter criteria
    essential_criteria = (
        (df['months_no_sale'] <= 6) &
        (df['demand'] > demand_threshold) 
    )
    df.loc[essential_criteria, 'inventory_category'] = 'essential'

    return df


def main(current_task, input_data):
    print('Creating Categories')
    try:
        # Set up logging
        log_filename = os.path.join(LOGGING_DIR, 'categorizer_script.log')  # Assume LOGGING_DIR is defined
        logging.basicConfig(filename=log_filename, level=logging.INFO)

        # Load dataset
        original_data = json.loads(input_data)  
        dataset = pd.DataFrame(original_data['data'], columns=original_data['columns'])

        current_task.update_state(state='PROGRESS', meta={'message': 'Creating inventory categories...'})

        dataset = categorizer(dataset)  # Assuming categorizer is defined elsewhere

        # Convert dataset to JSON string for return
        json_dataset = dataset.to_json(orient='split') 

        return json_dataset

    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        current_task.update_state(state='FAILURE', meta={'message': f'Error processing data: {str(e)}'})
        return False


