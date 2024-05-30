import os
import pandas as pd
import logging
import numpy as np
import json
from scipy.stats import norm

CONFIG_FILE = "Dashboard/configuration/SeasonalConfig.json"
LOGGING_DIR = "Logs"

def calculate_safety_stock(df, service_level=0.75, max_lead_time=90, min_lead_time=10):
    z_score = norm.ppf(service_level)
    average_lead_time_days = (min_lead_time + max_lead_time) / 2
    lead_time_std = (max_lead_time - min_lead_time) / np.sqrt(12)
    demand_score = df['demand']
    
    daily_demand = (df['rolling_12_month_sales'] / 365) * demand_score
    std_dev_demand = np.std(daily_demand) / 365  # Adjust this if daily_sales is not the correct column

    df['safety_stock'] = z_score * (std_dev_demand * np.sqrt(average_lead_time_days) + daily_demand * lead_time_std) 
    df['safety_stock'] = df['safety_stock'].astype(int)
    df.loc[df['months_no_sale'] >= 7, 'safety_stock'] = 0

    return df

def calculate_reorder_point(df, min_lead_time=10, max_lead_time=90):
    demand_score = df['demand']
    daily_demand = (df['rolling_12_month_sales'] / 365) * demand_score
    average_lead_time_days = (min_lead_time + max_lead_time) / 2
    lead_time_demand = daily_demand * average_lead_time_days

    df['reorder_point'] = (lead_time_demand + df['safety_stock']).astype(int)
    df.loc[df['months_no_sale'] >= 12, 'reorder_point'] = 0

    return df

def main(current_task, input_data):
    print('Reorder Point Script Running...')
    # Initialize logging
    
    log_filename = os.path.join(LOGGING_DIR, 'reorder_script.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Script started.")

    try:
        print('loading data...')
        data = json.loads(input_data)
        dataset = pd.DataFrame(data['data'], columns=data['columns'])

        print('dataset loaded. Loading month data now...')
        # Calculate safety stock and reorder points
        dataset = calculate_safety_stock(dataset)
        dataset = calculate_reorder_point(dataset, dataset['safety_stock'])
        logging.info(f"Length of reorder JSON file: {len(dataset)} rows")
        
        json_data = dataset.to_json(orient='split')
        
        return json_data

    except ValueError as e:
        logging.error(f"Invalid JSON format: {e}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: Invalid file format.'})
        return False
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: {str(e)}'})
        return False


