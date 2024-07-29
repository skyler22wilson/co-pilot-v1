import os
import polars as pl
import logging
import numpy as np
import json
from scipy.stats import norm
from io import StringIO

CONFIG_FILE = "dashboard/configuration/SeasonalConfig.json"
LOGGING_DIR = "Logs"

def calculate_safety_stock(df, service_level=0.75, max_lead_time=90, min_lead_time=10):
    z_score = norm.ppf(service_level)
    average_lead_time_days = (min_lead_time + max_lead_time) / 2
    lead_time_std = (max_lead_time - min_lead_time) / np.sqrt(12)
    
    daily_demand = (pl.col('rolling_12m_sales') / 365) * pl.col('demand')
    std_dev_demand = daily_demand.std() / 365  # Adjust this if daily_sales is not the correct column

    safety_stock = z_score * (std_dev_demand * np.sqrt(average_lead_time_days) + daily_demand * lead_time_std)
    df = df.with_columns(
        safety_stock.cast(pl.Int32).alias('safety_stock')
    )
    df = df.with_columns(
        pl.when(pl.col('months_no_sale') >= 7).then(0).otherwise(pl.col('safety_stock')).alias('safety_stock')
    )

    return df

def calculate_reorder_point(df, min_lead_time=10, max_lead_time=90):
    daily_demand = (pl.col('rolling_12m_sales') / 365) * pl.col('demand')
    average_lead_time_days = (min_lead_time + max_lead_time) / 2
    lead_time_demand = daily_demand * average_lead_time_days

    reorder_point = (lead_time_demand + pl.col('safety_stock')).cast(pl.Int32)
    df = df.with_columns(
        reorder_point.alias('reorder_point')
    )
    df = df.with_columns(
        pl.when(pl.col('months_no_sale') >= 12).then(0).otherwise(pl.col('reorder_point')).alias('reorder_point')
    )

    return df.select('part_number', 'safety_stock', 'reorder_point')

def main(current_task, input_data):
    print('Reorder Point Script Running...')
    # Initialize logging
    
    log_filename = os.path.join(LOGGING_DIR, 'reorder_script.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Script started.")

    try:
        print('loading data...')
        print(f'Input data type: {type(input_data)}')
        df = pl.read_json(StringIO(input_data))
        print(df.head())

        print('dataset loaded. Loading month data now...')
        # Calculate safety stock and reorder points
        df = calculate_safety_stock(df)
        df = calculate_reorder_point(df)
        
        
        # Convert DataFrame to JSON
        json_data = df.write_json()

        return json_data

    except ValueError as e:
        logging.error(f"Invalid JSON format: {e}")
        return False
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        return False


