import logging
import polars as pl
import os
from io import StringIO
from datetime import datetime, timezone
from dashboard.setup.utils import create_long_form_dataframe, load_configuration
import json

CONFIG_FILE = "dashboard/configuration/SeasonalConfig.json"
LOGGING_DIR = "Logs"
    
def create_parts_df(df):
    """
    Create parts DataFrame from the input DataFrame.
    
    Args:
        df (pl.DataFrame): Input DataFrame containing parts data.
    
    Returns:
        pl.DataFrame: DataFrame with parts information.

    Columns:
        - part_number: Unique identifier for each part
        - description: Description of the part
        - supplier_name: Name of each brand for each part
        - price: Price of each part
        - cost_per_unit: unit cost of each part
    """
    parts_df = df.select(['part_number', 'description', 'supplier_name', 'price', 'cost_per_unit'])
    return parts_df

def create_inventory_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create inventory DataFrame from the input DataFrame.
    
    Args:
        df (pl.DataFrame): Input DataFrame containing inventory data.
    
    Returns:
        pl.DataFrame: DataFrame with inventory information.

    Columns:
        - part_number: Unique identifier for each part
        - quantity: Quantity of each part
        - reorder_point: Stock level where the part should be reordered
        - safety_stock: The amount of stock kept on hand to prevent stock outs
        - negative_on_hand: Parts where negative on hand is greater than 0
        - part_status: Status of the inventory (essential, active, idle, obsolete)
    """
    inventory_df = df.select(['part_number', 'quantity', 'reorder_point', 'safety_stock', 'negative_on_hand', 'part_status'])
    return inventory_df

def create_metrics_df(df):
    """
    Create a DataFrame containing various metrics for parts.

    This function extracts specific columns from the input DataFrame to create
    a new DataFrame with metrics such as ROI, demand, obsolescence risk, etc.

    Args:
        df (polars.DataFrame): The input DataFrame containing all part data.

    Returns:
        polars.DataFrame: A new DataFrame containing only the metrics columns.

    Columns:
        - part_number: Unique identifier for each part
        - roi: Return on Investment
        - demand: Demand for the part
        - obsolescence_risk: Risk of the part becoming obsolete
        - days_of_inventory_outstanding: Number of days inventory is held
        - sell_through_rate: Rate at which inventory is sold
        - order_to_sales_ratio: Ratio of orders to sales
    """
    metrics_df = df.select(['part_number', 'roi', 'demand', 'obsolescence_risk', 'days_of_inventory_outstanding', 'sell_through_rate', 'order_to_sales_ratio'])
    return metrics_df

def create_temporal_metrics(df):
    """
    Create a DataFrame containing temporal metrics for parts.

    This function extracts time-based metrics from the input DataFrame to create
    a new DataFrame with various sales and supply metrics over different time periods.

    Args:
        df (polars.DataFrame): The input DataFrame containing all part data.

    Returns:
        polars.DataFrame: A new DataFrame containing only the temporal metrics columns.

    Columns:
        - part_number: Unique identifier for each part
        - months_no_sale: Number of months with no sales
        - sales_volatility: Measure of sales volatility
        - sales_trend: Overall sales trend
        - recent_sales_trend: Recent sales trend
        - twelve_m_days_supply: 12-month days of supply
        - three_m_days_supply: 3-month days of supply
        - one_m_days_supply: 1-month days of supply
        - three_m_turnover: 3-month inventory turnover
        - turnover: Overall inventory turnover
    """
    temporal_metrics_df = df.select(['part_number', 'months_no_sale', 'sales_volatility', 'sales_trend', 'recent_sales_trend', '12m_days_supply', '3m_days_supply', '1m_days_supply', '3m_turnover', 'turnover'])
    return temporal_metrics_df

def create_sales_df(df):
    """
    Create a long-form sales DataFrame.

    This function transforms the input DataFrame into a long-form sales DataFrame,
    dropping unnecessary columns in the process.

    Args:
        df (polars.DataFrame): The input DataFrame containing all part data.

    Returns:
        polars.DataFrame: A new long-form DataFrame containing sales data.

    Note:
        This function relies on an external 'create_long_form_dataframe' function
        to perform the initial transformation. It then drops 'month_number', 'day',
        and 'date' columns from the resulting DataFrame.
    """
    try:
        sales_df = create_long_form_dataframe(df)
        columns_to_drop = [col for col in ['date', 'month_number'] if col in sales_df.columns]
        sales_df = sales_df.drop(columns_to_drop)
        return sales_df
    except Exception as e:
        logging.error(f"Error in create_sales_df: {str(e)}")
        raise


def main(current_task, input_data):
    print('Dataframe creation task...')
    log_filename = os.path.join(LOGGING_DIR, 'parts_data_script.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = load_configuration(CONFIG_FILE)
    if config is None:
        logging.error("Failed to load configuration.")
        return False

    try:
        df = pl.read_json(StringIO(input_data))
        logging.info(f"Successfully read JSON data. Columns: {df.columns}")
        
        result = {}
        update_times = {}
        for name, func in [
            ('parts', create_parts_df),
            ('inventory', create_inventory_df),
            ('metrics', create_metrics_df),
            ('temporal', create_temporal_metrics),
            ('sales', create_sales_df)
        ]:
            try:
                result[name] = func(df).to_dict(as_series=False)
                update_times[name] = datetime.now().isoformat()
                logging.info(f"Successfully created {name} dataframe")
            except Exception as e:
                logging.error(f"Error creating {name} dataframe: {str(e)}")
                raise

        # Create a DataFrame for update times
        update_times_df = pl.DataFrame({
            'table_name': list(update_times.keys()),
            'update_time': list(update_times.values())
        })
        result['update_times'] = update_times_df.to_dict(as_series=False)
        
        # Save the data as JSON
        output_path = '/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/data/output_data/sql_ready_data.json'
        with open(output_path, 'w') as f:
            json.dump(result, f)
        
        logging.info(f"Successfully wrote SQL-ready data to {output_path}")

        return {'output_path': output_path, 'update_times': update_times}

    except ValueError as e:
        logging.error(f"Invalid JSON format: {e}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: Invalid file format.'})
        return False
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: {str(e)}'})
        return False