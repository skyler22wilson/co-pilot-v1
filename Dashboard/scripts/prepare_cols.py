import pandas as pd
import logging
import numpy as np
import json
import os
import pyfftw
import calendar
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

CONFIG_FILE = "Dashboard/configuration/SeasonalConfig.json"
LOGGING_DIR = "Logs"


def calculate_rolling_12_month_sales(df):
    print('Calculate 12 month rolling')
    now = datetime.now()
    current_month = now.month - 1 # February is the last completed month (2)

    # Create lists to hold sales columns from the last 12 months
    if current_month == 1:
        # If the current month is January, the last 12 months would be all of last year
        this_year_sales_columns = []
        last_year_sales_columns = [f'sales_last_{calendar.month_abbr[i].lower()}' for i in range(1, 13)]
    else:
        # Sales columns from last year from last completed month last year to December
        last_year_sales_columns = [f'sales_last_{calendar.month_abbr[i].lower()}' for i in range(current_month + 1, 13)]
        # Sales columns from this year from January to last completed month
        this_year_sales_columns = [f'sales_{calendar.month_abbr[i].lower()}' for i in range(1, current_month + 1)]

    # Concatenate the column lists appropriately
    rolling_columns = last_year_sales_columns + this_year_sales_columns

    # Calculate the rolling 12-month sales for each part
    df['rolling_12_month_sales'] = df[rolling_columns].sum(axis=1)
    print('12 month rolling calculated')

    return df

def calculate_3_month_rolling_sales(df):
    print('Calculating 3 month rolling sales')
    # Get the current month, adjusted for having only completed data through February
    current_month = datetime.now().month - 1  # -1 because last complete month is February if running in March

    # Define sales columns for the last three months
    if current_month <= 3:
        # If in the first three months of the year, calculate indexes for the remaining months from last year
        last_year_months = list(range(12 - (3 - current_month), 13))  # last months of the previous year
        last_year_sales_columns = [f'sales_last_{calendar.month_abbr[i].lower()}' for i in last_year_months]
        this_year_sales_columns = [f'sales_{calendar.month_abbr[i].lower()}' for i in range(1, current_month + 1)]
        sales_columns = last_year_sales_columns + this_year_sales_columns
    else:
        # Otherwise, just get the last three months of this year
        sales_columns = [f'sales_{calendar.month_abbr[i].lower()}' for i in range(current_month - 2, current_month + 1)]

    # Calculate the rolling 3-month sales for each part
    df['rolling_3_month_sales'] = df[sales_columns].sum(axis=1)
    print('3 month rolling sales calculated')

    return df

def calculate_unit_cost(df: pd.DataFrame) -> pd.DataFrame:
    df['cost_per_unit'] = np.round(df['price'] - df['margin'], 2)
    return df

def calculate_total_cost(df: pd.DataFrame) -> pd.DataFrame:
    df['total_cost'] = np.round(df['cost_per_unit'] * df['quantity'], 2)
    return df

def calculate_sales_revenue(df: pd.DataFrame) -> pd.DataFrame:
    df['sales_revenue'] = np.round(df['rolling_12_month_sales'] * df['price'], 2)
    return df

def calculate_cogs(df: pd.DataFrame) -> pd.DataFrame:
    df['cogs'] = np.round(df['rolling_12_month_sales'] * df['cost_per_unit'], 2)
    return df

def calculate_margin(df: pd.DataFrame) -> pd.DataFrame:
    df['margin_percentage'] = np.round(((df['price'] - df['cost_per_unit']) / df['price']) * 100, 2)
    df['margin'] = np.round(df['margin'], 2)
    return df

def compute_gross_profit(df: pd.DataFrame) -> pd.DataFrame:
    df['gross_profit'] = np.round((df['price'] - df['cost_per_unit']) * df['rolling_12_month_sales'], 2)
    return df

def calc_roi(df: pd.DataFrame) -> pd.DataFrame:
    df['roi'] = np.where(df['total_cost'] != 0,
                         np.round(((df['gross_profit'] - df['total_cost']) / df['total_cost']) * 100, 2), 0) 
    return df

def calculate_day_supply(df):

    avg_daily_sales = df['rolling_12_month_sales'] / 365

    # Calculate days' supply based on current inventory levels
    df['365_days_supply'] = df['quantity'] / avg_daily_sales
    df['365_days_supply'] = df['365_days_supply'].replace([np.inf, -np.inf], np.nan).fillna(0)  # Handle divisions by zero or missing sales data
    df['365_days_supply'] = np.round(df['365_days_supply'], 1)

    avg_daily_3_mo_sales = df['rolling_3_month_sales'] / 90

    # Calculate days' supply based on current inventory levels
    df['90_days_supply'] = df['quantity'] / avg_daily_3_mo_sales
    df['90_days_supply'] = df['90_days_supply'].replace([np.inf, -np.inf], np.nan).fillna(0)  # Handle divisions by zero or missing sales data
    df['90_days_supply'] = np.round(df['90_days_supply'], 1)
    return df


def calculate_turnover(df):
    # Calculate annual turnover using 365 days' supply
    if '365_days_supply' in df.columns:
        df['12_month_turnover'] = 365 / df['365_days_supply']
        df['12_month_turnover'] = df['12_month_turnover'].replace([np.inf, -np.inf], np.nan).fillna(0)
        df['12_month_turnover'] = np.round(df['12_month_turnover'], 1)
    else:
        logging.error("Annual days' supply data is missing")

    # Calculate 90-day turnover using 90 days' supply
    if '90_days_supply' in df.columns:
        df['90_day_turnover'] = 90 / df['90_days_supply']
        df['90_day_turnover'] = df['90_day_turnover'].replace([np.inf, -np.inf], np.nan).fillna(0)
        df['90_day_turnover'] = np.round(df['90_day_turnover'], 1)
    else:
        logging.error("90 days' supply data is missing")

    return df

def sales_2_stock(df):
    if 'rolling_12_month_sales' not in df.columns:
        logging.error("'rolling_12_month_sales' column is missing.")
        return df 
    df["sales_to_stock_ratio"] = np.where(df['quantity'] > 0, df['rolling_12_month_sales'] / df['quantity'], 0)
    return df

def order_2_sales(df):
    if 'rolling_12_month_sales' not in df.columns:
        logging.error("'rolling_12_month_sales' column is missing.")
        return df 
    df["order_to_sales_ratio"] = np.where(df['rolling_12_month_sales'] > 0, df['quantity_ordered_ytd'] / df['rolling_12_month_sales'], 0)

    return df

def prepare_and_melt_sales_data(df):
    # Make a copy of the original DataFrame
    df_copy = df.copy()

    # Define the current year and last year for labeling purposes
    current_year = datetime.now().year
    last_year = current_year - 1
    current_month_index = datetime.now().month

    # Define sales columns for this year up to the current month (not including) and all of last year
    this_year_sales_columns = [f'sales_{calendar.month_abbr[i].lower()}' for i in range(1, current_month_index + 1)]
    last_year_sales_columns = [f'sales_last_{calendar.month_abbr[i].lower()}' for i in range(1, 13)]

    # Melt this year's sales data
    df_this_year = pd.melt(df_copy, id_vars=['part_number'], value_vars=this_year_sales_columns, var_name='month', value_name='quantity_sold')
    df_this_year['year'] = current_year

    # Melt last year's sales data
    df_last_year = pd.melt(df_copy, id_vars=['part_number'], value_vars=last_year_sales_columns, var_name='month', value_name='quantity_sold')
    df_last_year['year'] = last_year

    # Map month abbreviations to full names
    month_mapping = {calendar.month_abbr[i].lower(): calendar.month_name[i] for i in range(1, 13)}
    df_this_year['month'] = df_this_year['month'].str.replace('sales_', '').map(month_mapping)
    df_last_year['month'] = df_last_year['month'].str.replace('sales_last_', '').map(month_mapping)

    # Combine the melted data for both years
    df_melted_sales = pd.concat([df_this_year, df_last_year], ignore_index=True)

    monthly_columns_to_drop = this_year_sales_columns + last_year_sales_columns
    df = df.drop(columns=monthly_columns_to_drop)
    return df, df_melted_sales

def extract_seasonal_component(sales_data, time_column, value_column, top_n_freq=1):
    
    # If the time column contains month names, convert them to numerical values
    if sales_data[time_column].dtype == object:
        month_to_num = {month: index for index, month in enumerate(calendar.month_name[1:], start=1)}
        sales_data[time_column] = sales_data[time_column].map(month_to_num)
    
    # Ensure sales_data is sorted by time_column to maintain correct order for Fourier analysis
    sales_data_sorted = sales_data.sort_values(by=time_column)
    y = sales_data_sorted[value_column].values
    
    # Prepare the array for FFT. PyFFTW works with numpy arrays.
    y_fft = pyfftw.empty_aligned(len(y), dtype='complex128')
    y_fft[:] = y
    
    # Perform the FFT using PyFFTW
    fft_object = pyfftw.builders.rfft(y_fft)
    fft_values = fft_object()
    
    # The frequency bins are not affected by the FFT library used, so we can still use numpy for this part
    fft_freq = np.fft.rfftfreq(len(y))
    
    # Identify dominant frequencies
    top_indices = np.argsort(-np.abs(fft_values))[:top_n_freq]
    top_freq = fft_freq[top_indices]
    
    amplitudes = np.abs(fft_values[top_indices])
    phases = np.angle(fft_values[top_indices])
    
    time_points = np.arange(len(y))
    frequencies = fft_freq[top_indices]

    seasonal_component = np.sum(amplitudes * np.cos(2 * np.pi * frequencies[:, np.newaxis] * time_points + phases[:, np.newaxis]), axis=0)
    seasonal_component_mean = np.mean(seasonal_component)
    
    return seasonal_component_mean


def calculate_additional_metrics(df) -> pd.DataFrame:
    df = calculate_rolling_12_month_sales(df)
    df = calculate_3_month_rolling_sales(df)
    df = calculate_unit_cost(df)
    df = calculate_total_cost(df)
    df = calculate_sales_revenue(df)
    df = calculate_cogs(df)
    df = calculate_margin(df)
    df = compute_gross_profit(df)
    df = calc_roi(df)
    df = calculate_day_supply(df)
    df = calculate_turnover(df)
    df = sales_2_stock(df)
    df = order_2_sales(df)
    
    return df

def load_configuration(config_path):
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        return config
    except Exception as e:
        logging.error("Error loading configuration: %s", str(e))
        return None

def main(current_task, input_data):
    print('Preparing Columns...')
    warnings.filterwarnings("ignore")
    log_filename = os.path.join(LOGGING_DIR,'prepare_cols_script.log')
    logging.basicConfig(filename=log_filename, 
                        level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    config = load_configuration(CONFIG_FILE)
    if not config:
        logging.error("Failed to load the configuration. Exiting.")
        return

    try:
        data = json.loads(input_data)  # Correctly load the JSON string into a Python dictionary
        parts_data = pd.DataFrame(data['data'], columns=data['columns'])
        parts_data_final = calculate_additional_metrics(parts_data)
    
        parts_data, df_melted_sales = prepare_and_melt_sales_data(parts_data)

        seasonal_components = df_melted_sales.groupby('part_number').apply(
            lambda group: extract_seasonal_component(group, 'month', 'quantity_sold')
        ).reset_index(name='seasonal_component')

        if 'seasonal_component' in parts_data:
            return parts_data
        else:
            # Merge seasonal components back to the original DataFrame
            parts_data_with_seasonal = parts_data_final.merge(seasonal_components, on='part_number', how='left')
        
        # Save the DataFrame with seasonal components to output_data
        parts_data_with_seasonal = parts_data_with_seasonal.to_json(orient='split')
        # Return success at the end
        return parts_data_with_seasonal

    except ValueError as e:
        logging.error(f"Invalid JSON format: {e}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: Invalid file format.'})
        return False
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: {str(e)}'})
        return False
