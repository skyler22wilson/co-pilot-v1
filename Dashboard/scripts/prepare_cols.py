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
    """
    Calculate the rolling 12-month sales for each part in the DataFrame.

    This function computes the sum of sales over the last 12 months for each part.
    The sales columns are dynamically determined based on the current month.

    Args:
        df (pd.DataFrame): DataFrame containing the sales data with columns 
                           named in the format 'sales_{month_abbr}' for current year
                           and 'sales_last_{month_abbr}' for last year.

    Returns:
        pd.DataFrame: DataFrame with an additional column 'rolling_12_month_sales'
                      representing the rolling 12-month sales for each part.
    """
    now = datetime.now()
    current_month = 2 #now.month - 1 # February is the last completed month (2)

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
    """
    Calculate the rolling 3-month sales for each part in the DataFrame.

    This function computes the sum of sales over the last 3 months for each part.
    The sales columns are dynamically determined based on the current month.

    Args:
        df (pd.DataFrame): DataFrame containing the sales data with columns 
                           named in the format 'sales_{month_abbr}' for current year
                           and 'sales_last_{month_abbr}' for last year.

    Returns:
        pd.DataFrame: DataFrame with an additional column 'rolling_3_month_sales'
                      representing the rolling 3-month sales for each part.
    """
    print('Calculating 3 month rolling sales')
    # Get the current month, adjusted for having only completed data through February
    current_month = 2 #<- set to feb bc data is old 

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

def current_month_sales(df):
    """
    Calculate the current month's sales for a 30-day supply.

    This function retrieves sales data for the current month and calculates the sales
    for a 30-day supply period based on the available data.

    Args:
        df (pd.DataFrame): DataFrame containing the sales data with columns 
                           named in the format 'sales_{month_abbr}' for current year.

    Returns:
        pd.Series: Series containing sales data for the current month.
    """
    print('Calculating current month sales for 30 days supply...')
    current_month = 2 #datetime.now().month <- set to february because data is old
    this_year_sales_columns = [f'sales_{calendar.month_abbr[i].lower()}' for i in range(1, current_month + 1)]
    this_month_sales = df[this_year_sales_columns[-1]]
    return this_month_sales

def calculate_unit_cost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the unit cost for each part.

    This function computes the cost per unit by subtracting the margin from the price
    for each part.

    Args:
        df (pd.DataFrame): DataFrame containing the price and margin data.

    Returns:
        pd.DataFrame: DataFrame with an additional column 'cost_per_unit'.
    """
    df['cost_per_unit'] = np.round(df['price'] - df['margin'], 2)
    return df

def calculate_total_cost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the total cost for each part.

    This function computes the total cost by multiplying the cost per unit by the quantity
    for each part.

    Args:
        df (pd.DataFrame): DataFrame containing the cost per unit and quantity data.

    Returns:
        pd.DataFrame: DataFrame with an additional column 'total_cost'.
    """
    df['total_cost'] = np.round(df['cost_per_unit'] * df['quantity'], 2)
    return df

def calculate_sales_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the sales revenue for each part.

    This function computes the sales revenue by multiplying the rolling 12-month sales
    by the price for each part.

    Args:
        df (pd.DataFrame): DataFrame containing the rolling 12-month sales and price data.

    Returns:
        pd.DataFrame: DataFrame with an additional column 'sales_revenue'.
    """
    df['sales_revenue'] = np.round(df['rolling_12_month_sales'] * df['price'], 2)
    return df

def calculate_cogs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the cost of goods sold (COGS) for each part.

    This function computes the COGS by multiplying the rolling 12-month sales
    by the cost per unit for each part.

    Args:
        df (pd.DataFrame): DataFrame containing the rolling 12-month sales and cost per unit data.

    Returns:
        pd.DataFrame: DataFrame with an additional column 'cogs'.
    """
    df['cogs'] = np.round(df['rolling_12_month_sales'] * df['cost_per_unit'], 2)
    return df

def calculate_margin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the margin and margin percentage for each part.

    This function computes the margin by subtracting the cost per unit from the price,
    and calculates the margin percentage.

    Args:
        df (pd.DataFrame): DataFrame containing the price and cost per unit data.

    Returns:
        pd.DataFrame: DataFrame with additional columns 'margin_percentage' and 'margin'.
    """
    df['margin_percentage'] = np.round(((df['price'] - df['cost_per_unit']) / df['price']) * 100, 2)
    df['margin'] = np.round(df['price'] - df['cost_per_unit'], 2)
    return df

def compute_gross_profit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the gross profit for each part.

    This function calculates the gross profit by multiplying the difference between
    the price and cost per unit by the rolling 12-month sales.

    Args:
        df (pd.DataFrame): DataFrame containing the price, cost per unit, and rolling 12-month sales data.

    Returns:
        pd.DataFrame: DataFrame with an additional column 'gross_profit'.
    """
    df['gross_profit'] = np.round((df['price'] - df['cost_per_unit']) * df['rolling_12_month_sales'], 2)
    return df

def calc_roi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the return on investment (ROI) for each part.

    This function computes the ROI by dividing the difference between gross profit
    and total cost by the total cost, and then converting it to a percentage.

    Args:
        df (pd.DataFrame): DataFrame containing the gross profit and total cost data.

    Returns:
        pd.DataFrame: DataFrame with an additional column 'roi'.
    """
    df['roi'] = np.where(df['total_cost'] != 0,
                         np.round(((df['gross_profit'] - df['total_cost']) / df['total_cost']) * 100, 2), 0) 
    return df

def calculate_day_supply(df):
    """
    Calculate the days' supply of inventory based on average daily sales.

    This function computes the days' supply of inventory for each part over three different periods:
    12 months, 3 months, and 1 month. It calculates the average daily sales for each period and 
    then determines the days' supply by dividing the quantity of each part by the average daily sales.

    Args:
        df (pd.DataFrame): DataFrame containing inventory and sales data with columns for rolling 
                           12-month sales, rolling 3-month sales, and current month sales.

    Returns:
        pd.DataFrame: DataFrame with additional columns for annual, three-month, and one-month days' supply.
    """
    # Calculate average daily sales over 12 months and 3 months
    avg_daily_sales_12mo = df['rolling_12_month_sales'] / 365
    avg_daily_sales_3mo = df['rolling_3_month_sales'] / 90
    avg_daily_sales_1mo = current_month_sales(df) / 30
    
    # Calculate days' supply for 12 months
    df['annual_days_supply'] = df['quantity'] / avg_daily_sales_12mo
    df['annual_days_supply'] = df['annual_days_supply'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df['annual_days_supply'] = np.round(df['annual_days_supply'], 0)
    
    # Calculate days' supply for 3 months
    df['three_month_days_supply'] = df['quantity'] / avg_daily_sales_3mo
    df['three_month_days_supply'] = df['three_month_days_supply'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df['three_month_days_supply'] = np.round(df['three_month_days_supply'], 0)

    # Calculate days' supply for 1 month
    df['one_month_days_supply'] = df['quantity'] / avg_daily_sales_1mo
    df['one_month_days_supply'] = df['one_month_days_supply'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df['one_month_days_supply'] = np.round(df['one_month_days_supply'], 0)

    return df

def calculate_turnover(df):
    """
    Calculate the inventory turnover ratio based on days' supply.

    This function computes the turnover ratio for each part over three different periods:
    12 months, 3 months, and 1 month. It calculates the turnover ratio by dividing the number of days in 
    each period by the corresponding days' supply.

    Args:
        df (pd.DataFrame): DataFrame containing inventory data with columns for annual, three-month,
                           and one-month days' supply.

    Returns:
        pd.DataFrame: DataFrame with additional columns for annual, three-month, and one-month turnover ratios.
    """
    # Calculate annual turnover using 365 days' supply
    if 'annual_days_supply' in df.columns:
        df['annual_turnover'] = 365 / df['annual_days_supply']
        df['annual_turnover'] = df['annual_turnover'].replace([np.inf, -np.inf], np.nan).fillna(0)
        df['annual_turnover'] = np.round(df['annual_turnover'], 1)
    else:
        logging.error("Annual days' supply data is missing")

    # Calculate 90-day turnover using 90 days' supply
    if 'three_month_days_supply' in df.columns:
        df['three_month_turnover'] = 90 / df['three_month_days_supply']
        df['three_month_turnover'] = df['three_month_turnover'].replace([np.inf, -np.inf], np.nan).fillna(0)
        df['three_month_turnover'] = np.round(df['three_month_turnover'], 1)
    else:
        logging.error("Three-month days' supply data is missing")

    # Calculate 30-day turnover using 30 days' supply
    if 'one_month_days_supply' in df.columns:
        df['one_month_turnover'] = 30 / df['one_month_days_supply']
        df['one_month_turnover'] = df['one_month_turnover'].replace([np.inf, -np.inf], np.nan).fillna(0)
        df['one_month_turnover'] = np.round(df['one_month_turnover'], 1)
    else:
        logging.error("One-month days' supply data is missing")

    return df


def sell_through_rate(df):
    """
    Calculate the sell through rate for each part.

    This function computes the sell through rate by dividing the rolling 12-month sales by the quantity * 100
    in stock for each part. If the 'rolling_12_month_sales' column is missing, an error is logged, and the 
    DataFrame is returned unchanged.

    Args:
        df (pd.DataFrame): DataFrame containing inventory and sales data, including columns for 
                           'rolling_12_month_sales' and 'quantity'.

    Returns:
        pd.DataFrame: DataFrame with an additional column for the sales-to-stock ratio.
    """
    if 'rolling_12_month_sales' not in df.columns:
        logging.error("'rolling_12_month_sales' column is missing.")
        return df 
    df["sell_through_rate"] = np.where(df['quantity'] > 0, (df['rolling_12_month_sales'] / df['quantity']) * 100, 0)
    return df

def days_of_inventory_outstanding(df):
    if 'annual_turnover' not in df.columns or 'months_no_sale' not in df.columns:
        logging.error("'annual_turnover' or 'months_no_sale' column is missing.")
        return df
    df['days_of_inventory_outstanding'] = (np.where(df['annual_turnover'] > 0, 365/ df['annual_turnover'], df['months_no_sale'] * 30)).astype(int)
    return df

def order_2_sales(df):
    """
    Calculate the order-to-sales ratio for each part.

    This function computes the order-to-sales ratio by dividing the year-to-date quantity ordered 
    by the rolling 12-month sales for each part. If the 'rolling_12_month_sales' column is missing, 
    an error is logged, and the DataFrame is returned unchanged.

    Args:
        df (pd.DataFrame): DataFrame containing inventory and sales data, including columns for 
                           'rolling_12_month_sales' and 'quantity_ordered_ytd'.

    Returns:
        pd.DataFrame: DataFrame with an additional column for the order-to-sales ratio.
    """
    if 'rolling_12_month_sales' not in df.columns:
        logging.error("'rolling_12_month_sales' column is missing.")
        return df 
    df["order_to_sales_ratio"] = np.where(df['rolling_12_month_sales'] > 0, df['quantity_ordered_ytd'] / df['rolling_12_month_sales'], 0)
    return df


def prepare_and_melt_sales_data(df):
    """
    Prepare and melt sales data from wide to long format.

    This function processes a DataFrame containing monthly sales data for two consecutive years. It separates the data for
    the current year and the previous year, melts them into a long format, and then combines the results into a single
    DataFrame. The function also removes the original monthly sales columns from the input DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing sales data for the current and previous years. The DataFrame should
                           include columns named 'sales_<month_abbr>' for the current year's sales and 'sales_last_<month_abbr>'
                           for the previous year's sales.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The original DataFrame with the monthly sales columns removed.
            - pd.DataFrame: The melted DataFrame with columns for 'part_number', 'month', 'quantity_sold', and 'year'.
    """
    # Make a copy of the original DataFrame
    df_copy = df.copy()

    # Define the current year and last year for labeling purposes
    current_year = datetime.now().year
    last_year = current_year - 1
    current_month_index = 2 #datetime.now().month <- set to two bc data is from february

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
    print('calculating seasonality...')
    """
    Extract the seasonal component from sales data using Fast Fourier Transform (FFT).

    This function identifies the dominant seasonal components in the sales data by applying FFT. It converts month names
    to numerical representations if necessary, sorts the data by the time column, and then computes the FFT. The top
    frequencies are identified, and their corresponding amplitudes and phases are used to calculate the seasonal component.

    Args:
        sales_data (pd.DataFrame): DataFrame containing sales data with time and value columns.
        time_column (str): Name of the column representing time (months).
        value_column (str): Name of the column representing sales values.
        top_n_freq (int): Number of top frequencies to consider for the seasonal component.

    Returns:
        float: The mean of the seasonal component calculated from the dominant frequencies.
    """
    # Efficiently convert month names to numbers if necessary
    if sales_data[time_column].dtype == object:
        month_to_num = {month: index for index, month in enumerate(calendar.month_name[1:], start=1)}
        sales_data[time_column] = sales_data[time_column].apply(lambda x: month_to_num[x] if x in month_to_num else x)

    # Sort in-place
    sales_data.sort_values(by=time_column, inplace=True)
    y = sales_data[value_column].values

    # Using PyFFTW for FFT computation, assuming it's properly configured for efficiency
    y_fft = pyfftw.empty_aligned(len(y), dtype='complex128')
    y_fft[:] = y
    fft_object = pyfftw.builders.rfft(y_fft)
    fft_values = fft_object()

    # Process FFT results to find seasonal components
    fft_freq = np.fft.rfftfreq(len(y))
    top_indices = np.argsort(-np.abs(fft_values))[:top_n_freq]

    # Calculate amplitudes and phases of dominant frequencies
    amplitudes = np.abs(fft_values[top_indices])
    phases = np.angle(fft_values[top_indices])

    # Compute seasonal component mean directly
    time_points = np.arange(len(y))
    frequencies = fft_freq[top_indices]
    seasonal_component = amplitudes * np.cos(2 * np.pi * frequencies[:, np.newaxis] * time_points + phases[:, np.newaxis])
    seasonal_component_mean = np.mean(np.sum(seasonal_component, axis=0))

    return seasonal_component_mean

def calculate_additional_metrics(df) -> pd.DataFrame:
    print('calculating additional metrics')
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
    df = sell_through_rate(df)
    df = days_of_inventory_outstanding(df)
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
        data = json.loads(input_data)
        parts_data = pd.DataFrame(data['data'], columns=data['columns'])
        parts_data_final = calculate_additional_metrics(parts_data)
        logging.info(f"Created Columns: {parts_data_final.columns}.")

    
        parts_data, df_melted_sales = prepare_and_melt_sales_data(parts_data)

        seasonal_components = df_melted_sales.groupby('part_number').apply(
        lambda group: extract_seasonal_component(group, 'month', 'quantity_sold')
        ).reset_index(name='seasonal_component')

        if 'seasonal_component' in parts_data:
            return parts_data
        else:
            # Merge seasonal components back to the original DataFrame
            parts_data_with_seasonal = parts_data_final.merge(seasonal_components, on='part_number', how='left')
            if 'sell_through_rate' not in parts_data_with_seasonal:
                logging.error("Failed to add new columns properly.")
        # Save the DataFrame with seasonal components to output_data
        parts_data_with_seasonal = parts_data_with_seasonal.to_json(orient='split')
        # Return success at the end
        print("columns created successfully...")
        return parts_data_with_seasonal
        
    except ValueError as e:
        logging.error(f"Invalid JSON format: {e}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: Invalid file format.'})
        return False
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: {str(e)}'})
        return False