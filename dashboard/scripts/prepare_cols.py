import polars as pl
from io import StringIO
import logging
import numpy as np
import os
import calendar
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from dashboard.setup.utils import load_configuration, create_long_form_dataframe

CONFIG_FILE = "dashboard/configuration/SeasonalConfig.json"
LOGGING_DIR = "Logs"


def calculate_unit_cost(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the unit cost for each part.

    This function computes the cost per unit by subtracting the margin from the price
    for each part.

    Args:
        df (pl.DataFrame): DataFrame containing the price and margin data.

    Returns:
        pl.DataFrame: DataFrame with an additional column 'cost_per_unit'.
    """
    # Compute the cost per unit
    df = df.with_columns(
        (pl.col('price') - pl.col('margin')).round(2).alias('cost_per_unit')
    )

    return df

def calculate_total_cost(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the total cost for each part.

    This function computes the total cost by multiplying the cost per unit by the quantity
    for each part.

    Args:
        df (pl.DataFrame): DataFrame containing the cost per unit and quantity data.

    Returns:
        pl.DataFrame: DataFrame with an additional column 'total_cost'.
    """
    df = df.with_columns(
            (pl.col('cost_per_unit') * pl.col('quantity')).round(2).alias('total_cost')
        )

    return df


def calculate_margin(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the margin and margin percentage for each part.

    This function computes the margin by subtracting the cost per unit from the price,
    and calculates the margin percentage.

    Args:
        df (pl.DataFrame): DataFrame containing the price and cost per unit data.

    Returns:
        pl.DataFrame: DataFrame with additional column 'margin_percentage'.
    """
    df = df.with_columns(
            (((pl.col('price') - pl.col('cost_per_unit')) / pl.col('price')) * 100).round(2).alias('margin_percentage')
            )

    return df

def calculate_annual_financial_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate annual financial metrics including total sales year-to-date, sales revenue, gross profit, COGS, and ROI.

    This function retrieves sales data from each month of the current year up to the current month,
    calculates the total sales year-to-date, and then uses this data to compute sales revenue,
    gross profit, COGS, and ROI.

    Args:
        df (pl.DataFrame): DataFrame containing the sales data with columns named in the format 
                           'sales_{month_abbr}', and 'price' and 'cost_per_unit' for the current year.

    Returns:
        pl.DataFrame: DataFrame with additional columns 'total_sales_ytd', 'sales_revenue', 'gross_profit', 
                      'cogs', and 'roi', representing the total sales year-to-date, calculated sales revenue, 
                      gross profit, cost of goods sold, and return on investment.
    """
    current_month =  datetime.now().month
    this_year_sales_columns = [f'sales_{calendar.month_abbr[i].lower()}' for i in range(1, current_month + 1)]


    current_month =  datetime.now().month
    this_year_sales_columns = [f'sales_{calendar.month_abbr[i].lower()}' for i in range(1, current_month + 1)]

    df = df.with_columns(
            (pl.col('price') - pl.col('margin')).round(2).alias('cost_per_unit')
        )
    total_sales_ytd = pl.sum_horizontal(this_year_sales_columns).alias('total_sales_ytd')

    df = df.with_columns(total_sales_ytd)

    df = df.with_columns([
        (pl.col('total_sales_ytd') * pl.col('price')).round(2).alias('sales_revenue'),
        (pl.col('total_sales_ytd') * pl.col('cost_per_unit')).round(2).alias('cogs')
    ])

    # Calculate 'gross_profit'
    df = df.with_columns([
        (pl.col('sales_revenue') - pl.col('cogs')).round(2).alias('gross_profit')
    ])

    # Calculate 'roi'
    df = df.with_columns([
        pl.when(pl.col('cogs') > 0)
        .then((pl.col('gross_profit') / pl.col('cogs')) * 100)
        .otherwise(0.00)
        .round(2)
        .alias('roi')
    ])
    return df

def current_month_sales(df: pl.DataFrame) -> pl.Expr:
    """
    Calculate the current month's sales for a 30-day supply.

    This function retrieves sales data for the current month and calculates the sales
    for a 30-day supply period based on the available data.

    Args:
        df (pl.DataFrame): DataFrame containing the sales data with columns 
                           named in the format 'sales_{month_abbr}' for the current year.

    Returns:
        pl.Expr: Expression containing sales data for the current month.
    """
    current_month = datetime.now().month - 1
    month_abbr = calendar.month_abbr[current_month].lower()
    this_month_sales_column = f'sales_{month_abbr}'

    this_month_sales_sum = df.select([
            pl.col(this_month_sales_column).sum().alias('total_sales_current_month')
        ])

    return this_month_sales_sum.item()

def calculate_rolling_sales(df_long, df_original):
    """
    Calculate rolling sales, aggregate them, and join the results back to the original DataFrame.

    Args:
        df_long (pl.DataFrame): DataFrame in long format with 'part_number', 'month', and 'quantity_sold'.
        df_original (pl.DataFrame): Original DataFrame before conversion to long format.

    Returns:
        pl.DataFrame: Original DataFrame enriched with aggregated rolling sales data.
    """
    # Calculate rolling sales
    df_rolling = df_long.with_columns([
        pl.col("quantity_sold").rolling_sum(window_size=3, min_periods=1).over("part_number").alias("rolling_3m_sales"),
        pl.col("quantity_sold").rolling_sum(window_size=12, min_periods=1).over("part_number").alias("rolling_12m_sales")
    ])

    # Aggregate rolling sales over all months for each part number
    df_aggregated = df_rolling.group_by("part_number").agg([
        pl.sum("rolling_3m_sales").alias("rolling_3m_sales"),
        pl.sum("rolling_12m_sales").alias("rolling_12m_sales")
    ])

    # Join the aggregated data back to the original DataFrame
    df_enriched = df_original.join(df_aggregated, on="part_number", how="left")

    return df_enriched


def calculate_day_supply(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the days' supply of inventory based on average daily sales.
    """
    # Calculate average daily sales over 12 months, 3 months, and the current month sales obtained
    avg_daily_sales_12mo = pl.col('rolling_12m_sales') / 365
    avg_daily_sales_3mo = pl.col('rolling_3m_sales') / 90
    avg_daily_sales_1mo = current_month_sales(df) / 30  

    # Define helper to calculate days' supply given a period's average daily sales
    def calculate_days_supply(quantity_col, avg_daily_sales, label):
        days_supply_expr = (
            pl.when((quantity_col / avg_daily_sales).is_infinite() | 
                (quantity_col / avg_daily_sales).is_null() | 
                (quantity_col / avg_daily_sales).is_nan()) 
            .then(0)
            .otherwise(quantity_col / avg_daily_sales)
            .cast(pl.Int64)  # Corrected line
            .alias(f'{label}_days_supply')
        )
        return days_supply_expr

    # Update DataFrame with days' supply calculations
    df = df.with_columns([
        calculate_days_supply(pl.col('quantity'), avg_daily_sales_12mo, '12m'),
        calculate_days_supply(pl.col('quantity'), avg_daily_sales_3mo, '3m'),
        calculate_days_supply(pl.col('quantity'), avg_daily_sales_1mo, '1m')
    ])

    return df

def calculate_turnover(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the inventory turnover ratio based on proportional estimation of full-year orders.

    Args:
        df (pl.DataFrame): DataFrame containing inventory data with columns 'quantity_ordered_ytd',
                           'quantity', 'total_sales_ytd', and 'cogs'.

    Returns:
        pl.DataFrame: DataFrame with an additional column for inventory turnover.
    """
    months_covered = datetime.now().month

    # Estimate full year orders based on YTD orders
    estimated_fy_orders = (pl.col('quantity_ordered_ytd') / months_covered) * 12 
    

    # Estimating beginning inventory (assuming you have some way to calculate or estimate this)
    starting_inventory = pl.col('quantity') + estimated_fy_orders - pl.col('rolling_12m_sales')
    ending_inventory = pl.col('quantity')
    average_inventory = (starting_inventory + ending_inventory) / 2
    turnover = pl.col('cogs') / average_inventory

    return df.with_columns(
        turnover.fill_nan(0).round(2).alias('turnover')
    )

def calculate_3mo_turnover(df):
    months_covered = datetime.now().month
    estimated_3mo_orders = (pl.col('quantity_ordered_ytd') / months_covered) * 3
    
    # Estimating beginning inventory (assuming you have some way to calculate or estimate this)
    starting_inventory = pl.col('quantity') + estimated_3mo_orders - pl.col('rolling_3m_sales')
    ending_inventory = pl.col('quantity')
    average_inventory = (starting_inventory + ending_inventory) / 2
    turnover = pl.col('cogs') / average_inventory

    return df.with_columns(
        turnover.fill_nan(0).round(2).alias('3m_turnover')
    )

def sell_through_rate(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the sell through rate for each part.

    This function computes the sell through rate by dividing the rolling 12-month sales by the quantity * 100
    in stock for each part. If the 'rolling_12_month_sales' column is missing, an error is logged, and the 
    DataFrame is returned unchanged.

    Args:
        df (pl.DataFrame): DataFrame containing inventory and sales data, including columns for 
                           'rolling_12_month_sales' and 'quantity'.

    Returns:
        pl.DataFrame: DataFrame with an additional column for the sales-to-stock ratio.
    """
    if 'rolling_12m_sales' not in df.columns:
        logging.error("'rolling_12m_sales' column is missing.")
        return df 
    

    sell_through_rate = (pl.when(
        pl.col('quantity') > 0)
        .then(pl.col('rolling_12m_sales') / pl.col('quantity') * 100)
        .otherwise(0)
        .round(2)
        .alias('sell_through_rate')
    )

    return df.with_columns(sell_through_rate)

def days_of_inventory_outstanding(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate days of inventory outstanding based on turnover and months of no sale.

    Args:
        df (pl.DataFrame): DataFrame containing the columns 'annual_turnover' and 'months_no_sale'.

    Returns:
        pl.DataFrame: DataFrame with an additional column 'days_of_inventory_outstanding'.
    """
    if 'turnover' not in df.columns or 'months_no_sale' not in df.columns:
        logging.error("'turnover' or 'months_no_sale' column is missing.")
        return df

    days_of_inventory_outstanding = (
        pl.when(pl.col('turnover') > 0)
        .then(365 / pl.col('turnover'))
        .otherwise(pl.col('months_no_sale') * 30)
        .cast(pl.Int64)
        .alias('days_of_inventory_outstanding')
    )

    return df.with_columns(days_of_inventory_outstanding)

def order_2_sales(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the order-to-sales ratio for each part.

    This function computes the order-to-sales ratio by dividing the year-to-date quantity ordered, 
    annualized, by the rolling 12-month sales for each part. If necessary columns are missing, an error is logged, 
    and the DataFrame is returned unchanged.

    Args:
        df (pl.DataFrame): DataFrame containing inventory and sales data, including columns for 
                           'rolling_12_month_sales' and 'quantity_ordered_ytd'.

    Returns:
        pl.DataFrame: DataFrame with an additional column for the order-to-sales ratio.
    """
    if 'rolling_12m_sales' not in df.columns or 'quantity_ordered_ytd' not in df.columns:
        logging.error("Necessary columns ('rolling_12m_sales' or 'quantity_ordered_ytd') are missing.")
        return df
    
    months_covered = datetime.now().month
    estimated_fy_orders = (pl.col('quantity_ordered_ytd') / months_covered) * 12 

    order_2_sales = (
        pl.when(pl.col('rolling_12m_sales') > 0)
        .then(estimated_fy_orders / pl.col('rolling_12m_sales'))
        .otherwise(0)
        .alias('order_to_sales_ratio')
    )

    return df.with_columns(order_2_sales)

def create_seasonal_component(df):
    """
    Create a datetime column from year and month, and add cyclical features based on the datetime.

    Args:
        df (pl.DataFrame): DataFrame containing year and month columns.
        year_col (str): Column name with the year.
        month_col (str): Column name with the month name.

    Returns:
        pl.DataFrame: Enhanced DataFrame with datetime and cyclical features.
    """

    # Map month names to numbers (if needed)
    logging.debug(f"Shape before adding cyclical features: {df.shape}")
    month_to_number = {month.lower(): i for i, month in enumerate(calendar.month_abbr) if month}

    # Create a datetime column assuming the first day of each month
    month_number_expr = pl.when(pl.col("month").str.to_lowercase() == "jan").then(1)
    for month, num in month_to_number.items():
        month_number_expr = month_number_expr.when(pl.col("month").str.to_lowercase() == month).then(num)

    # Apply the case expression
    df = df.with_columns(month_number_expr.alias("month_number"))

    month_numbers = df.get_column("month_number").to_numpy()  
    month_sin = np.sin(2 * np.pi * month_numbers / 12)
    month_cos = np.cos(2 * np.pi * month_numbers / 12)

    # Add calculated columns back to DataFrame
    df = df.with_columns([
        pl.Series("month_sin", month_sin),
        pl.Series("month_cos", month_cos)
    ])

    logging.debug(f"Shape after adding cyclical features: {df.shape}")
    logging.debug(f"DataFrame with cyclical features: {df.head()}")
    print(f'Seasonality Columns: {df.columns}')
    
    return df

def create_seasonal_df(df, window_size=12):
    """
    Enhance a DataFrame with a rolling average and combine it with cyclical features to create a seasonal component.

    Args:
        df (pl.DataFrame): DataFrame with columns for part number, date, quantity sold, and cyclical features.
        window_size (int): Size of the rolling window for calculating the moving average.

    Returns:
        pl.DataFrame: DataFrame with added rolling average and seasonal adjustment features.
    """
    required_columns = ["quantity_sold", "month_sin", "month_cos"]
    for col in required_columns:
        if col not in df.columns:
            logging.error(f"Missing column in DataFrame: {col}")
            raise ValueError(f"Missing column in DataFrame: {col}")
    # Calculate the rolling mean for the quantity sold
    df = df.with_columns(
        pl.col("quantity_sold").rolling_mean(window_size=window_size, min_periods=1).alias("rolling_avg")
    )

    # Calculate deviation from the rolling average
    df = df.with_columns(
        (pl.col("quantity_sold") - pl.col("rolling_avg")).alias("deviation")
    )

    # Combine cyclical features with deviations to create a seasonal score
    df = df.with_columns(
        (pl.col("deviation") * pl.col("month_sin") + pl.col("deviation") * pl.col("month_cos")).alias("seasonal_score")
    )

    # You could also normalize this score by the number of observations (i.e., months) for each part number
    df = df.group_by("part_number").agg([
        pl.mean("seasonal_score").alias("normalized_seasonal_score")
    ])

    return df

def calculate_additional_metrics(df) -> pl.DataFrame:
    logging.info('Starting to calculate additional metrics...')
    
    try:
        logging.info('Calculating unit cost...')
        df = calculate_unit_cost(df)
        assert 'cost_per_unit' in df.columns, "cost_per_unit column not created"

        logging.info('Calculating total cost...')
        df = calculate_total_cost(df)
        assert 'total_cost' in df.columns, "total_cost column not created"

        logging.info('Calculating margin...')
        df = calculate_margin(df)
        assert 'margin_percentage' in df.columns, "margin_percentage column not created"

        logging.info('Calculating annual financial metrics...')
        df = calculate_annual_financial_metrics(df)
        assert 'gross_profit' in df.columns, "gross_profit column not created"

        logging.info('Creating long form dataframe for rolling sales...')
        df_long = create_long_form_dataframe(df)
        df_long = create_seasonal_component(df_long)
        df_seasonal = create_seasonal_df(df_long)
        logging.info('Joining seasonal data with long form...')
        df = df_seasonal.join(df, on="part_number", how="left")

        logging.info('Calculating rolling sales...')
        df = calculate_rolling_sales(df_long, df)

        logging.info('Calculating day supply...')
        df = calculate_day_supply(df)

        logging.info('Calculating turnover...')
        df = calculate_turnover(df)

        logging.info('Calculating 3-month turnover...')
        df = calculate_3mo_turnover(df)

        logging.info('Calculating sell through rate...')
        df = sell_through_rate(df)

        logging.info('Calculating days of inventory outstanding...')
        df = days_of_inventory_outstanding(df)

        logging.info('Calculating order to sales ratio...')
        df = order_2_sales(df)

        logging.info('All calculations complete.')
        return df

    except AssertionError as e:
        logging.error(f"Assertion Error: {e}")
        raise
    except Exception as e:
        logging.error(f"Error during metric calculation: {str(e)}")
        raise

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
        df = pl.read_json(StringIO(input_data))
        print('Data loaded successfully!')
        parts_data = calculate_additional_metrics(df)
        logging.info(f"Created Columns: {parts_data.columns}.")

        parts_data.write_csv('/Users/skylerwilson/Desktop/PartsWise/Data/Processed/parts_data_prepared.csv')
        
        parts_data_json = parts_data.write_json()
    
        return parts_data_json
        
    except ValueError as e:
        logging.error(f"Invalid JSON format: {e}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: Invalid file format.'})
        return False
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: {str(e)}'})
        return False