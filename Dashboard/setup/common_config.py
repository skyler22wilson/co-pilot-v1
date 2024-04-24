import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta

def load_config():
    config_path = 'Dashboard/configuration/partswise_config.json'  
    with open(config_path, 'r') as file:
        config_data = json.load(file)
    return config_data

def get_db_connection(db_file_path):
    conn = sqlite3.connect(db_file_path)
    return conn

def format_currency(value):
    return "${:,.2f}".format(value)

def format_perc(value):
    return "{:,.3f}%".format(value)

def calculate_combined_metrics(conn):
    # Combined query for sales revenue, cogs, and inventory metrics
    combined_query = f"""
    SELECT 
        SUM(p.sales_revenue) AS sales_revenue,
        SUM(p.cost_per_unit * s.quantity_sold) as cogs
        AVG(p._12_month_turnover) as turnover
    FROM parts
    JOIN sales s ON s.part_number = p.part_number;
    """
    result = conn.execute(combined_query).fetchone()
    sales_revenue, cogs, turnover = result

    return {
        "sales_revenue": sales_revenue,
        "cogs": cogs,
        "turnover": turnover
    }

def calculate_percentage_change(current_year_sum, last_year_sum, decrease_is_positive=False):
    # Check if last year's value is zero to avoid division by zero
    if last_year_sum == 0:
        return "N/A", "black"

    # Calculate the percentage change
    percentage_change = ((current_year_sum - last_year_sum) / last_year_sum) * 100

    # Determine the color based on whether the change is positive or negative
    color = "green" if percentage_change >= 0 else "red"
    if decrease_is_positive:
        color = "red" if percentage_change > 0 else "green"
    
    # Format the percentage change as a string
    percentage_change_str = "{:+.2f}%".format(percentage_change)
    return percentage_change_str, color

def turn_result(perc_change, industry_standard):
    if perc_change > industry_standard:
        comparison_text = "Above Benchmark"
        
    elif perc_change < industry_standard:
        comparison_text = "Below Benchmark"
    else:
        comparison_text = "At Benchmark"
    return comparison_text


def get_latest_completed_month():
    today = datetime.today()
    first_day_of_current_month = datetime(today.year, today.month, 1)
    last_day_of_previous_month = first_day_of_current_month - timedelta(days=1)
    return last_day_of_previous_month.month, last_day_of_previous_month.year

def get_current_month():
    # Get the current date
    today = datetime.now()
    # Return the current month name
    return today.strftime("%B")

def get_current_year_and_month():
    today = datetime.today()
    current_month = today.month
    current_year = today.year
    return current_year, current_month

def create_temp_month_index_table(conn):
    # Create the temporary table
    create_temp_table_query = """
    CREATE TEMPORARY TABLE IF NOT EXISTS MonthIndex (
        month_name TEXT,
        month_index INTEGER
    );
    """
    conn.execute(create_temp_table_query)
    
    # Delete any existing data in the temporary table
    conn.execute("DELETE FROM MonthIndex;")
    
    # Insert data into the temporary table
    insert_data_queries = [
        "INSERT INTO MonthIndex (month_name, month_index) VALUES ('January', 1);",
        "INSERT INTO MonthIndex (month_name, month_index) VALUES ('February', 2);",
        "INSERT INTO MonthIndex (month_name, month_index) VALUES ('March', 3);",
        "INSERT INTO MonthIndex (month_name, month_index) VALUES ('April', 4);",
        "INSERT INTO MonthIndex (month_name, month_index) VALUES ('May', 5);",
        "INSERT INTO MonthIndex (month_name, month_index) VALUES ('June', 6);",
        "INSERT INTO MonthIndex (month_name, month_index) VALUES ('July', 7);",
        "INSERT INTO MonthIndex (month_name, month_index) VALUES ('August', 8);",
        "INSERT INTO MonthIndex (month_name, month_index) VALUES ('September', 9);",
        "INSERT INTO MonthIndex (month_name, month_index) VALUES ('October', 10);",
        "INSERT INTO MonthIndex (month_name, month_index) VALUES ('November', 11);",
        "INSERT INTO MonthIndex (month_name, month_index) VALUES ('December', 12);"
    ]

    for query in insert_data_queries:
        conn.execute(query)
    
    conn.commit()

def calculate_financial_metrics(conn):
    current_year, current_month = get_current_year_and_month()
    
    # Corrected and simplified SQL query
    query = f"""
    SELECT 
        SUM(p.cost) AS total_cost,
        AVG(p._12_month_turnover) AS average_turnover,
        (
            SELECT SUM(p.price * s.quantity_sold)
            FROM sales s
            JOIN parts p ON s.part_number = p.part_number
            JOIN MonthIndex mi ON mi.month_name = s.month
            WHERE mi.month_index <= {current_month} AND s.year = {current_year}
        ) AS ytd_sales_revenue,
        (
            SELECT SUM((p.price - p.cost_per_unit) * s.quantity_sold)
            FROM sales s
            JOIN parts p ON s.part_number = p.part_number
            JOIN MonthIndex mi ON mi.month_name = s.month
            WHERE mi.month_index <= {current_month} AND s.year = {current_year}
        ) AS ytd_gross_profit
    FROM parts p
    """
    result = conn.execute(query).fetchone()
    if result:
        total_cost, average_turnover, ytd_sales_revenue, ytd_gross_profit = result

        # Calculate the gross margin percentage if sales revenue is not zero
        gross_margin_percentage = (ytd_gross_profit / ytd_sales_revenue * 100) if ytd_sales_revenue else 0

        return {
            "ytd_sales_revenue": ytd_sales_revenue,
            "total_cost": total_cost,
            "ytd_gross_profit": ytd_gross_profit,
            "average_turnover": average_turnover,
            "ytd_gross_margin_percentage": gross_margin_percentage
        }
    else:
        print("No data found.")
        return {
            "ytd_sales_revenue": 0,
            "total_cost": 0,
            "ytd_gross_profit": 0,
            "average_turnover": 0,
            "ytd_gross_margin_percentage": 0
        }
    
def calculate_mtd_financial_metrics(conn):
    latest_month, current_year = get_latest_completed_month()
    
    # Corrected and simplified SQL query
    query = f"""
    SELECT 
        (
            SELECT SUM(p.price * s.quantity_sold)
            FROM sales s
            JOIN parts p ON s.part_number = p.part_number
            JOIN MonthIndex mi ON mi.month_name = s.month
            WHERE mi.month_index == {latest_month} AND s.year = {current_year}
        ) AS ytd_sales_revenue,
        (
            SELECT SUM((p.price - p.cost_per_unit) * s.quantity_sold)
            FROM sales s
            JOIN parts p ON s.part_number = p.part_number
            JOIN MonthIndex mi ON mi.month_name = s.month
            WHERE mi.month_index == {latest_month} AND s.year = {current_year}
        ) AS ytd_gross_profit
    FROM parts p
    """
    result = conn.execute(query).fetchone()
    if result:
        mtd_sales_revenue, mtd_gross_profit = result

        # Calculate the gross margin percentage if sales revenue is not zero
        mtd_gross_margin_percentage = (mtd_gross_profit / mtd_sales_revenue * 100) if mtd_sales_revenue else 0

        return {
            "mtd_sales_revenue": mtd_sales_revenue,
            "mtd_gross_profit": mtd_gross_profit,
            "mtd_gross_margin_percentage": mtd_gross_margin_percentage
        }
    else:
        print("No data found.")
        return {
            "mtd_sales_revenue": 0,
            "mtd_gross_profit": 0,
            "mtd_gross_margin_percentage": 0
        }

def calculate_last_year_metrics(conn):
    # Assuming get_latest_completed_month() returns the latest full month and current year
    latest_month, current_year = get_latest_completed_month()
    previous_year = current_year - 1

    # Query to fetch sales revenue and COGS for the previous year up to the current month index
    query = f"""
    SELECT 
        (
            SELECT SUM(p.price * s.quantity_sold)
            FROM sales s
            JOIN parts p ON s.part_number = p.part_number
            JOIN MonthIndex mi ON mi.month_name = s.month
            WHERE mi.month_index <= {latest_month} AND s.year = {previous_year}
        )   AS ytd_sales_revenue_last_year,
        (
            SELECT SUM((p.price - p.cost_per_unit) * s.quantity_sold)
            FROM sales s
            JOIN parts p ON s.part_number = p.part_number
            JOIN MonthIndex mi ON mi.month_name = s.month
            WHERE mi.month_index <= {latest_month} AND s.year = {previous_year}
        )  AS ytd_gross_profit_last_year
    """
    result = conn.execute(query).fetchone()
    if result:
        sales_revenue_last_year, gross_profit_last_year = result

        # Calculate gross margin percentage for the previous year YTD
        gross_margin_last_year_percentage = (gross_profit_last_year / sales_revenue_last_year) * 100 if sales_revenue_last_year > 0 else 0
        return {
            "ytd_sales_revenue_last_year": sales_revenue_last_year,
            "ytd_gross_profit_last_year": gross_profit_last_year,
            "ytd_gross_margin_last_year_percentage": gross_margin_last_year_percentage
        }
    else:
        return {
            "ytd_sales_revenue_last_year": 0,
            "ytd_gross_profit_last_year": 0,
            "gross_margin_last_year_percentage": 0
        }
    
def calculate_mtd_last_year_metrics(conn):
    # Assuming get_latest_completed_month() returns the latest full month and current year
    latest_month, current_year = get_latest_completed_month()
    previous_year = current_year - 1

    # Query to fetch sales revenue and COGS for the previous year up to the current month index
    query = f"""
    SELECT 
        (
            SELECT SUM(p.price * s.quantity_sold)
            FROM sales s
            JOIN parts p ON s.part_number = p.part_number
            JOIN MonthIndex mi ON mi.month_name = s.month
            WHERE mi.month_index == {latest_month} AND s.year = {previous_year}
        )   AS ytd_sales_revenue_last_year,
        (
            SELECT SUM((p.price - p.cost_per_unit) * s.quantity_sold)
            FROM sales s
            JOIN parts p ON s.part_number = p.part_number
            JOIN MonthIndex mi ON mi.month_name = s.month
            WHERE mi.month_index == {latest_month} AND s.year = {previous_year}
        )  AS ytd_gross_profit_last_year
    """
    result = conn.execute(query).fetchone()
    if result:
        mtd_sales_revenue_last_year, mtd_gross_profit_last_year = result

        # Calculate gross margin percentage for the previous year YTD
        mtd_gross_margin_last_year_percentage = (mtd_gross_profit_last_year / mtd_sales_revenue_last_year) * 100 if mtd_sales_revenue_last_year > 0 else 0
        return {
            "mtd_sales_revenue_last_year": mtd_sales_revenue_last_year,
            "mtd_gross_profit_last_year": mtd_gross_profit_last_year,
            "mtd_gross_margin_last_year_percentage": mtd_gross_margin_last_year_percentage
        }
    else:
        return {
            "mtd_sales_revenue_last_year": 0,
            "mtd_gross_profit_last_year": 0,
            "mtd_gross_margin_last_year_percentage": 0
        }

def calculate_ytd_metrics(conn):
    # Calculate current year metrics
    create_temp_month_index_table(conn)
    current_year_metrics = calculate_financial_metrics(conn)
    last_year_metrics = calculate_last_year_metrics(conn)
    
    # Format YTD results
    ytd_results = {
        'Year-to-Date Sales Revenue': current_year_metrics['ytd_sales_revenue'],
        'Year-to-Date Gross Profit': current_year_metrics['ytd_gross_profit'],
        'Year-to-Date Gross Margin': np.round(current_year_metrics['ytd_gross_margin_percentage'], 3),
        'Year-to-Date Sales Revenue Last Year': last_year_metrics['ytd_sales_revenue_last_year'],
        'Year-to-Date Gross Margin Last Year': np.round(last_year_metrics['ytd_gross_margin_last_year_percentage'], 3),
        'Year-to-Date Gross Profit Last Year': last_year_metrics['ytd_gross_profit_last_year']
    }
    return ytd_results

def calculate_mtd_metrics(conn):
    # Calculate current MTD metrics
    create_temp_month_index_table(conn)
    current_mtd_metrics = calculate_mtd_financial_metrics(conn)
    last_year_mtd_metrics = calculate_mtd_last_year_metrics(conn)
    
    # Format MTD results
    mtd_results = {
        'Month-to-Date Sales Revenue': current_mtd_metrics['mtd_sales_revenue'],
        'Month-to-Date Gross Profit': current_mtd_metrics['mtd_gross_profit'],
        'Month-to-Date Gross Margin': np.round(current_mtd_metrics['mtd_gross_margin_percentage'], 3),
        'Month-to-Date Sales Revenue Last Year': last_year_mtd_metrics['mtd_sales_revenue_last_year'],
        'Month-to-Date Gross Margin Last Year': np.round(last_year_mtd_metrics['mtd_gross_margin_last_year_percentage'], 3),
        'Month-to-Date Gross Profit Last Year': last_year_mtd_metrics['mtd_gross_profit_last_year']
    }
    return mtd_results

def calculate_totals_metrics(conn):
    create_temp_month_index_table(conn)
    # Assume calculate_financial_metrics function can also provide total cost and turnover
    current_year_metrics = calculate_financial_metrics(conn)
    
    # Format Totals results
    totals_results = {
        'Total Cost': current_year_metrics['total_cost'],
        '12 Month Turnover': np.round(current_year_metrics['average_turnover'], 3),
    }
    return totals_results
