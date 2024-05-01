from celery import Celery
from celery import chord, group, chain
from Dashboard.scripts.data_clean import main as data_clean
from Dashboard.scripts.prepare_cols import main as column_preperation
from Dashboard.Models.demand_predictor.predictions import main as demand_ml_funct
from Dashboard.scripts.feature_importances import main as calculate_features
from Dashboard.scripts.demand_calc import main as create_demand_score
from Dashboard.scripts.create_categories import main as create_categories
from Dashboard.scripts.check_reorder import main as calculate_reorder
from Dashboard.Models.obsolecence_predictor.obsolete_predictor import main as obsolescence_risk
from Dashboard.scripts.monthly_sales import main as monthly_sales
from Dashboard.scripts.db_setup import main as build_database
from functools import reduce
import json
import pandas as pd
import logging

# Use Redis as the message broker
app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')


@app.task(bind=True)
def run_data_cleaning_task(self, input_data):
    """
    A Celery task that wraps the data cleaning and processing logic.
    This task is designed to run the main function and handle its state updates and exceptions.
    """

    # Pass the current task instance to your main function for state updates
    return data_clean(self, input_data)

@app.task(bind=True)
def prepare_data_task(self, input_data):
    """
    A Celery task that wraps the column preperation and processing logic.
    This task is designed to run the main function and handle its state updates and exceptions.
    """
    return column_preperation(self, input_data)

@app.task(bind=True)
def run_demand_predictor_task(self, input_data):
    """
    A Celery task that wraps the demand machine learning model logic.
    This task is designed to run the main function and handle its state updates and exceptions.
    """
    return demand_ml_funct(self, input_data)

@app.task(bind=True)
def feature_importances_task(self, input_data):
    return calculate_features(self, input_data)

@app.task(bind=True)
def create_demand_score_task(self, input_data):
    return create_demand_score(self, input_data)

@app.task(bind=True)
def categorization_task(self, input_data):
    return create_categories(self, input_data)

@app.task(bind=True)
def reordering_task(self, input_data):
    return calculate_reorder(self, input_data)

@app.task(bind=True)
def obsolescence_risk_task(self, input_data):
    return obsolescence_risk(self, input_data)

@app.task(bind=True)
def aggregate_results(self, data_from_previous_tasks):
    # Convert JSON strings back to DataFrames
    dataframes = [pd.DataFrame(json.loads(data)['data'], columns=json.loads(data)['columns']) for data in data_from_previous_tasks]

    # Reduce function to merge DataFrames without duplicating columns
    def merge_dfs(left, right):
        # Identify overlapping columns, excluding the merging key 'part_number'
        overlapping_columns = left.columns.intersection(right.columns).drop('part_number')
        
        # Drop overlapping columns from the right DataFrame
        right_adjusted = right.drop(columns=overlapping_columns)
        
        # Merge the adjusted right DataFrame with the left DataFrame
        return pd.merge(left, right_adjusted, on='part_number', how='outer')

    # Use functools.reduce with the custom merge function
    aggregated_data = reduce(merge_dfs, dataframes)

    # Convert the aggregated DataFrame back to JSON string if passing to another task
    aggregated_json = aggregated_data.to_json(orient='split')

    return aggregated_json

@app.task(bind=True)
def monthly_sales_task(self, input_data):
    result = monthly_sales(self, input_data)
    return result

@app.task(bind=True)
def database_builder(self, input_data):
    return build_database(self, input_data)


@app.task(bind=True)
def start_pipeline(self, data):
    # Initial sequential tasks
    initial_tasks = chain(
        run_data_cleaning_task.s(data),
        prepare_data_task.s(),
        run_demand_predictor_task.s(),
        feature_importances_task.s(),
        create_demand_score_task.s()
    )
    # First group of parallel tasks
    categorization_group = group(
        categorization_task.s(),
        reordering_task.s(),
        obsolescence_risk_task.s(),
    )
    # Workflow with chord to ensure synchronization
    workflow = (
        initial_tasks |
        chord(categorization_group, aggregate_results.s()) |  
        monthly_sales_task.s() |
        database_builder.s()
    )

    result = workflow.apply_async()
    return result
