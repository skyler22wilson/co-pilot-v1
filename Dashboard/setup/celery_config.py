from celery import Celery, chord, group, chain
from Dashboard.scripts.data_clean import main as data_clean
from Dashboard.scripts.prepare_cols import main as column_preperation
from Dashboard.scripts.feature_importances import main as calculate_features
from Dashboard.scripts.demand_calc import main as create_demand_score
from Dashboard.scripts.create_categories import main as create_categories
from Dashboard.scripts.check_reorder import main as calculate_reorder
from Dashboard.Models.obsolecence_predictor.obsolete_predictor import main as obsolescence_risk
from Dashboard.scripts.monthly_sales import main as monthly_sales
from Dashboard.scripts.parts_data import main as get_parts_data
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
    """Aggregate results from multiple tasks and apply post-aggregation filtering."""
    dataframes = []

    for idx, data in enumerate(data_from_previous_tasks):
        try:
            # Check the structure of the JSON string before converting to DataFrame
            json_data = json.loads(data)
            columns = json_data.get("columns")
            index = json_data.get("index")
            data_values = json_data.get("data")

            if not columns or not data_values:
                raise ValueError(f"Missing columns or data in DataFrame {idx}")

            df = pd.DataFrame(data_values, columns=columns, index=index)
            dataframes.append(df)
            logging.info(f"DataFrame {idx} shape: {df.shape}")

        except Exception as e:
            logging.error(f"Error processing DataFrame {idx}: {e}")
            logging.error(f"Problematic data content: {data}")
            raise

    def merge_dfs(left, right):
        overlapping_columns = left.columns.intersection(right.columns).drop('part_number', errors='ignore')
        right_adjusted = right.drop(columns=overlapping_columns)
        return pd.merge(left, right_adjusted, on='part_number', how='outer')

    try:
        aggregated_data = reduce(merge_dfs, dataframes)

        # Apply filtering logic after aggregation
        filtered_aggregated_data = aggregated_data[(aggregated_data['quantity'] > 0) | (aggregated_data['negative_on_hand'] != 0)]
        filtered_json = filtered_aggregated_data.to_json(orient='split')

        logging.info(f"Length of filtered aggregated JSON file: {len(filtered_aggregated_data)} rows")
        return filtered_json
    except Exception as e:
        logging.error(f"Error during aggregation: {e}")
        raise


@app.task(bind=True)
def monthly_sales_task(self, input_data):
    result = monthly_sales(self, input_data)
    return result


@app.task(bind=True)
def parts_data_task(self, input_data):
    result = get_parts_data(self, input_data)
    return result

@app.task(bind=True)
def combine_datasets(self, results):
    """Combine all results into a single dictionary."""
    parts_data, monthly_sales = results
            
    # Parse each result from JSON to a dictionary
    parts_data = json.loads(parts_data)
    monthly_sales = json.loads(monthly_sales)

    combined_data = {
        "parts_data": parts_data,
        "monthly_sales": monthly_sales,
    }

    return combined_data

@app.task(bind=True)
def database_builder(self, input_data):
    return build_database(self, input_data)


@app.task(bind=True)
def start_pipeline(self, data):
    # Initial sequential tasks
    initial_tasks = chain(
        run_data_cleaning_task.s(data),
        prepare_data_task.s(),
        feature_importances_task.s(),
        create_demand_score_task.s()
    )

    # First group of parallel tasks wrapped in a chord to ensure synchronization
    categorization_group = chord(
        group(
            categorization_task.s(),
            reordering_task.s(),
            obsolescence_risk_task.s(),
        ),
        aggregate_results.s()  # This is the callback for the chord
    )

    # Final group of tasks that depends on the output of the categorization group
    data_group = chord(
        group(
            parts_data_task.s(),
            monthly_sales_task.s(),
        ),
        combine_datasets.s() 
    )

    # Combine all parts into one flow
    workflow = chain(
        initial_tasks,
        categorization_group,
        data_group, 
        database_builder.s()
    )

    # Apply the entire workflow asynchronously
    result = workflow.apply_async()
    return result
