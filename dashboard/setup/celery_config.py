from celery import Celery, chord, group, chain, shared_task
from dashboard.setup.utils import convert_to_float, convert_to_int, load_configuration, create_schema, update_schema
from dashboard.scripts.data_clean import main as data_clean
from dashboard.scripts.prepare_cols import main as column_preperation
from dashboard.scripts.demand_calc import main as create_demand_score
from dashboard.scripts.create_categories import main as create_categories
from dashboard.scripts.check_reorder import main as calculate_reorder
from dashboard.models.obsolecence_predictor.obsolete_predictor import main as obsolescence_risk
from dashboard.scripts.parts_data import main as get_parts_data
from dashboard.scripts.sql_converter import main as sql_conversion
import json
import polars as pl
import logging

# Use Redis as the message broker
app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

@shared_task(bind=True)
def read_in_data_task(self, file_path):
    config = load_configuration('/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/dashboard/configuration/InitialConfig.json')
    new_column_names = config['ColumnNames']
    json_data = None  # Initialize json_data at the start to avoid UnboundLocalError
    schema_overrides = {column: pl.Utf8 for column in new_column_names}
    try:
        df = pl.read_csv(file_path, new_columns=new_column_names, schema_overrides=schema_overrides, infer_schema_length=100, ignore_errors=True)
        df = df.filter(~pl.col('part_number').str.contains("Parts Inventory: 73897 records"))
        df = convert_to_float(df, 'price')
        df = convert_to_float(df, 'margin')
        
        for col in config['SalesColumns']:
            df = convert_to_int(df, col)
        
        #update the schema
        schema = create_schema()
        df = update_schema(df, schema)
        json_data = df.write_json()  # Convert DataFrame to JSON
    except ValueError as ve:
        print(f'ValueError during processing: {ve}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
    
    if json_data:
        return json_data
    else:
        logging.error("Failed to process data into JSON.")
        return False


@shared_task(bind=True)
def run_data_cleaning_task(self, input_data):
    """
    A Celery task that wraps the data cleaning and processing logic.
    This task is designed to run the main function and handle its state updates and exceptions.
    """
    if not input_data:
        logging.error("Input JSON data is missing.")
        return False

    # Pass the current task instance to your main function for state updates
    return data_clean(self, input_data)

@shared_task(bind=True)
def prepare_data_task(self, input_data):
    """
    A Celery task that wraps the column preperation and processing logic.
    This task is designed to run the main function and handle its state updates and exceptions.
    """
    return column_preperation(self, input_data)

@shared_task(bind=True)
def create_demand_score_task(self, input_data):
    return create_demand_score(self, input_data)

@shared_task(bind=True)
def categorization_task(self, input_data):
    return create_categories(self, input_data)

@shared_task(bind=True)
def reordering_task(self, input_data):
    return calculate_reorder(self, input_data)

@shared_task(bind=True)
def obsolescence_risk_task(self, input_data):
    return obsolescence_risk(self, input_data)

@shared_task(bind=True)
def aggregate_results(self, data_from_previous_tasks):
    """Aggregate results from multiple tasks and apply post-aggregation filtering."""
    logging.info(f"Number of tasks received: {len(data_from_previous_tasks)}")

    dataframes = []
    #original_df = pl.read_json(StringIO(json.loads(original_data))) if isinstance(original_data, str) else pl.DataFrame(original_data)
    for idx, data in enumerate(data_from_previous_tasks):
        try:
            logging.info(f"Processing data from task {idx}")
            logging.info(f"Type of data: {type(data)}")
            
            if isinstance(data, str):
                # If it's a string, parse it as JSON
                parsed_data = json.loads(data)
            else:
                # If it's already a list or dict, use it directly
                parsed_data = data

            # Convert the list of dictionaries to a Polars DataFrame
            df = pl.DataFrame(parsed_data)
            
            dataframes.append(df)
            logging.info(f"DataFrame {idx} shape: {df.shape}")
            logging.info(f"DataFrame {idx} columns: {df.columns}")
        except Exception as e:
            logging.error(f"Error processing data from task {idx}: {e}")
            logging.error(f"Problematic data content: {str(data)[:1000]}...")  # Convert to string before slicing
            raise

    try:
        # Merge all dataframes from parallel tasks
        concat_df = pl.concat(dataframes, how="align")
        
        # Group by 'part_number' and aggregate
        aggregated_df = concat_df.group_by('part_number').agg([
            pl.all().exclude('part_number').first()
        ])

        logging.info(f'Aggregated DF columns: {aggregated_df.columns}')


        aggregated_df.write_csv("/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/data/processed_data/processed_data_final.csv")

        filtered_json = aggregated_df.write_json()

        logging.info(f"Length of filtered aggregated JSON file: {len(aggregated_df)} rows")
        logging.info(f"Columns in final DataFrame: {aggregated_df.columns}")
        return filtered_json
    except Exception as e:
        logging.error(f"Error during aggregation: {e}")
        logging.error(f"DataFrames shapes: {[df.shape for df in dataframes]}")
        logging.error(f"DataFrames columns: {[df.columns for df in dataframes]}")
        raise


@shared_task(bind=True)
def create_dataframes(self, input_data):
    result = get_parts_data(self, input_data)
    if result is None:
        return None
    return result['output_path']

@shared_task(bind=True)
def convert_to_sql(self, input_data):
    conn = sql_conversion(input_data)
    return "Database created successfully"

@shared_task(bind=True)
def start_pipeline(self, data):
    # Initial sequential tasks
    initial_tasks = chain(
        read_in_data_task.s(data),
        run_data_cleaning_task.s(),
        prepare_data_task.s(),
        create_demand_score_task.s()
    )

    categorization_group = chord(
        group(
            categorization_task.s(),
            reordering_task.s(),
            obsolescence_risk_task.s(),
        ),
        aggregate_results.s()
    )

    final_tasks = chain(
        create_dataframes.s(),
        convert_to_sql.s()
    )

    # Combine all parts into one flow
    workflow = chain(
        initial_tasks,
        categorization_group,
        final_tasks
    )

    # Apply the entire workflow asynchronously
    result = workflow.apply_async()
    return result

