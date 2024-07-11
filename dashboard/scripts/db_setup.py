import logging
import pandas as pd
from dashboard.scripts.create_db import create_database_schema
from dashboard.scripts.create_db import Parts, Sales
from dashboard.setup.utils import get_engine_and_session
from dashboard.scripts.data_insertion import insert_data_using_pandas, preprocess_dataframe

def main(current_task, combined_data):
    logging.basicConfig(level=logging.INFO)
    db_file_path = 'data/databases/partswise_island_moto.db'
    create_database_schema(db_file_path)
    engine, _ = get_engine_and_session(db_file_path)

    try:
        logging.info("Loading data into DataFrames...")
        parts_data = pd.DataFrame(combined_data['parts_data']['data'], columns=combined_data['parts_data']['columns'])
        new_monthly_sales_df = pd.DataFrame(combined_data['monthly_sales']['data'], columns=combined_data['monthly_sales']['columns'])


        logging.info("Renaming columns...")
        rename_map_parts = {'total_cost': 'cost'}
        drop_cols_parts = ['rolling_12_month_sales', 'rolling_3_month_sales', 'seasonal_component']

        parts_data = preprocess_dataframe(parts_data, rename_map_parts, drop_cols_parts)

        # Insert data into database using Pandas .to_sql
        insert_data_using_pandas(engine, 'parts', parts_data, Parts, 'replace')
        insert_data_using_pandas(engine, 'sales', new_monthly_sales_df, Sales, 'append')

        return {'status': 'SUCCESS', 'message': 'Database created successfully, now building your dashboard...', 'data': db_file_path}

    except ValueError as e:
        logging.error(f"Invalid JSON format: {e}")
        return {'status': 'FAILURE', 'message': f'Error processing data: Invalid file format.'}
    except Exception as e:
        logging.error(f"Error processing data: {str(e)}")
        return {'status': 'FAILURE', 'message': f'Error processing data: {str(e)}'}

