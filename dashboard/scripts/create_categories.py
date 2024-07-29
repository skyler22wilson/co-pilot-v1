import polars as pl
import logging
import os
from io import StringIO

CONFIG_FILE = "dashboard/configuration/SeasonalConfig.json"
LOGGING_DIR = "Logs"

def categorizer(df, demand_threshold = 0.75):
    """
    Categorize inventory items based on their sales history and demand.

    This function categorizes inventory items into four categories: 'essential', 'non-essential', 
    'nearing_obsolete', and 'obsolete' based on the number of months without sales and the demand.
    The demand threshold is computed as the 95th percentile of the demand column.

    Args:
        df (pl.DataFrame): Input DataFrame containing inventory data. 
                           Must include 'months_no_sale' and 'demand' columns.

    Raises:
        ValueError: If the required columns ('months_no_sale', 'demand') are not present in the DataFrame.

    Returns:
        pl.DataFrame: DataFrame with an additional 'part_status' column indicating the category of each item.
    """
    # Ensure that required columns are present
    required_columns = ['months_no_sale', 'demand']
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns {missing_columns} in the dataframe")

    # Initialize the 'inventory_category' column to 'idle inventory'
    try:
        df = df.with_columns(pl.lit('idle').alias('part_status'))
        logging.info("Initialized 'part_status' column")
    except Exception as e:
        logging.error(f"Error initializing 'part_status' column: {str(e)}")
        raise

    # Assign 'essential active' category
    try:
        df = df.with_columns(
            pl.when((pl.col('months_no_sale') <= 3) & (pl.col('demand') >= demand_threshold))
            .then(pl.lit('essential'))
            .otherwise(pl.col('part_status'))
            .alias('part_status')
        )
        logging.info("Assigned 'essential' category")
    except Exception as e:
        logging.error(f"Error assigning 'essential' category: {str(e)}")
        raise

    # Assign 'active parts' category
    try:
        df = df.with_columns(
            pl.when((pl.col('months_no_sale') > 3) & (pl.col('months_no_sale') <= 6))
            .then(pl.lit('active'))
            .otherwise(pl.col('part_status'))
            .alias('part_status')
        )
        logging.info("Assigned 'active' category")
    except Exception as e:
        logging.error(f"Error assigning 'active' category: {str(e)}")
        raise

    # Assign 'obsolete inventory' category
    try:
        df = df.with_columns(
            pl.when(pl.col('months_no_sale') > 12)
            .then(pl.lit('obsolete'))
            .otherwise(pl.col('part_status'))
            .alias('part_status')
        )
        logging.info("Assigned 'obsolete' category")
    except Exception as e:
        logging.error(f"Error assigning 'obsolete' category: {str(e)}")
        raise

    return df.select('part_number', 'part_status')

def main(current_task, input_data):
    print('Creating Categories')
    try:
        # Set up logging
        log_filename = os.path.join(LOGGING_DIR, 'categorizer_script.log')
        logging.basicConfig(filename=log_filename, level=logging.INFO)
        
        # Load dataset
        df = pl.read_json(StringIO(input_data))

        current_task.update_state(state='PROGRESS', meta={'message': 'Creating part_status...'})

        df = categorizer(df)
        logging.info(f"Categorization completed. Dataframe head: {df.head()}")
        print(f'Categorizer Columns: {df.columns}')

        # Convert DataFrame to JSON string for return
        json_dataset = df.write_json()

        return json_dataset

    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        return False

