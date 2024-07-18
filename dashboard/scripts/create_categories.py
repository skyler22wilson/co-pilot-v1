import polars as pl
import logging
import os
from io import StringIO

CONFIG_FILE = "dashboard/configuration/SeasonalConfig.json"
LOGGING_DIR = "Logs"

def categorizer(df):
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
        pl.DataFrame: DataFrame with an additional 'inventory_category' column indicating the category of each item.
    """
    # Ensure that required columns are present
    required_columns = ['months_no_sale', 'demand']
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns {missing_columns} in the dataframe")

    # Compute the demand threshold for a stricter cutoff
    demand_threshold = df['demand'].quantile(0.95)
    print(demand_threshold)

    # Initialize the 'inventory_category' column to 'non-essential'
    df = df.with_columns(pl.lit('non-essential').alias('inventory_category'))

    # Assign 'obsolete' category
    df = df.with_columns(
        pl.when(pl.col('months_no_sale') >= 12)
        .then('obsolete')
        .otherwise(pl.col('inventory_category'))
        .alias('inventory_category')
    )

    # Assign 'nearing_obsolete' category
    df = df.with_columns(
        pl.when((pl.col('months_no_sale') >= 7) & (pl.col('months_no_sale') < 12))
        .then('nearing_obsolete')
        .otherwise(pl.col('inventory_category'))
        .alias('inventory_category')
    )

    # Assign 'essential' category based on stricter criteria
    essential_criteria = (
        (pl.col('months_no_sale') <= 6) &
        (pl.col('demand') > demand_threshold)
    )
    df = df.with_columns(
        pl.when(essential_criteria)
        .then('essential')
        .otherwise(pl.col('inventory_category'))
        .alias('inventory_category')
    )

    return df

def main(current_task, input_data):
    print('Creating Categories')
    try:
        # Set up logging
        log_filename = os.path.join(LOGGING_DIR, 'categorizer_script.log')
        logging.basicConfig(filename=log_filename, level=logging.INFO)

        # Load dataset
        original_data = pl.read_json(StringIO(input_data))
        df = pl.DataFrame(original_data['data'], schema=original_data['columns'])
        logging.info(f"Length of categories JSON file: {len(df)} rows")

        current_task.update_state(state='PROGRESS', meta={'message': 'Creating inventory categories...'})

        df = categorizer(df)

        # Convert DataFrame to JSON string for return
        json_dataset = df.write_json()

        return json_dataset

    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        current_task.update_state(state='FAILURE', meta={'message': f'Error processing data: {str(e)}'})
        return False

