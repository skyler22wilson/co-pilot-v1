import pandas as pd
import json
import logging
import os
import uuid

LOGGING_DIR = "Logs"

def create_summary_parts_table(df):
    summary_df = df.groupby('supplier_name').agg(
        total_quantity=pd.NamedAgg(column='quantity', aggfunc='sum'),
        total_negative_on_hand=pd.NamedAgg(column='negative_on_hand', aggfunc='sum'),
        total_cost=pd.NamedAgg(column='total_cost', aggfunc='sum'),
        average_margin=pd.NamedAgg(column='margin_percentage', aggfunc='mean'),
        total_sales_revenue=pd.NamedAgg(column='sales_revenue', aggfunc='sum'),
        total_cogs=pd.NamedAgg(column='cogs', aggfunc='sum'),
        total_gross_profit=pd.NamedAgg(column='gross_profit', aggfunc='sum'),
        average_turnover=pd.NamedAgg(column='annual_turnover', aggfunc='mean'),
        average_days_supply=pd.NamedAgg(column='annual_days_supply', aggfunc='mean'),
        average_months_no_sale=pd.NamedAgg(column='months_no_sale', aggfunc='mean'),
        average_obsolescence_risk=pd.NamedAgg(column='obsolescence_risk', aggfunc='mean'),
        average_demand=pd.NamedAgg(column='demand', aggfunc='mean')
    ).reset_index()

    # Assign a UUID to each supplier
    summary_df['supplier_id'] = summary_df['supplier_name'].apply(lambda x: str(uuid.uuid5(uuid.NAMESPACE_DNS, x)))

    return summary_df

def main(current_task, input_data):
    print('Starting data processing workflow...')
    log_filename = os.path.join(LOGGING_DIR, 'supplier_parts_summary.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Load input data
        # Check if input_data is a string or already a dictionary
        if isinstance(input_data, str):
            logging.info("Loading the JSON data")
            original_data = json.loads(input_data)
        elif isinstance(input_data, dict):
            logging.info("Input data is already a dictionary")
            original_data = input_data
        else:
            raise ValueError("Input data should be a JSON string or a dictionary.")

        # Ensure that the necessary keys are present
        if 'data' not in original_data or 'columns' not in original_data:
            raise ValueError("Input data should contain 'data' and 'columns' keys.")

        logging.info(f"Input data keys: {original_data.keys()}")
        logging.info(f"Columns: {original_data['columns']}")

        dataset = pd.DataFrame(original_data['data'], columns=original_data['columns'])
        logging.info(f"Input data after conversion: {dataset.columns}")

        # Log dataset details
        logging.info(f"Initial dataset loaded with {len(dataset)} rows.")

        # Generate summary parts table
        parts_summary_df = create_summary_parts_table(dataset)
        parts_summary_df_path = "/Users/skylerwilson/Desktop/PartsWise/Data/Processed/parts_summary.feather"
        parts_summary_df.to_feather(parts_summary_df_path)
        logging.info("Parts summary table saved successfully.")

        # Serialize DataFrame to JSON
        parts_summary_json = parts_summary_df.to_json(orient='split')

        # Log completion
        logging.info("Data processing completed successfully.")

        # Return the JSON string directly
        return parts_summary_json

    except ValueError as e:
        logging.error(f"Invalid JSON format: {e}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: Invalid file format.'})
        return False
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: {str(e)}'})
        return False




