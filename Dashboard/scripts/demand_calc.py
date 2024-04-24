import pandas as pd
import json
import logging
import os

CONFIG_FILE = "Dashboard/configuration/SeasonalConfig.json"
LOGGING_DIR = "Logs"

def load_config(config_file):
    try:
        with open(config_file, 'r') as f:
            config = json.load(f) 
        return config
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in configuration file.")
    except FileNotFoundError:
        raise ValueError("Configuration file not found.")


def calculate_demand_score(df, feature_importances_df):
    # Ensure that the feature importances DataFrame has the necessary columns
    if 'scaled_importance' not in feature_importances_df.columns or 'feature' not in feature_importances_df.columns:
        logging.error("'scaled_importance' or 'feature' column is missing in the feature_importances_df.")
        return df

    # Map the importances to the corresponding features
    feature_importances = feature_importances_df.set_index('feature')['scaled_importance'].to_dict()

    # Select only the relevant columns for calculating demand score
    feature_columns = [col for col in df.columns if col in feature_importances]

    # Calculate demand score based on feature importances
    df['demand'] = df[feature_columns].apply(
        lambda row: sum(feature_importances[col] * row[col] for col in feature_columns),
        axis=1
    )

    # Set 'demand' to 0 for obsolete items
    df['demand'] = df.apply(lambda x: 0 if x['months_no_sale'] >= 12 else x['demand'], axis=1)

    # Normalize the demand score for non-obsolete items by ranking
    non_obsolete_mask = df['months_no_sale'] < 12
    df.loc[non_obsolete_mask, 'demand'] = df.loc[non_obsolete_mask, 'demand'].rank(method='dense', pct=True)

    return df

def main(current_task, input_data):
    print('Demand script running...')
    logging.basicConfig(filename=os.path.join(LOGGING_DIR, 'demand_script.log'), 
                        level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')    
    try:
        print("Loading combined data from input...")
        combined_data = json.loads(input_data)  # Load the JSON string into a Python dictionary
        print("Successfully loaded combined data.")
        
        # Deserialize the parts data from the JSON string within combined_data
        print("Deserializing parts data from JSON...")
        parts_data_json = combined_data['parts_data']
        parts_data = pd.read_json(parts_data_json, orient='split')

        
        # Deserialize the feature importances data from the JSON string within combined_data
        print("Deserializing feature importances data from JSON...")
        feature_importances_json = combined_data['importance_data']
        feature_importances_df = pd.read_json(feature_importances_json, orient='split')
        print(f"Feature importances data loaded successfully. DataFrame shape: {feature_importances_df.shape}")
        
        # Print a snippet of the feature importances DataFrame for inspection
        print("Preview of feature importances DataFrame:", feature_importances_df.head(), sep='\n')
        
        print("Calculating demand score adjustment based on feature importances...")
        df_adjusted = calculate_demand_score(parts_data, feature_importances_df)
        print("Demand score adjustment completed successfully.")
        df_adjusted.to_feather("/Users/skylerwilson/Desktop/PartsWise/Data/Processed/parts_data.feather")

        # Save the adjusted dataframe
        if df_adjusted is not None:
            data = df_adjusted.to_json(orient='split')
            print('Demand script Finished')
            return data

    except ValueError as e:
        logging.error(f"Invalid JSON format: {e}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: Invalid file format.'})
        return False
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        current_task.update_state(state='FAILURE', meta={'progress': 0, 'message': f'Error processing data: {str(e)}'})
        return False

