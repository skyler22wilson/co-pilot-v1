import polars as pl
import logging
from dashboard.setup.utils import load_model_info
import os
from io import StringIO
import traceback
from functools import reduce

# Load model hyperparameters and preprocessor
LOGGING_DIR = "/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/dashboard/models/obsolecence_predictor/log_file.log"


def get_feature_indices(preprocessor, selected_features):
    """ Map selected feature names to their corresponding indices in the preprocessed array. """
    try:
        # Extract the names of all features in the preprocessed array
        # `transformers_` is a list of tuples representing each transformer
        # Each tuple: (name, transformer_object, feature_names_list)
        all_features = preprocessor.transformers_[0][2]

        # Find indices of each selected feature in the preprocessed feature list
        feature_indices = [list(all_features).index(feature) for feature in selected_features]
        return feature_indices

    except ValueError as e:
        logging.error(f"Error finding feature indices: {e}")
        raise

def calculate_obsolescence_risk(df):
    model_info_path = '/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/dashboard/models/obsolecence_predictor/general_model_details.json'
    preprocessor_path = '/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/dashboard/models/obsolecence_predictor/preprocessor.joblib'
    model_path = '/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/dashboard/models/obsolecence_predictor/best_calibrated_random_forest_model.joblib'

    # Load model configuration, preprocessor, and model
    model_config, preprocessor, model = load_model_info(model_info_path, preprocessor_path, model_path)
    
    selected_features = model_config['selected_features']

    # Filter out obsolete items and assign obsolescence_dummy
    non_obsolete_df = df.filter(pl.col('months_no_sale') < 12).with_columns(
        pl.when(pl.col('months_no_sale') >= 6)
        .then(1)
        .otherwise(0)
        .alias('obsolescence_dummy')
    )

    # Preprocess the data and select features
    X = non_obsolete_df.select(selected_features).to_pandas()
    X_transformed = preprocessor.transform(X)
    feature_indices = get_feature_indices(preprocessor, selected_features)
    logging.info(f"Feature indices: {feature_indices}")
    X_selected = X_transformed[:, feature_indices]
    logging.info(f"Selected columns: {X_selected}")

    # Predict probabilities using the calibrated model
    predicted_probs = model.predict_proba(X_selected)[:, 1]

    # Assign obsolescence risk to the original DataFrame
    non_obsolete_df = non_obsolete_df.with_columns(pl.Series(name='obsolescence_risk', values=predicted_probs))
    
    # Merge the results back to the original dataframe
    df = df.join(non_obsolete_df.select(['part_number', 'obsolescence_risk']), on='part_number', how='left')
    print(f'Obsolete data frame type: {type(df)}')
    
    # Fill NaN values (for obsolete items) with 1.0
    df = df.with_columns(pl.col('obsolescence_risk').fill_null(1.0))

    return df

def main(current_task, input_data):
    print('Obsolescence Risk Calculation Script Running...')
    
    # Initialize logging
    log_filename = os.path.join(LOGGING_DIR, 'obsolescence_risk_script.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Script started.")

    try:
        print('Loading data...')
        print(f'Input data type: {type(input_data)}')
        
        
        df = pl.read_json(StringIO(input_data))
        print(df.head())
        logging.info('Data loaded successfully.')

        print('Calculating obsolescence risk...')
        df = calculate_obsolescence_risk(df)
        
        # Convert DataFrame to JSON
        json_data = df.write_json()
        logging.info('Obsolescence risk calculation completed successfully.')

        return json_data

    except ValueError as e:
        error_msg = f"Invalid JSON format: {str(e)}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error in processing: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        raise Exception(error_msg)


