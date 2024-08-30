import json
import logging
import joblib
import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import load
from dashboard.models.demand_predictor.predictions import calculate_demand
from dashboard.setup.utils import load_configuration
from scipy import stats

CONFIG_FILE = "dashboard/configuration/SeasonalConfig.json"
DEMAND_PREDICTOR = '/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/dashboard/models/demand_predictor/preprocessor.joblib'
PREPROCESSOR_PATH = '/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/dashboard/models/demand_predictor/preprocessor.joblib'
MODEL_INFO_PATH = '/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/dashboard/models/demand_predictor/general_model_details.json'

def verify_selected_features(parts_data, selected_features):
    logging.info("Verifying selected features...")
    missing_features = [feature for feature in selected_features if feature not in parts_data.columns]
    if missing_features:
        logging.warning(f"Missing features in data: {missing_features}")
    logging.info("Feature verification completed.")

def preprocess_data(parts_data, selected_features):
    """ Preprocess data for demand model """
    logging.info("Loading and preprocessing data for demand model...")
    logging.info(f"Parts data type: {type(parts_data)}")
    try:
        model_data = parts_data.filter(pl.col('months_no_sale') < 12)
        X = model_data.select(selected_features)
        y = model_data.select('rolling_12m_sales')
        return X, y
    except Exception as e:
        logging.error(f"Error in preprocessing data: {str(e)}")
        raise

def check_for_nan(df, name):
    if df.select(pl.all().is_null().sum()).to_numpy().sum() > 0:
        logging.warning(f"{name} contains NaN values.")

def calculate_demand_score(parts_data, shap_values, negative_features):
    # Adjust for negative features
    for feature in negative_features:
        if feature in parts_data.columns:
            parts_data = parts_data.with_columns((pl.col(feature) * -1).alias(feature))

    # Calculate weighted scores using SHAP values
    shap_sum = np.abs(shap_values).sum(axis=1)
    weighted_scores = pl.Series(shap_sum)

    # Apply Yeo-Johnson transformation to the weighted scores
    transformed_demand_scores, _ = stats.yeojohnson(weighted_scores)
    transformed_demand_scores = pl.Series(transformed_demand_scores).cast(pl.Float64)

    scaler = StandardScaler()
    demand_score_scaled = scaler.fit_transform(transformed_demand_scores.to_numpy().reshape(-1, 1)).flatten()

    # Apply MinMaxScaler to scale the scores to [0, 1]
    demand_score_normalized = stats.norm.cdf(demand_score_scaled)

    # Convert the normalized scores back to a Polars Series
    demand_score_normalized = pl.Series(demand_score_normalized)

    # Set demand to 0 for obsolete parts
    parts_data = parts_data.with_columns(
        pl.when(pl.col('months_no_sale') >= 12)
        .then(0.0)  # Ensure the value is float
        .otherwise(demand_score_normalized)
        .alias('demand')
    )

    for feature in negative_features:
        if feature in parts_data.columns:
            parts_data = parts_data.with_columns((pl.col(feature) * -1).alias(feature))

    return parts_data

def main(current_task, input_data):
    print('Calculating Features...')
    logging.basicConfig(filename='Logs/feature_selection.log', level=logging.INFO)
    logging.info("Demand Score script is running...")

    config = load_configuration(CONFIG_FILE)
    if config is None:
        logging.error('Failed to load configuration.')
        return
    
    try:
        # Ensure input_data is a JSON object and load it into a Polars DataFrame
        input_json = json.loads(input_data)  # Ensure input_data is a JSON object
        parts_data = pl.DataFrame(input_json)  # Load data as a Polars DataFrame

        logging.info(f"Parts data loaded: {parts_data.head()}")
        logging.info(f"Parts data loaded: {parts_data.shape}")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    try:
        with open(MODEL_INFO_PATH, 'r') as file:
            general_model_info = json.load(file)
            selected_features = general_model_info.get('selected_features', [])
        logging.info(f"Selected features: {selected_features}")
    except Exception as e:
        logging.error(f"Error loading model details: {e}")
        return

    try:
        verify_selected_features(parts_data, selected_features)
        logging.info(f'Selected Features Verified: {selected_features}')
    except Exception as e:
        logging.error(f"Error verifying selected features: {e}")
        return

    try:
        X, y = preprocess_data(parts_data, selected_features)
        check_for_nan(X, "X (features)")
        check_for_nan(y, "y (target)")
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        return

    try:
        pipeline = load(PREPROCESSOR_PATH)
        logging.info(f"Loaded preprocessor: {pipeline}")
    except Exception as e:
        logging.error(f"Error loading preprocessor: {e}")
        return

    try:
        negative_features = ['1m_days_supply', '3m_days_supply', '12m_days_supply', 'days_of_inventory_outstanding', 'negative_on_hand']
        # Call the calculate_demand function to get demand scores and SHAP values
        logging.info(f"Loading the demand score model...")
        shap_values = calculate_demand(input_data)
        logging.info(f"SHAP values calculated: {shap_values}")

        # Calculate demand score using the SHAP values
        parts_data = calculate_demand_score(parts_data, shap_values, negative_features)

        parts_data.write_csv('/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/data/processed_data/processed_data_with_demand_score.csv')
        logging.info(f"Processed data saved")

        parts_data_json = parts_data.write_json()
    
        return parts_data_json

    except Exception as e:
        logging.error(f"Error processing data: {e}")
        return


