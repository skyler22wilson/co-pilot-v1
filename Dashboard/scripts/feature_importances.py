import pandas as pd
from sklearn.inspection import permutation_importance
import tempfile
import logging
import json
from joblib import load
from sklearn.preprocessing import MinMaxScaler

# Constants
CONFIG_FILE = "Dashboard/configuration/SeasonalConfig.json"
DEMAND_PREDICTOR = "Dashboard/Models/demand_predictor/demand_pipeline.joblib"
RANDOM_STATE = 42


def check_for_nan(df, name):
    if df.isnull().any().any():
        nan_columns = df.columns[df.isnull().any()].tolist()
        error_message = f"NaN values found in {name}. Columns with NaN: {nan_columns}"
        logging.error(error_message)
        print(error_message)  # Print to console
    else:
        logging.info(f"No NaN values found in {name}")
        print(f"No NaN values found in {name}")  # Print to console

def verify_selected_features(dataset, selected_features):
    missing_features = [f for f in selected_features if f not in dataset.columns]
    if missing_features:
        logging.error(f"Selected features not found in dataset: {missing_features}")
    else:
        logging.info("All selected features are present in the dataset.")

def calculate_permutation_importance(model, X, y, n_repeats):
    try:
        y_pred = model.predict(X)
        if pd.Series(y_pred).isnull().any():
            print("NaN values found in model predictions")
    except Exception as e:
        print(f"Error during model prediction: {e}")

    # Calculate permutation importance
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=-1)
    permutation_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values(by='importance', ascending=False)

    # Scaling the importance scores
    scaler = MinMaxScaler()
    permutation_importance_df['scaled_importance'] = scaler.fit_transform(permutation_importance_df[['importance']])

    return permutation_importance_df

def load_configuration(config_path):
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        return config
    except Exception as e:
        logging.error("Error loading configuration: %s", str(e))
        return None

def preprocess_data(dataset, selected_features):
    print(f"Dataset type: {type(dataset)}")
    print(f"Selected_features type: {type(selected_features)}")
    print(f"Selected_features: {selected_features}")
    # Keep only selected features and target variables
    X = dataset[selected_features]
    y = dataset['rolling_12_month_sales']
    return X, y

def main(current_task, input_data):
    print('Calculating Features...')
    logging.basicConfig(filename='Logs/feature_selection.log', level=logging.INFO)
    logging.info("Feature selection script is running...")

    # Load configuration
    config = load_configuration(CONFIG_FILE)
    if config is None:
        logging.error('Failed to load configuration.')
        return

    # Load data
    print(type(input_data))
    data = json.loads(input_data)  # Correctly load the JSON string into a Python dictionary
    parts_data = pd.DataFrame(data['data'], columns=data['columns'])

    if parts_data is None:
        logging.error("Failed to load data.")
        return
    
    # Load selected features from JSON path specified in config
    json_path = 'Dashboard/Models/demand_predictor/general_model_details.json'
    with open(json_path, 'r') as file:
        general_model_info = json.load(file)
        selected_features = general_model_info.get('selected_features', [])

    # Verify selected features
    verify_selected_features(parts_data, selected_features)

    print(f'Selected Features {selected_features}')
    # Preprocess data using only selected features
    X, y = preprocess_data(parts_data, selected_features)

    # Check for NaN values in X and y
    check_for_nan(X, "X (features)")
    check_for_nan(pd.DataFrame(y), "y (target)")

    pipeline = load(DEMAND_PREDICTOR)
    logging.info(f"Loaded model: {pipeline}")

    try:
        logging.info("Starting permutation importance calculation.")
        importance_df = calculate_permutation_importance(pipeline, X, y, n_repeats=5)
        importance_data = importance_df.to_json(orient='split')
        parts_data = parts_data.to_json(orient='split')

        logging.info("Feature selection script completed.")
        print('Significant features finished calculating')
        combined_data = {
            'parts_data': parts_data,
            'importance_data': importance_data
        }
        
        # Return the combined data as a JSON string
        return json.dumps(combined_data)
    except Exception as e:
        logging.error(f"Error during permutation importance calculation: {e}")
        print(f"Error during permutation importance calculation: {e}")  # Also print to console
        return False







