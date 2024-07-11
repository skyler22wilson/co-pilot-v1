import pandas as pd
import json

def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def concatenate_dataframes(file_paths):
    """Concatenates dataframes from a list of file paths."""
    dfs = [pd.read_csv(path, low_memory=False) for path in file_paths]
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def rename_columns_based_on_config(df, config):
    """Renames DataFrame columns based on the order specified in the configuration."""
    # Extract the ordered list of new column names from the configuration
    new_column_names = list(config["columns"].keys())
    
    # Rename the DataFrame columns based on the new column names
    # This step is done after concatenation to ensure the columns are correctly aligned
    df.columns = new_column_names[:len(df.columns)]  # Adjust column names only up to the length of existing columns
    return df

def remove_empty_rows_columns(df):
    """Removes entirely empty rows and columns."""
    df.dropna(axis=0, how='all', inplace=True)  # Remove rows where all elements are NaN
    df.dropna(axis=1, how='all', inplace=True)  # Remove columns where all elements are NaN
    return df

def load_clean_save_data(file_paths, output_file, config_path):
    """
    Loads data from a list of file paths, concatenates them, then renames columns based on configuration,
    removes entirely empty rows and columns, and saves the cleaned DataFrame to a CSV file.
    """
    # Load the configuration
    config = load_config(config_path)
    
    # Concatenate DataFrames
    combined_df = concatenate_dataframes(file_paths)
    
    # Remove entirely empty rows and columns
    combined_df = remove_empty_rows_columns(combined_df)

    # Rename columns based on configuration
    combined_df = rename_columns_based_on_config(combined_df, config)
    
    # Save the cleaned DataFrame to a CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")


def rename_columns_based_on_config(df, config):
    """
    Renames DataFrame columns based on the order specified in the configuration.
    
    Parameters:
    - df: DataFrame to be processed.
    - config: The configuration dictionary with a 'columns' key detailing the desired schema.
    """
    # Extract the ordered list of new column names from the configuration
    new_column_names = list(config["columns"].keys())
    
    # Check if the number of new column names matches the number of columns in the DataFrame
    if len(new_column_names) != len(df.columns):
        raise ValueError("The number of columns in the DataFrame does not match the number of columns specified in the configuration.")
    
    # Assign the new column names to the DataFrame
    df.columns = new_column_names

    return df

def main():
    print('Merging the data...')
    # Example usage
    config_path = "Dashboard/configuration/InitialConfig.json"
    file_paths = [
        "Data/Raw/PartsWise_data.csv",
        "Data/Raw/PartsWise_data2.csv"
    ]
    output_file = "Data/Raw/PartsWiseData.csv"

    load_clean_save_data(file_paths, output_file, config_path)
    print('Merge complete')
if __name__ == "__main__":
    main()
