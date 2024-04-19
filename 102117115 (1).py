import sys
import pandas as pd
import numpy as np

def check_input_parameters():
    if len(sys.argv) != 5:
        print("Usage: python 102117131.py 102117131-data.csv “1,1,1,2” “+,+,-,+” 102117131-result.csv")
        sys.exit(1)

def load_data(input_file):
    try:
        data = pd.read_csv(input_file)
        return data
    except FileNotFoundError:
        print("Error: File not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty.")
        sys.exit(1)
    except pd.errors.ParserError:
        print("Error: Unable to parse the input file. Make sure it is a valid CSV file.")
        sys.exit(1)

def validate_input_data(data):
    if data.shape[1] < 3:
        print("Error: Input file must contain three or more columns.")
        sys.exit(1)

    for col in data.columns[1:]:
        if not pd.to_numeric(data[col], errors='coerce').notnull().all():
            print(f"Error: Non-numeric values found in column {col}.")
            sys.exit(1)

def preprocess_weights_impacts(weights_str, impacts_str, data):
    weights = list(map(float, weights_str.split(',')))
    impacts = impacts_str.split(',')

    if len(weights) != len(impacts) or len(weights) != (data.shape[1] - 1):
        print("Error: Number of weights, impacts, and columns must be the same.")
        sys.exit(1)

    if not all(impact in ['+', '-'] for impact in impacts):
        print("Error: Impacts must be either '+' or '-'.")
        sys.exit(1)

    return np.array(weights), np.array(impacts)

def save_result(output_file, result_data):
    try:
        result_data.to_csv(output_file, index=False)
        print(f"Result saved successfully to {output_file}")
    except Exception as e:
        print(f"Error saving result: {e}")

def topsis(input_file, weights, impacts, output_file):
    data = load_data(input_file)
    validate_input_data(data)

    weights, impacts = preprocess_weights_impacts(weights, impacts, data)

    # Normalize the data
    norm_data = data.copy()
    norm_data.iloc[:, 1:] = data.iloc[:, 1:].apply(lambda x: x / np.linalg.norm(x), axis=0)

    # Calculate the weighted normalized decision matrix
    weighted_norm_data = norm_data.iloc[:, 1:] * weights

    # Calculate ideal positive and ideal negative solutions
    ideal_positive = weighted_norm_data.max()
    ideal_negative = weighted_norm_data.min()

    # Calculate separation measures (d+ and d-)
    d_positive = np.linalg.norm(weighted_norm_data - ideal_positive, axis=1)
    d_negative = np.linalg.norm(weighted_norm_data - ideal_negative, axis=1)

    # Calculate Topsis Score
    topsis_score = d_negative / (d_positive + d_negative)

    # Rank the alternatives
    rank = np.argsort(topsis_score, kind='quicksort') + 1

    # Create the result DataFrame
    result_data = pd.concat([data, pd.Series(topsis_score, name='Topsis_Score'), pd.Series(rank, name='Rank')], axis=1)

    # Save the result
    save_result(output_file, result_data)

if __name__ == "__main__":
    check_input_parameters()
    input_file = sys.argv[1]
    weights = sys.argv[2]
    impacts = sys.argv[3]
    output_file = sys.argv[4]

    topsis(input_file, weights, impacts, output_file)
