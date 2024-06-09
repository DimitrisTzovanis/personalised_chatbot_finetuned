import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load the data from a CSV file
file_path = 'data.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Shuffle the data and split into train and validation sets
train_data, validate_data = train_test_split(data, test_size=0.2, random_state=42)

# Function to convert DataFrame to JSON format (list of lists)
def convert_to_json_format(df):
    return df.apply(lambda row: [row['user'], row['response']], axis=1).tolist()

# Convert both subsets to the desired JSON format
json_train_data = convert_to_json_format(train_data)
json_validate_data = convert_to_json_format(validate_data)

# Save the train data to a JSON file
with open('train.json', 'w', encoding='utf-8') as f:
    json.dump(json_train_data, f, ensure_ascii=False, indent=2)

# Save the validate data to a JSON file
with open('validate.json', 'w', encoding='utf-8') as f:
    json.dump(json_validate_data, f, ensure_ascii=False, indent=2)

print("Data has been successfully saved to train.json and validate.json")
