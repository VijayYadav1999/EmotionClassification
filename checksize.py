import pandas as pd

# Provide the path to the CSV file
csv_file_path = 'trainData1_Of_Extract_Feature.csv'

# Read the CSV file
data = pd.read_csv(csv_file_path)

# Get the number of rows and columns
num_rows = data.shape[0]
num_columns = data.shape[1]

# Print the number of rows and columns
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")
