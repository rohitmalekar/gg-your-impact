import pandas as pd

# Load the modified data file
modified_data_file_path = 'all_donations_by_GG.csv'
data_df = pd.read_csv(modified_data_file_path)

# Check if 'AmountUSD' needs to be converted to a numeric type. This is necessary if the column is not already in a numeric format.
data_df['AmountUSD'] = pd.to_numeric(data_df['AmountUSD'], errors='coerce')

# Group by 'New Column' and sum 'AmountUSD'
grouped_data = data_df.groupby('New Column')['AmountUSD'].sum().reset_index()

# Display the grouped data
print(grouped_data)

# Optionally, save the grouped data to a new CSV file
grouped_data.to_csv('grouped_data_by_new_column.csv', index=False)
