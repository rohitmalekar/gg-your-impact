import pandas as pd
import os

# Specify the directory containing the CSV files
data_directory = 'data'
lookup_file_path = 'GS GG Rounds.csv'

# Read the lookup file
lookup_df = pd.read_csv(lookup_file_path)

# Convert 'Start Date' and 'End Date' in lookup_df to datetime and make them timezone-naive
lookup_df['Start Date'] = pd.to_datetime(lookup_df['Start Date']).dt.tz_localize(None)
lookup_df['End Date'] = pd.to_datetime(lookup_df['End Date']).dt.tz_localize(None)

# Iterate over all CSV files in the directory
for filename in os.listdir(data_directory):
    if filename.endswith(".csv") and filename != 'GS GG Rounds.csv':
        data_file_path = os.path.join(data_directory, filename)
        
        # Load the data file
        data_df = pd.read_csv(data_file_path, dtype={1: str})

        # Convert 'Tx Timestamp' to datetime while ensuring it's timezone-aware
        data_df['Tx Timestamp'] = pd.to_datetime(data_df['Tx Timestamp'], format='mixed').dt.tz_localize(None)

        # Define the lookup function
        def lookup(row):
            if row['Source'] in ['CGrants', 'Alpha']:
                # Find rows where 'Tx Timestamp' is within the date range
                within_range = lookup_df[(lookup_df['Start Date'] <= row['Tx Timestamp']) & (lookup_df['End Date'] >= row['Tx Timestamp'])]
                if not within_range.empty:
                    return within_range['Aggregate Name'].values[0]
                else:
                    # Find the closest past 'Start Date'
                    past_starts = lookup_df[lookup_df['Start Date'] < row['Tx Timestamp']]
                    if not past_starts.empty:
                        closest_row = past_starts.loc[(row['Tx Timestamp'] - past_starts['Start Date']).abs().idxmin()]
                        return closest_row['Aggregate Name']
            elif row['Source'] == 'GrantsStack':
                applicable_row = lookup_df[lookup_df['ID'] == row['Round Address']]
                return applicable_row['Aggregate Name'].values[0] if not applicable_row.empty else None
            return None
            
        # Apply the function to create a new 'Round' column
        data_df['Round'] = data_df.apply(lookup, axis=1)

        # Save the modified DataFrame
        modified_filename = f"modified_{filename}"
        modified_file_path = os.path.join(data_directory, modified_filename)
        data_df.to_csv(modified_file_path, index=False)

        # Group by 'Round' and sum 'AmountUSD'
        data_df['AmountUSD'] = pd.to_numeric(data_df['AmountUSD'], errors='coerce')
        summary = data_df.groupby('Round')['AmountUSD'].sum().reset_index()

        # Create a log file with the summary
        log_filename = f"log_{filename.replace('.csv', '')}.txt"
        log_file_path = os.path.join(data_directory, log_filename)
        with open(log_file_path, 'w') as log_file:
            summary.to_string(log_file, index=False)

        print(f"Processed {filename}. Results saved to {modified_file_path} and log to {log_file_path}.")
