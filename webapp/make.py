import pandas as pd

# Open the CSV file and create a DataFrame
input_csv_file = 'plot_recsys.csv'
df = pd.read_csv(input_csv_file)

# Select specific columns
selected_columns = ['Title', 'Year', 'Poster','imdbID','similar_movies']  # Replace with the column names you want
selected_df = df[selected_columns]

# Export the selected DataFrame to a new CSV file
output_csv_file = 'recsys.csv'
selected_df.to_csv(output_csv_file, index=False)