import pandas as pd
import sys


def remove_empty_rows(input_csv, output_csv):
    try:
        # Read the CSV file
        df = pd.read_csv(input_csv)

        # Drop rows with any empty elements
        df_cleaned = df.dropna()

        # Save the cleaned dataframe to a new CSV file
        df_cleaned.to_csv(output_csv, index=False)
        print(f"Successfully removed empty rows. Cleaned file saved as: {output_csv}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
        input_csv = "data/imdb_top_1000.csv"
        output_csv = "data/imdb_top_1000.csv"
        remove_empty_rows(input_csv, output_csv)
