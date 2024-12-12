from pathlib import Path
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Process the cache metadata CSV file to generate a quick index.")
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file")
    args = parser.parse_args()

    csv_path = args.csv_path
    csv_file = Path(csv_path)

    df = pd.read_csv(csv_file)

    unique_file_names = df['file_name'].apply(lambda x: str(Path(x).parent)).unique()
    unique_df = pd.DataFrame(unique_file_names, columns=['file_name'])

    file_stem = csv_file.stem
    file_suffix = csv_file.suffix

    new_file_name = f"{file_stem}_quick_index{file_suffix}"
    new_path = csv_file.with_name(new_file_name)

    unique_df.to_csv(new_path, index=False)

    print(f"Processed file saved to: {new_path}")

if __name__ == "__main__":
    main()