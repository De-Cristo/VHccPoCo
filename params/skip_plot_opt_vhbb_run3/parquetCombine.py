import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def process_subdir(subdir, root_dir, output_dir):
    """
    Process a single subdirectory: combine all Parquet files and save a combined file.
    
    Parameters:
    - subdir (str): Path to the subdirectory.
    - root_dir (str): Root directory for calculating relative paths.
    - output_dir (str): Directory where the combined Parquet file will be saved.
    """
    # Get all Parquet files in the subdirectory
    parquet_files = [os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith('.parquet')]

    if not parquet_files:
        return f"No Parquet files in {subdir}, skipping."

    # Create a name for the output file reflecting the full subdirectory path
    relative_path = os.path.relpath(subdir, root_dir)
    sanitized_path = relative_path.replace(os.sep, "_")  # Replace directory separators with underscores
    output_file = os.path.join(output_dir, f"{sanitized_path}_combined.parquet")

    print(f"Processing {len(parquet_files)} Parquet files in directory: {subdir}")
    dataframes = []
    for file in parquet_files:
        print(f"  Reading {file}...")
        df = pd.read_parquet(file)
        dataframes.append(df)

    # Combine all DataFrames in the subdirectory
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Save the combined DataFrame to a Parquet file
    print(f"  Saving combined Parquet file to {output_file}...")
    combined_df.to_parquet(output_file, index=False)

    return f"Combined Parquet saved for {subdir} as {output_file}"

def combine_parquet_files_in_nested_dirs_multithreaded(root_dir, output_dir, max_workers=4):
    """
    Traverse a nested directory structure, combine Parquet files in each subdirectory,
    and save one combined Parquet file for each subdirectory using multithreading.

    Parameters:
    - root_dir (str): Root directory containing the nested directories.
    - output_dir (str): Directory where the combined Parquet files will be saved.
    - max_workers (int): Maximum number of threads to use for processing.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Collect all subdirectories
    subdirs = [os.path.join(root, d) for root, dirs, _ in os.walk(root_dir) for d in dirs]

    # Use ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_subdir, subdir, root_dir, output_dir) for subdir in subdirs]

        # Wait for all threads to complete and print their results
        for future in futures:
            print(future.result())

    print(f"All Parquet files have been processed and saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    root_directory = "/eos/user/l/lichengz/Column_output_vhbb_zll_sig_1116"  # Replace with the root directory path
    output_directory = "/eos/user/l/lichengz/Column_output_vhbb_zll_1115"  # Replace with the output directory path
    combine_parquet_files_in_nested_dirs_multithreaded(root_directory, output_directory)
