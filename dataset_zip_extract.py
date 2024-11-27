import os
import zipfile
import argparse

def extract_zip(zip_file_path, base_directory='datasets'):
    # Get the current working directory
    current_dir = os.getcwd()

    # Define the datasets directory path
    datasets_dir = os.path.join(current_dir, base_directory)

    # Create the datasets directory if it doesn't exist
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)
        print(f"Created directory: {datasets_dir}")

    # Extract the zip file
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(datasets_dir)
            print(f"Extracted '{zip_file_path}' to '{datasets_dir}'")
    except zipfile.BadZipFile:
        print(f"Error: '{zip_file_path}' is not a valid zip file.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Extract a zip file to the datasets directory.")

    # Add the --zip_name argument
    parser.add_argument('--zip_name', type=str, required=True, help="The zip file to be extracted.")
    parser.add_argument('--destination', type=str, required=True, help="The path for the destination to be extracted.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the extract_zip function with the provided zip file name
    extract_zip(args.zip_name, args.destination)
