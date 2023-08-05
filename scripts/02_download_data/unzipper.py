import zipfile
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument("--zip_file_path", type=str, required=True)
    parser.add_argument("--unzipped_folder_path", type=str)
    args = parser.parse_args()

    with zipfile.ZipFile(args.zip_file_path, "r") as zip_file:
        zip_file.extractall(args.unzipped_folder_path)
