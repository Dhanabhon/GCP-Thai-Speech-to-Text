import os
import sys
import gconfig
import argparse
from tqdm import tqdm
from google.cloud import storage
from typing import List

def main(args):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(gconfig.BUCKET_NAME)

    # Determine files to upload.
    ls: List[str] = os.listdir(args.data_dir)
    print(f"Data Dir: {args.data_dir}")

    if args.exclude_dir != "":
        before = len(ls)
        print(f"Exclusion Dir: {args.exclude_dir}")
        exclude_ls = os.listdir(args.exclude_dir)
        ls = list(set(ls) - set(exclude_ls))
        print(f"Applied Exclusion Rule: {before} -> {len(ls)}")

    print(f"Uploading files from: {args.data_dir}")
    for i in tqdm(range(len(ls))):
        filename = ls[i]
        fqn = os.path.join(args.data_dir, filename)
        blob = bucket.blob(
            filename
        )  # Create the target blob/file on the bucket.
        blob.upload_from_filename(
            fqn
        )  # Upload the local file into the target blob/file.


if __name__ == "__main__":
    # Google python API requires we set the OS environment variable.
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gconfig.KEY
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        type=str,
        help="Location of files to upload. This is a superset of `exclude_dir`.",
    )
    parser.add_argument(
        "--exclude_dir",
        type=str,
        default="",
        help="Location of folder containing exclude files.",
    )
    args = parser.parse_args()
    main(args)