import argparse
import os
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    args = parser.parse_args()
    dataset_path = args.dataset_path

    # Create the directory path
    postprocessed_dir = dataset_path + "_postprocessed"
    os.makedirs(postprocessed_dir, exist_ok=True)

    # Loop through all files in the dataset
    for filename in glob.glob(os.path.join(dataset_path, '*.jsonl')):
        
        postprocessed_file = os.path.join(postprocessed_dir, filename)