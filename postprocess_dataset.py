import argparse
import os
import glob
import jsonlines
from tqdm import tqdm
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    args = parser.parse_args()
    dataset_path = args.dataset_path
    
    data = {"prompt": [], "completion": []}
    for filename in glob.glob(os.path.join(dataset_path, '*.jsonl')):
        with jsonlines.open(filename) as reader:
            for row in tqdm(reader):
                data["prompt"].append(row["prompt"])
                data["completion"].append(row["completion"])
                
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(dataset_path, 'data.csv'), index=False)