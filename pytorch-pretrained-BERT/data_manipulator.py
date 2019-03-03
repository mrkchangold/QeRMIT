import json
import os
import argparse
import csv
import numpy as np

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--json_file", default=None, type=str, required=True, help="predictions jsonfile location (output of run_squad). E.g., train-v1.1.json")
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--create_dbg", action='store_true', help="Whether to run training.")
    parser.add_argument("--augment", action='store_true', help="Whether to run training.")

    args = parser.parse_args()

    if args.create_dbg:
    # just a function to create a smaller set of data set examples
        with open(args.json_file, "r", encoding='utf-8') as reader:
            input_dict = json.load(reader)
        
        output = {}
        threshold = 0.1

        input_data = input_dict["data"]
        input_ver = input_dict["version"]


        output_data = []
        while not output_data: #False = (dict empty)
            for entry in input_data:
                if np.random.rand() < threshold:
                    output_data.append(entry)
        
        output["version"] = input_ver
        output["data"] = output_data

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        sub_path = os.path.join(args.output_dir, 'dbg_sampled.json')
        with open(sub_path, "w") as writer:
            writer.write(json.dumps(output))
            

if __name__ == "__main__":
    main()