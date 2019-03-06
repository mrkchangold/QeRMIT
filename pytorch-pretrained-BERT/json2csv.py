import json
import os
import argparse
import csv

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--json_file", default=None, type=str, help="predictions jsonfile location (output of run_squad). E.g., train-v1.1.json")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--OG", action='store_true', help="test")

    args = parser.parse_args()

    with open(args.json_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Write csv submission file 
    sub_path = os.path.join(args.output_dir, 'dev_submission.csv')
    with open(sub_path, 'w') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for qas_id in sorted(input_data):
            # TODO: determine if else this needed (if input_data[qas_id] != 'empty':)
            if not args.OG:
                csv_writer.writerow([qas_id, input_data[qas_id]])
            else:
                if input_data[qas_id] == "empty":
                    csv_writer.writerow([qas_id, ""])
                else:
                    csv_writer.writerow([qas_id, input_data[qas_id]])

            
                

            

if __name__ == "__main__":
    main()