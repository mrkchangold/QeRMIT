********************************useful commands***************************************
scp -r  boss@40.80.149.205:/tmp/debug_squad/ /home/mrkchang/Documents/Stanford/CS224N/fromVM

git config --global credential.helper cache
git config --global credential.helper 'cache --timeout=7200'

nvidia-smi
tmux new -s [name]
ctrl - b - [
tmux list-sessions
^b = cntrl-b
cntrl - b - d
$ tmux a -t [name]
$ tmux a -t #

sudo prime-select intel
sudo prime-select nvidia
prime-select query

rm = remove CAREFUL!!
mv = move/rename
***********************************************************************
python train.py   --bert_model bert-base-uncased   --do_train   --do_predict   --do_lower_case   --train_file ./data/train-v2.0.json   --predict_file ./data/dev-v2.0.json   --train_batch_size 12   --learning_rate 3e-5   --num_train_epochs 2.0   --max_seq_length 384   --doc_stride 128   --output_dir ./tmp/debug_squad/ --version_2_with_negative



python setup.py install --cuda_ext --cpp_ext


data_path (str): Path to .npz file containing pre-processed dataset.
dataset = np.load(data_path)
self.ids = torch.from_numpy(dataset['ids']).long()

python test_bidaf.py --split dev --load_path save/train/baseline-01/step_1951227.pth.tar  --name dbging

python run_squad.py   --bert_model bert-base-uncased   --do_train   --do_predict   --do_lower_case   --train_file ./data/train-v2.0.json   --predict_file ./data/dev-v2.0.json   --train_batch_size 24   --learning_rate 3e-5   --num_train_epochs 2.0   --max_seq_length 384   --doc_stride 128   --output_dir /tmp/debug_squad_redo_from_run_squad/ --version_2_with_negative --fp16 --loss_scale 128


#remember
scp -r /home/mrkchang/Documents/Stanford/CS224N/cs224n-project/data  boss@40.80.149.205:~/hugface/hugface/pytorch-pretrained-BERT/


export squad_dir=~/hugface/hugface/pytorch-pretrained-BERT/data

#normal <- I think this ran out of memory
python run_squad.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $squad_dir/train-v2.0.json \
  --predict_file $squad_dir/dev-v2.0.json \
  --train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/

# below is for using fp16
# more changes to accommodate smaller memory
python run_squad.py \
  --bert_model bert-large-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $squad_dir/train-v2.0.json \
  --predict_file $squad_dir/dev-v2.0.json \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/ \
  --train_batch_size 8 \
  --fp16 \
  --loss_scale 128 \
  --version_2_with_negative \
  --gradient_accumulation_steps 4




# for predicting on dev set given a certain model dir
export squad_dir=~/hugface/hugface/pytorch-pretrained-BERT/data
export model_dir=~/tmp/debug_squad/

python run_squad.py \
  --bert_model $model_dir \
  --do_predict \
  --do_lower_case \
  --predict_file $squad_dir/dev-v2.0.json \
  --doc_stride 128 \
  --output_dir $model_dir/output/ \
  --fp16 \
  --loss_scale 128 \
  --version_2_with_negative \
  --gradient_accumulation_steps 4

# for iterating on model test
export squad_dir=~/hugface/hugface/pytorch-pretrained-BERT/data
export model_dir=~/tmp/debug_squad/

python run_squad.py \
  --bert_model $model_dir \
  --do_predict \
  --do_lower_case \
  --predict_file $squad_dir/dev-v2.0.json \
  --doc_stride 128 \
  --output_dir $model_dir/output/ \
  --fp16 \
  --loss_scale 128 \
  --version_2_with_negative \
  --gradient_accumulation_steps 4 \
  --freeze_BERT_embed

# for iterating on training the model...
python run_squad.py \
  --bert_model $model_dir \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $squad_dir/train-v2.0.json \
  --predict_file $squad_dir/dev-v2.0.json \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $model_dir/output/ \
  --train_batch_size 8 \
  --fp16 \
  --loss_scale 128 \
  --version_2_with_negative \
  --gradient_accumulation_steps 4
  --freeze_BERT_embed



*****************************CUSTOM scripts***********************************
0. Used to create baseline model for single layer
python run_squad.py   --bert_model bert-large-uncased   --do_train   --do_predict   --do_lower_case   --train_file $squad_dir/train-v2.0.json   --predict_file $squad_dir/dev-v2.0.json   --learning_rate 3e-5   --num_train_epochs 2   --max_seq_length 384   --doc_stride 128   --output_dir /tmp/debug_squad/   --train_batch_size 8   --fp16   --loss_scale 128   --version_2_with_negative   --gradient_accumulation_steps 4 --freeze_BERT_embed


0.1. Download file to local
scp -r  boss@40.80.149.205:/tmp/debug_squad/ /home/mrkchang/Documents/Stanford/CS224N/fromVM

0.2 Rename folder and move it in VM
export latest_model_dir=~/hugface/debug_squad_baseline_03032019

1. To create smaller dataset
export data_dir=/home/mrkchang/Documents/Stanford/CS224N/hugface/pytorch-pretrained-BERT/data
python data_manipulator.py  --json_file $data_dir/dev-v2.0.json   --output_dir $data_dir --create_dbg --threshold 0.1 --output_file dev_small.json

python data_manipulator.py  --json_file $data_dir/train-v2.0.json   --output_dir $data_dir --create_dbg --threshold 0.1 --output_file train_small.json

2. To create csv from json
export output_dir=/home/mrkchang/Documents/Stanford/CS224N/hugface/pytorch-pretrained-BERT/tmp/debug_squad_baseline_03032019

python json2csv.py --json_file $output_dir/predictions.json --output_dir $output_dir

Submitted:
EM: 77.526 (+77.526)
F1: 80.407 (+80.407)

optional. To dbg locally
export squad_dir=./data
export dbg_dir=./tmp/debug
python run_squad.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $squad_dir/dbg_sampled.json \
  --predict_file $squad_dir/dbg_sampled.json \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $dbg_dir \
  --train_batch_size 8 \
  --fp16 \
  --loss_scale 128 \
  --version_2_with_negative \
  --gradient_accumulation_steps 4 \
  --freeze_BERT_embed
 
3. For debugging on new model...
tmux a -t returnOfTheKing
export squad_dir=~/hugface/hugface/pytorch-pretrained-BERT/data
export model_dir=~/hugface/debug_squad_baseline_03032019
export output_dir=~/hugface/debug_squad_baseline_03032019_cnn
python run_squad.py \
  --bert_model $model_dir \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $squad_dir/train-v2.0.json \
  --predict_file $squad_dir/dev-v2.0.json \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $output_dir \
  --train_batch_size 8 \
  --fp16 \
  --loss_scale 128 \
  --version_2_with_negative \
  --gradient_accumulation_steps 4 \
  --freeze_BERT_embed \
  --new_model \
  --cache_example 

4. After it's ready more get rid of --new_model --cache_example to save time <- preprocessing data
# TRAINING again
export squad_dir=~/hugface/hugface/pytorch-pretrained-BERT/data
export model_dir=~/hugface/debug_squad_baseline_03032019
export output_dir=~/hugface/debug_squad_baseline_03032019_cnn
python run_squad.py   --bert_model $model_dir   --do_train   --do_predict   --do_lower_case   --train_file $squad_dir/train-v2.0.json   --predict_file $squad_dir/dev-v2.0.json   --learning_rate 3e-5   --num_train_epochs 2   --max_seq_length 384   --doc_stride 128   --output_dir $output_dir   --train_batch_size 4   --fp16   --loss_scale 128   --version_2_with_negative   --gradient_accumulation_steps 4   --freeze_BERT_embed

5. WITH V100
5.1 download model to dir and github repo
scp -r  /home/mrkchang/Documents/Stanford/CS224N/fromVM/debug_squad_baseline_03032019 boss@13.77.171.174:~/models_squad

5.2 run to train


export squad_dir=~/hugface/pytorch-pretrained-BERT/data
export model_dir=~/models_squad/debug_squad_baseline_03032019
export output_dir=~/models_squad/debug_squad_baseline_03032019_cnn
python run_squad.py   --bert_model $model_dir   --do_train   --do_predict   --do_lower_case   --train_file $squad_dir/train-v2.0.json   --predict_file $squad_dir/dev-v2.0.json   --learning_rate 3e-5   --num_train_epochs 2   --max_seq_length 384   --doc_stride 128   --output_dir $output_dir   --train_batch_size 12 --version_2_with_negative --freeze_BERT_embed

export squad_dir=~/hugface/pytorch-pretrained-BERT/data
export model_dir=~/models_squad/debug_squad_baseline_03032019
export output_dir=~/models_squad/debug_squad_baseline_03032019_model2
python ./run_squad.py \
  --bert_model $model_dir \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $squad_dir/train-v2.0.json \
  --predict_file $squad_dir/dev-v2.0.json \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $output_dir \
  --train_batch_size 4 \
  --fp16 \
  --loss_scale 128 \
  --version_2_with_negative \
  --freeze_BERT_embed \
  --model2&> train_output.txt


  --new_model \
  --cache_example \ 

5.3 NV24 for model 2 with MAXPOOL
export output_dir=~/models_squad/debug_squad_baseline_03032019_model2
python ./run_squad.py   --bert_model $model_dir   --do_train   --do_predict   --do_lower_case   --train_file $squad_dir/train-v2.0.json   --predict_file $squad_dir/dev-v2.0.json   --learning_rate 3e-5   --num_train_epochs 2   --max_seq_length 384   --doc_stride 128   --output_dir $output_dir   --train_batch_size 4   --fp16   --loss_scale 128   --version_2_with_negative   --freeze_BERT_embed --model2
# 10 hour run time/iteration


5.4 NV24 for model 2 with OG, freeze layers works, batch size 4096?
export output_dir=~/models_squad/debug_squad_baseline_03032019_OG
python ./run_squad.py   --bert_model bert-large-uncased   --do_train   --do_predict   --do_lower_case   --train_file $squad_dir/train-v2.0.json   --predict_file $squad_dir/dev-v2.0.json   --learning_rate 3e-5   --num_train_epochs 2   --max_seq_length 384   --doc_stride 128   --output_dir $output_dir   --train_batch_size 4   --fp16   --loss_scale 128   --version_2_with_negative   --freeze_BERT_embed --OG

5.5 flags so far.... --OG, --freeze_BERT_embed --model2 --model3 --new_model (uses hardcoded tokenizer) --cache_example
5.6 CNN 
export squad_dir=~/hugface/hugface/pytorch-pretrained-BERT/data
export model_dir=~/hugface/debug_squad_baseline_03032019
export output_dir=~/hugface/debug_squad_baseline_03032019_cnn
python ./run_squad.py   --bert_model $model_dir   --do_train   --do_predict   --do_lower_case   --train_file $squad_dir/train-v2.0.json   --predict_file $squad_dir/dev-v2.0.json   --learning_rate 3e-5   --num_train_epochs 2   --max_seq_length 384   --doc_stride 128   --output_dir $output_dir   --train_batch_size 4   --fp16   --loss_scale 128   --version_2_with_negative   --freeze_BERT_embed 

5.6b maxpool model 2
export squad_dir=~/hugface/pytorch-pretrained-BERT/data
export model_dir=~/models_squad/debug_squad_baseline_03032019
export output_dir=~/models_squad/debug_squad_baseline_03032019_maxpool
python ./run_squad.py   --bert_model $model_dir   --do_train   --do_predict   --do_lower_case   --train_file $squad_dir/train-v2.0.json   --predict_file $squad_dir/dev-v2.0.json   --learning_rate 3e-5   --num_train_epochs 2   --max_seq_length 384   --doc_stride 128   --output_dir $output_dir   --train_batch_size 256   --fp16   --loss_scale 128   --version_2_with_negative   --freeze_BERT_embed --model2 --new_model --cache_example

5.6 CLS model 3
export squad_dir=~/hugface/pytorch-pretrained-BERT/data
export model_dir=~/models_squad/debug_squad_baseline_03032019
export output_dir=~/models_squad/debug_squad_baseline_03032019_cls
python ./run_squad.py   --bert_model $model_dir   --do_train   --do_predict   --do_lower_case   --train_file $squad_dir/train-v2.0.json   --predict_file $squad_dir/dev-v2.0.json   --learning_rate 3e-5   --num_train_epochs 2   --max_seq_length 384   --doc_stride 128   --output_dir $output_dir   --train_batch_size 256   --fp16   --loss_scale 128   --version_2_with_negative   --freeze_BERT_embed --model3 --new_model --cache_example

5.6d baseline (just last layerm batch size 1024?)

5.7 Trained models
-BiDAF (56.965 EM, 60.21 F1)
-BERT-fine tune all layers (77.526 EM, 80.407 F1)
-BERT freeze -> last qa outputs layer only (18.921 EM, 21.113 F1)
-BERT freeze -> CNN layer + last layer (77.493 EM, 80.359 F1)
-BERT freeze -> max pool + last layer (76.505 EM, 79.601 F1)
-BERT freeze -> CLS + last layer

5.8 Future work
-finetune threshold (bonus)
-what percent of question + paragraph have > 384?
-implement tensorchart for bayesian hyperparameter search
-character embedding
-data augmentation (Future suggestions) - we need more hardware

ceback (most recent call last):
  File "./run_squad.py", line 1210, in <module>
    main()
  File "./run_squad.py", line 1146, in main
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_segment_ids_flipped, all_query_length, all_example_index) # added_flag all_segment_ids_flipped, all_query_length
  File "/data/anaconda/envs/squad/lib/python3.6/site-packages/torch/utils/data/dataset.py", line 36, in __init__
    assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
AssertionError
(squad) boss@cs224n-nv24-2:/data/home/boss/hugface/pytorch-pretrained-BERT$ 

6. Trying to get predict to work with new models..
cd ~/hugface/pytorch-pretrained-BERT
export squad_dir=~/hugface/pytorch-pretrained-BERT/data
export model_dir=~/models_squad/debug_squad_baseline_03032019_maxpool
export output_dir=~/models_squad/debug_squad_baseline_03032019_maxpool
python ./run_squad.py   --bert_model $model_dir   --do_predict   --do_lower_case  --predict_file $squad_dir/dev-v2.0.json   --learning_rate 3e-5   --num_train_epochs 2   --max_seq_length 384   --doc_stride 128   --output_dir $output_dir   --train_batch_size 4   --fp16   --loss_scale 128   --version_2_with_negative   --freeze_BERT_embed --new_model
