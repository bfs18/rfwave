#!/bin/bash

usage() {
    echo "Usage: $0 pretrained_dir config generate_dir log_dir"
    echo "  pretrained_dir   Path to the directory containing pretrained model"
    echo "  config           Configuration file"
    echo "  generate_dir     Directory to save generated files"
    echo "  log_dir          Directory to save log files"
    exit 1
}


split_file_list() {
    local generate_dir=$1
    local temp_file_list=$(mktemp)

    ls "$generate_dir" > "$temp_file_list"

    head -n 1000 "$temp_file_list" > "$generate_dir/filelist.valid"
    tail -n +1001 "$temp_file_list" > "$generate_dir/filelist.train"

    rm "$temp_file_list"
}


if [ $# -ne 4 ]; then
    echo "Error: Missing required arguments."
    usage
fi

pretrained_dir=$1
config=$2
generate_dir=$3
log_dir=$4
num_pairs=10000


if [ ! -d "$pretrained_dir" ]; then
    echo "Error: Pretrained directory '$pretrained_dir' does not exist."
    exit 1
fi

if [ ! -f "$config" ]; then
    echo "Error: Configuration file '$config' does not exist."
    exit 1
fi

mkdir -p "$generate_dir"
mkdir -p "$log_dir"

if [ ! -d "$generate_dir" ]; then
    echo "Error: Failed to create generate directory '$generate_dir'."
    exit 1
fi

if [ ! -d "$log_dir" ]; then
    echo "Error: Failed to create log directory '$log_dir'."
    exit 1
fi

last_ckpt_path=$(find "$pretrained_dir" -name "last.ckpt" -print -quit)
if [ -z "$last_ckpt_path" ]; then
    echo "Error: 'last.ckpt' not found in '$pretrained_dir' or its subdirectories."
    exit 1
else
    echo "Found 'last.ckpt' at: $last_ckpt_path"
fi

rf1_generate_dir="${generate_dir}/rf1"
mkdir -p "$rf1_generate_dir"

rf2_generate_dir="${generate_dir}/rf2"
mkdir -p "$rf2_generate_dir"

rf2_log_dir="${log_dir}/rf2"
mkdir -p "$rf2_log_dir"

rfd_log_dir="${log_dir}/rfd"
mkdir -p "$rfd_log_dir"

export PYTHONPATH=$(pwd):$PYTHONPATH

python3 reflow/generate_data.py --model_dir "$pretrained_dir" \
    --save_dir "$rf1_generate_dir" \
    --num_pairs $num_pairs
echo "generate rf1 data successfully."

split_file_list "$rf1_generate_dir"

python3 train.py --config "$config" \
  --model.init_args.pretrained_ckpt_path "$last_ckpt_path" \
  --trainer.logger.init_args.save_dir "$rf2_log_dir" \
  --data.init_args.train_filelist "$rf1_generate_dir/filelist.train" \
  --data.init_args.val_filelist "$rf1_generate_dir/filelist.valid"

python3 reflow/generate_data.py --model_dir "$rf2_log_dir" \
    --save_dir "$rf2_generate_dir" \
    --num_pairs $num_pairs
echo "generate rf2 data successfully."

split_file_list "$rf2_generate_dir"

rf2_last_ckpt_path=$(find "$rf2_log_dir" -name "last.ckpt" -print -quit)
python3 train.py --config "$config" \
  --model.init_args.one_step "true" \
  --model.init_args.pretrained_ckpt_path "$rf2_last_ckpt_path" \
  --trainer.logger.init_args.save_dir "$rfd_log_dir" \
  --data.init_args.train_filelist "$rf2_generate_dir/filelist.train" \
  --data.init_args.val_filelist "$rf2_generate_dir/filelist.valid"

