#! /bin/bash
#SBATCH -J llama-factory
#SBATCH -o fine_tuning/logs/llama-factory_%N_%j.log
#SBATCH -e fine_tuning/logs/llama-factory_%N_%j.log
#SBATCH -N 1
#SBATCH -t 24:00:00

train_yaml_file=$1
echo "train_yaml_file: $train_yaml_file"

# USAGE: sbatch -w gpu06 -c 16 --mem=150G --gres="gpu:a100-sxm4-80gb:1" sbatch/sft_train.sh *.yaml

echo $SHELL
pwd

source ~/.profile

# switch virtual python environment
py_env_name="anaconda3-2023.07-0/envs/llama-factory"
echo -e "\nBefore 'pyenv shell ${py_env_name}'"
echo $PATH
pyenv versions
pyenv shell ${py_env_name}
echo -e "\nAfter 'pyenv shell ${py_env_name}'"
echo $PATH
pyenv versions


gpustat
nvidia-smi

# display python version and important packages
which python
python -V
which pip
pip -V
echo -e "\nImportant Packages:"
pip list | grep -E 'torch|transformers|metrics|peft|vllm|sglang|deepspeed|cuda|datasets|flash|accelerate|llama|wandb'

current_dir=$(pwd)
llama_factory_dir=/home/dcteng/work/LLM-Engines/LLaMA-Factory

cd ${llama_factory_dir}
train_yaml_file=${current_dir}/${train_yaml_file}

# start training
## get the output_dir from the yaml file, one line starting with 'output_dir:', and trim the leading and trailing spaces
output_dir=$(grep -E '^output_dir:' ${train_yaml_file} | sed -e 's/^[ \t]*output_dir:[ \t]*//' | sed -e 's/[ \r]*$//g')
echo "\noutput_dir: '${output_dir}'\n"

# assert that the number of the files or dirs in output_dir is smaller or equal to 1
if [ -d "${output_dir}" ] && [ $(ls -A ${output_dir} | wc -l) -gt 1 ]; then
    echo "Error: The output directory '${output_dir}' already exists and contains more than one file or directory."
    echo "Please (1) remove the existing files or directories before starting the training, or (2) specify a different output directory in the YAML file."
    exit 1
fi

## create the output_dir if it does not exist
mkdir -p ${output_dir}
## copy the yaml file to the output_dir
copied_yaml_file="${output_dir}/llama-factory-train.yaml"
cp ${train_yaml_file} ${copied_yaml_file}
echo "Copied ${train_yaml_file} to ${copied_yaml_file}"
echo ""
## start training
#DISABLE_VERSION_CHECK=1
FORCE_TORCHRUN=1 llamafactory-cli train ${copied_yaml_file}
