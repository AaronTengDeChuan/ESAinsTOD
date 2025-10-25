#! /bin/bash

source ~/.profile

py_env_name="anaconda3-2023.07-0/envs/llama-factory"
echo -e "\nBefore 'pyenv shell ${py_env_name}'"
pyenv versions
pyenv shell ${py_env_name}
echo -e "\nAfter 'pyenv shell ${py_env_name}'"
pyenv versions

merge_yaml_file=$1
echo "merge_yaml_file: $merge_yaml_file"

current_dir=$(pwd)
llama_factory_dir=/home/dcteng/work/LLM-Engines/LLaMA-Factory

cd ${llama_factory_dir}
merge_yaml_file=${current_dir}/${merge_yaml_file}

saved_model_dir=$2
# Check if the saved model directory exists
if [ ! -d "${saved_model_dir}" ]; then
    echo "Directory ${saved_model_dir} does not exist. Please check the path."
    exit 1
fi

# Check if the merge YAML file exists
if [ ! -f "${merge_yaml_file}" ]; then
    echo "Merge YAML file ${merge_yaml_file} does not exist. Please check the path."
    exit 1
fi

# find all checkpoint directories starting with "checkpoint-"
checkpoint_dirs=$(find ${saved_model_dir} -type d -name "checkpoint-*" | sort)
# Check if any checkpoint directories were found
if [ -z "${checkpoint_dirs}" ]; then
    echo "No checkpoint directories found in ${saved_model_dir}. Please check the path."
    exit 1
fi

# Loop through each checkpoint directory and export the model
for checkpoint_dir in ${checkpoint_dirs}; do
    # replace saves with output to get the export directory
    export_dir=$(echo ${checkpoint_dir} | sed 's/saves/output/g')
    echo "Exporting model from ${checkpoint_dir} to ${export_dir}"
    # record running time
    time llamafactory-cli export ${merge_yaml_file} \
        adapter_name_or_path="${checkpoint_dir}" \
        export_dir="${export_dir}"
    echo "Model exported from ${checkpoint_dir} to ${export_dir}"
    echo "----------------------------------------"
#    exit 0
done


