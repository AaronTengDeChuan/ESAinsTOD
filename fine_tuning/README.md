# Fine-tuning LLMs with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

---

## Installation
### Install from Source
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics,deepspeed,vllm]" --no-build-isolation
```

### DeepSpeed Multi-GPU Training Support
LLaMA-Factory provides support for DeepSpeed and basic configuration files to enable efficient multi-GPU training. Please ensure DeepSpeed is installed.

### Directory Structure
```
    📂 LLaMA-Factory/
    ├── 📂 examples/
    │   ├── 📂 deepspeed/
    │   │   ├── 📄 ds_z0_config.json
    │   │   └── 📄 ds_z2_config.json
    │   │   └── 📄 ds_z3_config.json
    │   ├── 📂 train_full/
    │   │   ├── 🔡 llama3_full_sft.yaml
    │   │   └── ...
    │   ├── 📂 train_lora/
    │   │   ├── 🔡 llama3_lora_sft.yaml
    │   │   └── ...
    │   ├── 📂 merge_lora/
    │   │   ├── 🔡 llama3_lora_sft.yaml
    │   │   └── ...
    │   └── ...
    └── ... 
   ```


## Data Preparation
When training with LLaMA-Factory, you need to load the custom fine-tuning dataset for this work.
```json
{
    "qwen25_smkcfbs_irs_mix_schema_raw_uttr_ddb_train": {
        "script_url": "qwen25_ESAinsTOD",
        "subset": "smkcfbs_irs_mix_schema_raw_uttr_ddb",
        "split": "train",
        "formatting": "sharegpt"
    },
    "qwen25_smkcfbs_irs_mix_schema_raw_uttr_ddb_dev": {
        "script_url": "qwen25_ESAinsTOD",
        "subset": "smkcfbs_irs_mix_schema_raw_uttr_ddb",
        "split": "validation",
        "formatting": "sharegpt",
        "num_samples": 1000
    }
}
```
The format of the custom dataset has been added to [dataset_info.json](../llama_factory_data/dataset_info.json), and the custom loading script is located in the [llama_factory_data/qwen25_ESAinsTOD](../llama_factory_data/qwen25_ESAinsTOD) directory.

***Note: The `self.data_dir` parameter in the loading script needs to be modified according to the actual storage path of the fine-tuning dataset.***

For detailed instructions on data preparation, please refer to [Dataset Format](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md).

## Training Examples
### Full Fine-tuning (SFT)
Example of full fine-tuning [***Qwen2.5-3B-Instruct***](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) on our fine-tuning dataset:
  - **Training Configuration**: [fine_tuning/LLaMA-Factory/train_configs/train_full/qwen25_3B_full_sft.yaml](LLaMA-Factory/train_configs/train_full/qwen25_3B_full_sft.yaml). Some parameters need to be modified according to your environment, such as `model_name_or_path` pointing to the path of the pre-trained model, and `dataset_dir` pointing to the dataset path.
  - **Training Script**: [fine_tuning/LLaMA-Factory/launch_llamafactory.sh](LLaMA-Factory/launch_llamafactory.sh). You need to modify the environment activation command and the installation path of LLaMA-Factory within the script to suit your environment.
  - **Training Command**:
    ```bash
    cd ESAinsTOD
    yaml_file="fine_tuning/LLaMA-Factory/train_configs/train_full/qwen25_3B_full_sft.yaml"
    bash fine_tuning/LLaMA-Factory/launch_llamafactory.sh ${yaml_file}
    ```
    
### LoRA Fine-tuning
Example of using LoRA to fine-tune [***Qwen2.5-3B-Instruct***](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) on our fine-tuning dataset:
  - **Training Configuration**: [fine_tuning/LLaMA-Factory/train_configs/train_lora/qwen25_3B_lora_sft.yaml](LLaMA-Factory/train_configs/train_lora/qwen25_3B_lora_sft.yaml)
  - **Training Script**: [fine_tuning/LLaMA-Factory/launch_llamafactory.sh](LLaMA-Factory/launch_llamafactory.sh)
  - **Run Training**:
    ```bash
    cd ESAinsTOD
    yaml_file="fine_tuning/LLaMA-Factory/train_configs/train_lora/qwen25_3B_lora_sft.yaml"
    bash fine_tuning/LLaMA-Factory/launch_llamafactory.sh ${yaml_file}
    ```
  - **LoRA Weight Export Configuration**: [fine_tuning/LLaMA-Factory/train_configs/merge_lora/qwen25_3B_lora_sft.yaml](LLaMA-Factory/train_configs/merge_lora/qwen25_3B_lora_sft.yaml)
  - **Merge LoRA Weights**:
    ```bash
    cd ESAinsTOD
    yaml_file="fine_tuning/LLaMA-Factory/train_configs/merge_lora/qwen25_3B_lora_sft.yaml"
    lora_model_path="saves/qwen25-3b-instruct/lora/ToD_Qwen25-3B-Instruct_lora_e2-b64-lr5e-5-cosine_mr4e-1-wd1e-1-smkcfbs-irs-mix_schema-raw_uttr-ddb"
    bash fine_tuning/LLaMA-Factory/export_lora_model.sh ${yaml_file} ${lora_model_path}
    ```