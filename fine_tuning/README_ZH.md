# 使用 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 微调 LLMs

---

## 安装
### Install from Source
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics,deepspeed,vllm]" --no-build-isolation
```

### DeepSpeed 多卡训练支持
LLaMA-Factory 提供了对 DeepSpeed 的支持和基本的配置文件，以实现高效的多卡训练。请确保已安装 DeepSpeed。

### 目录结构
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


## 数据准备
使用 LLaMA-Factory 进行训练时，需要加载本工作的自定义微调数据集。
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
自定义数据集的格式已经添加到 [dataset_info.json](../llama_factory_data/dataset_info.json)，自定义加载脚本位于 [llama_factory_data/qwen25_ESAinsTOD](../llama_factory_data/qwen25_ESAinsTOD) 目录下。

***注意：加载脚本中的 `self.data_dir` 参数需要根据微调数据集的实际存放路径进行修改。***

关于数据准备的详细说明，请参考 [数据格式](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md)。

## 训练样例
### 全量微调
以在我们的微调数据集上全量微调 [***Qwen2.5-3B-Instruct***](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) 为例：
  - **训练配置**: [fine_tuning/LLaMA-Factory/train_configs/train_full/qwen25_3B_full_sft.yaml](LLaMA-Factory/train_configs/train_full/qwen25_3B_full_sft.yaml)，部分参数需要根据您的环境进行修改，例如 `model_name_or_path` 指向预训练模型的路径，`dataset_dir` 指向数据集所在路径等。
  - **训练脚本**: [fine_tuning/LLaMA-Factory/launch_llamafactory.sh](LLaMA-Factory/launch_llamafactory.sh)，需要修改其中的环境激活命令和 LLaMA-Factory 的安装路径等以适应您的环境。
  - **训练命令**:
    ```bash
    cd ESAinsTOD
    yaml_file="fine_tuning/LLaMA-Factory/train_configs/train_full/qwen25_3B_full_sft.yaml"
    bash fine_tuning/LLaMA-Factory/launch_llamafactory.sh ${yaml_file}
    ```
    
### LoRA微调
以在我们的微调数据集上使用 LoRA 微调 [***Qwen2.5-3B-Instruct***](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) 为例：
  - **训练配置**: [fine_tuning/LLaMA-Factory/train_configs/train_lora/qwen25_3B_lora_sft.yaml](LLaMA-Factory/train_configs/train_lora/qwen25_3B_lora_sft.yaml)
  - **训练脚本**: [fine_tuning/LLaMA-Factory/launch_llamafactory.sh](LLaMA-Factory/launch_llamafactory.sh)
  - **训练模型**:
    ```bash
    cd ESAinsTOD
    yaml_file="fine_tuning/LLaMA-Factory/train_configs/train_lora/qwen25_3B_lora_sft.yaml"
    bash fine_tuning/LLaMA-Factory/launch_llamafactory.sh ${yaml_file}
    ```
  - **LoRA权重导出配置**: [fine_tuning/LLaMA-Factory/train_configs/merge_lora/qwen25_3B_lora_sft.yaml](LLaMA-Factory/train_configs/merge_lora/qwen25_3B_lora_sft.yaml)
  - **合并LoRA权重**:
    ```bash
    cd ESAinsTOD
    yaml_file="fine_tuning/LLaMA-Factory/train_configs/merge_lora/qwen25_3B_lora_sft.yaml"
    lora_model_path="saves/qwen25-3b-instruct/lora/ToD_Qwen25-3B-Instruct_lora_e2-b64-lr5e-5-cosine_mr4e-1-wd1e-1-smkcfbs-irs-mix_schema-raw_uttr-ddb"
    bash fine_tuning/LLaMA-Factory/export_lora_model.sh ${yaml_file} ${lora_model_path}
    ```