import os
import json
import datasets


# 1. (可選但推薦) 為您的數據集定義一個清晰的 Config 類
# 這個類繼承自 datasets.BuilderConfig，並添加您需要的自定義屬性
class CustomDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for our custom dataset."""

    def __init__(self, file_path_template, **kwargs):
        """
        Args:
          data_file: `str`, the path to the data file for this subset.
          **kwargs: keyword arguments forwarded to super.
        """
        super(CustomDatasetConfig, self).__init__(**kwargs)
        self.data_dir = "/home/dcteng/work/Dialogue/ToD/InstructToD/sft_data/qwen25"
        print(f"Data directory of Qwen25 ESAinsTOD: {self.data_dir}")
        self.file_path_template = file_path_template

    def get_split_file(self, split):
        """
        Returns the file path for the given split.
        This is useful if you have different files for train/validation/test.
        """
        split_mapping = {
            datasets.Split.TRAIN: "train",
            datasets.Split.VALIDATION: "dev",
            datasets.Split.TEST: "test",
        }
        return os.path.join(self.data_dir, self.file_path_template.format(split=split_mapping[split]))


# 2. 繼承 GeneratorBasedBuilder 來創建您的主類
class CustomMultiturn(datasets.GeneratorBasedBuilder):
    """A custom dataset for multiturn dialogues with multiple subsets."""

    # 3. 定義 BUILDER_CONFIGS，這是您提供給用戶的子集“菜單”
    BUILDER_CONFIGS = [
        CustomDatasetConfig(
            name="smkcfbs_irs_mix_schema_raw_uttr_ddb",
            version=datasets.Version("1.0.0"),
            description="Stories from source Alpha",
            file_path_template="merged_7-irs-mix_schema_{split}-raw_uttr-ddb.json",
        ),
        CustomDatasetConfig(
            name="smkcfbs_irs_raw_uttr_no_sa_ddb",
            version=datasets.Version("1.0.0"),
            description="Dialogues from source Beta",
            file_path_template="merged_7-irs-mix_schema_{split}-raw_uttr-no_sa-ddb.json",
        ),
        CustomDatasetConfig(
            name="smkcfbs_irs_raw_uttr_no_ia_ddb",
            version=datasets.Version("1.0.0"),
            description="Dialogues from source Beta",
            file_path_template="merged_7-irs-mix_schema_{split}-raw_uttr-no_ia-ddb.json",
        ),
        CustomDatasetConfig(
            name="smkcfbs_irs_raw_uttr_no_ia_no_sa_ddb",
            version=datasets.Version("1.0.0"),
            description="Dialogues from source Beta",
            file_path_template="merged_7-irs-mix_schema_{split}-raw_uttr-no_ia-no_sa-ddb.json",
        ),

        CustomDatasetConfig(
            name="smkcfbs_raw_uttr_pptod_ddb",
            version=datasets.Version("1.0.0"),
            description="Dialogues from source Beta",
            file_path_template="merged_7_{split}-pptod-raw_uttr-ddb.json",
        ),

        CustomDatasetConfig(
            name="skfbs_irs_mix_schema_raw_uttr_ddb",
            version=datasets.Version("1.0.0"),
            description="Stories from source Alpha",
            file_path_template="merged_5-irs-mix_schema-no_woz_{split}-raw_uttr-ddb.json",
        ),
    ]

    def _info(self):
        # 定義數據集的欄位結構
        return datasets.DatasetInfo(
            description="A custom multiturn dataset with multiple configurations.",
            features=datasets.Features({
                "id": datasets.Value("string"),
                "source": datasets.Value("string"),
                "conversations": [{"from": datasets.Value("string"), "value": datasets.Value("string")}]
            }),
            builder_name="Qwen25_ESAinsTOD",
        )

    def _split_generators(self, dl_manager):
        """
        這個方法是核心！它會根據用戶選擇的 config 來下載/定位數據文件。
        """
        # dl_manager 可以用來下載遠程文件，對於本地文件，我們直接拼接路徑。
        # self.config 會自動被設置為用戶在 load_dataset 中用 name 參數選擇的那個 config。
        # 例如，如果用戶調用 load_dataset(..., name='alpha')，
        # self.config 就是上面 BUILDER_CONFIGS 列表中的第一個實例。

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # gen_kwargs 將會被傳遞給 _generate_examples 方法
                gen_kwargs={"filepath": self.config.get_split_file(datasets.Split.TRAIN)},
            ),
            # 如果有其他分割（如驗證集或測試集），可以在這裡添加更多的 SplitGenerator
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": self.config.get_split_file(datasets.Split.VALIDATION)},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": self.config.get_split_file(datasets.Split.TEST)},
            ),
        ]

    @staticmethod
    def _replace_special_tokens(text):
        """
        替換文本中的特殊標記。
        這裡可以根據需要添加更多的替換規則。
        """
        replacements = {
            "|>AI assistant": "|>assistant",
            "<|beginofutterance|>": "<|im_start|>",
            "<|endofutterance|>": "<|im_end|>",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _generate_examples(self, filepath):
        """
        這個方法會從 `_split_generators` 傳遞過來的 filepath (json: List[Dict])中讀取數據並生成樣本。
        """
        print(f"Generating examples from {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, item in enumerate(data):
            # 假設每個 item 都有 'id', 'source', 'conversations' 等欄位
            conversations = []

            # assert len(item["conversations"][0]["turn_texts"]) == 1, "The first conversation turn must have exactly one system text."
            conv_idx = 0
            if len(item["conversations"][0]["turn_texts"]) == 1:
                conversations.append({
                    "from": "system",
                    "value": self._replace_special_tokens(item["conversations"][0]["turn_texts"][0])
                })
                conv_idx += 1

            no_loss_text = ""
            for turn in item["conversations"][conv_idx:]:
                texts, loss_labels = turn["turn_texts"], turn["turn_labels"]
                for text, label in zip(texts, loss_labels):
                    if label and no_loss_text:
                        conversations.extend([
                            {"from": "human", "value": no_loss_text},
                            {"from": "gpt", "value": self._replace_special_tokens(text)},
                        ])
                        no_loss_text = ""
                    elif label:
                        assert conversations[-1]["from"] == "gpt", \
                            ("The last conversation turn must be from the assistant "
                             "when a loss label is present and no_loss_text is empty.")
                        conversations[-1]["value"] += self._replace_special_tokens(text)
                    else:
                        no_loss_text += self._replace_special_tokens(text)

            yield idx, {
                "id": item.get("id", str(idx)),  # 如果沒有 id，則使用索引作為 id
                "source": item.get("source", "unknown"),
                "conversations": conversations,
            }


if __name__ == '__main__':
    script_url = "llama_factory_data/qwen25_ESAinsTOD/qwen25_ESAinsTOD.py"

    config_names = datasets.get_dataset_config_names(script_url, trust_remote_code=True)
    print(f"Available configurations: {config_names}")

    dataset_infos = {k: v._to_yaml_dict() for k, v in datasets.get_dataset_infos(script_url, trust_remote_code=True).items()}
    # print(f"Dataset info: {json.dumps(dataset_infos, indent=2)}")

    split_names = datasets.get_dataset_split_names(
        script_url, config_name=config_names[-1], trust_remote_code=True)
    print(f"Available splits: {split_names}")

    tod_data = datasets.load_dataset(
        script_url,
        name="smkcfbs_irs_mix_schema_no_i_no_sa_ddb",
        split=split_names[-1],
        trust_remote_code=True,
    )
    print(f"Loaded dataset: {tod_data}")