import os
import sys
import pathlib

sys.path.append("../..")
path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve())
sys.path.append(path)
from rlhf_rw.reward_utils.dataset_proceser_reward import DatasetProceserReward
from peft import prepare_model_for_kbit_training
import torch
from transformers import BitsAndBytesConfig
import warnings
from transformers import TrainingArguments, get_scheduler
import math
from datasets import concatenate_datasets

warnings.filterwarnings("ignore")
from trl import RewardTrainer
from peft import LoraConfig, TaskType, get_peft_model
import wandb



def custom_cosine_scheduler(
    optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.9
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    return get_scheduler(
        "lambda",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        lr_lambda=lr_lambda,
    )


class RewardTrainerConstructor:
    def __init__(
        self,
        model_name="mistralai/Mistral-7B-v0.1",
        dataset_name="OpenAssistant/oasst1",
        dataset_split="",
        fold=0,
        subsample_percentage=1.0,
        use_lora=False,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        use_quantization=False,
        batch_size=4,
        train_epochs=20,
        gradient_acum_steps=64,
        logging_steps=0,
        gradient_checkpointing=False,
        learning_rate=1e-5,
        lr_scheduler_type="constant_with_warmup",
        min_lr_ratio=0.8,
        weight_decay=0.1,
    ):
        self.wandb_project = "rlhf_rewardmodel"
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_quantization = use_quantization
        self.bnb_config = None
        self.batch_size = int(batch_size)
        self.fold = fold
        self.subsample_percentage = subsample_percentage
        self.train_epochs = train_epochs
        self.gradient_acum_steps = gradient_acum_steps
        self.gradient_checkpointing = gradient_checkpointing
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.min_lr_ratio = min_lr_ratio
        self.weight_decay = weight_decay
        if logging_steps > 0:
            self.logging_steps = logging_steps
        else:
            self.logging_steps = int(max(1, round(128 * 2 / (self.batch_size), 0)))
        # self.logging_steps = max(
        #     1, round(64 / (self.gradient_acum_steps * self.batch_size), 0)
        # )
        self.model_name_log = ""
        print(f"batch_size: {batch_size}")

    def config_quantization(self):
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        return self.bnb_config

    def load_dataset(
        self,
        train_samples=0,
        split="",
        eval_mode=False,
        max_length=10000,
        max_tokens=None,
    ):
        # dataset_name = 'OpenAssistant/oasst1'
        # dataset_name = "Anthropic/hh-rlhf"
        dataset_procesor = DatasetProceserReward.from_datasets(
            dataset_name=self.dataset_name,
            train_samples=train_samples,
            model_name=self.model_name,
            split=split,
            tokenizer=self.tokenizer,
            fold=self.fold,
            subsample_percentage=self.subsample_percentage,
            max_length=max_length,
        )
        self.dataset_procesor = dataset_procesor
        raw_datasets = dataset_procesor.preprocess_data_reward(
            batch_size=self.batch_size, eval_mode=eval_mode, max_tokens=max_tokens
        )
        if dataset_procesor.train_samples > 0:
            # Shuffle the dataset
            # shuffled_dataset = dataset.shuffle()
            raw_datasets["train"] = raw_datasets["train"].select(
                range(dataset_procesor.train_samples)
            )
        # print(f"Raw_datasets train num samples: {raw_datasets["train"].num_rows}")
        # print(
        #     f"Raw_datasets validation num samples: {raw_datasets["validation"].num_rows}"
        # )
        # print(f"Raw_datasets test num samples: {raw_datasets["test"].num_rows}")
        self.train_dataset = raw_datasets["train"]
        self.eval_dataset = raw_datasets["validation"]
        self.test_dataset = raw_datasets["test"]

    def load_dataset_rewardbench(self, max_length=None, max_tokens=None, model="all"):
        # dataset_name = 'OpenAssistant/oasst1'
        # dataset_name = "Anthropic/hh-rlhf"
        dataset_procesor = DatasetProceserReward.from_datasets(
            dataset_name=self.dataset_name,
            model_name=self.model_name,
            tokenizer=self.tokenizer,
            fold=self.fold,
            subsample_percentage=self.subsample_percentage,
            max_length=max_length,
        )
        self.dataset_procesor = dataset_procesor
        self.dataset_procesor.data["test"] = self.dataset_procesor.data["filtered"]
        # self.dataset_procesor.data['test'] = concatenate_datasets([self.dataset_procesor.data['raw'], self.dataset_procesor.data['filtered'        self.dataset_procesor.data["test"] = self.dataset_procesor.data["filtered"]
        # concatenate all dataset in dataset dic in one called 'test'
        raw_datasets = dataset_procesor.preprocess_data_reward(
            batch_size=self.batch_size, eval_mode=True, max_tokens=max_tokens
        )

        if model == "all":
            self.test_dataset = raw_datasets["test"]
        else:
            rb_all = {}
            for rb_subset in list(set(raw_datasets["test"]["subset"])):
                rb_all[rb_subset] = raw_datasets["test"].filter(
                    lambda x: x["subset"] == rb_subset
                )
            self.test_dataset = rb_all

    def config_lora(self, lora_r=8, lora_alpha=32, lora_dropout=0.1):
        self.peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.model = get_peft_model(self.model, self.peft_config)

    def set_name_run(self):
        # chekc if model_name_log is in self
        if hasattr(self, "input_layer"):
            if self.input_layer is not None:
                self.model_name_log += "_fixlay_" + ",".join(
                    [str(x) for x in self.input_layer]
                )
        if hasattr(self, "freeze_layer"):
            if self.freeze_layer is not None:
                self.model_name_log += "_freezlay_" + str(self.freeze_layer)
        if hasattr(self, "use_quantization"):
            self.model_name_log += "_quant" if self.use_quantization else ""
        if hasattr(self, "use_lora"):
            self.model_name_log += "_lora" if self.use_lora else ""
        if hasattr(self.model, "freeze"):
            self.model_name_log += "_freeze" if self.model.freeze else ""
        prefix = "add_"
        if hasattr(self, "concat"):
            if self.concat:
                if hasattr(self, "use_softprompt"):
                    if self.use_softprompt:
                        prefix = "concat_sp_"
                    else:
                        prefix = "concat_"
        self.model_name_log = prefix + self.model_name_log

    def set_trainer(self):
        wandb.login()
        os.environ["WANDB_PROJECT"] = self.wandb_project
        self.set_name_run()
        print(f"model_name_log: {self.model_name_log}")
        print(f"gradient_acum_steps: {self.gradient_acum_steps}")
        training_args = TrainingArguments(
            output_dir="./reward_model",
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_acum_steps,
            gradient_checkpointing=self.gradient_checkpointing,
            evaluation_strategy="steps",
            logging_steps=self.logging_steps,
            num_train_epochs=self.train_epochs,
            report_to="wandb",
            run_name=self.model_name_log,
        )
        if self.use_lora is True:
            print("Using LORA for trainer")
            self.trainer = RewardTrainer(
                model=self.model,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                peft_config=self.peft_config,
            )
        else:
            print("Not using Using LORA")
            self.trainer = RewardTrainer(
                model=self.model,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

    def train_model(self, train_samples=0, fold=None):
        if self.use_quantization:
            print("Config quantization")
            self.config_quantization()
        self.load_model()
        if self.use_quantization:
            print("Prepare_model_for_kbit_training")
            self.model = prepare_model_for_kbit_training(self.model)
        self.load_dataset(train_samples=train_samples, split=self.dataset_split)
        # check if freeze is in self.model
        if not hasattr(self.model, "freeze"):
            self.model.freeze = False
        if self.use_lora:
            self.config_lora()
            
        self.trainer.train()

    def load_model(self):
        raise NotImplementedError("Subclasses should implement this method.")
