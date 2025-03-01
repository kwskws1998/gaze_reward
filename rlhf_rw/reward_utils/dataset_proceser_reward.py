from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets, table
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter
from functools import partial
from utils.dataset_proceser import DatasetProceser
from transformers import AutoTokenizer
from sklearn.model_selection import KFold
from typing import Union
import re

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
B_TEXT, B_TEXT = "<s>", "</s>"


def preprocess_data_reward(
    data: pd.DataFrame,
    tokenizer: AutoTokenizer,
    chosen_name: str,
    rejected_name: str,
    max_tokens=None,
) -> dict:
    data_processed = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }

    for chosen, rejected in zip(data[chosen_name], data[rejected_name]):
        if max_tokens:
            tokenized_chosen = tokenizer(chosen, max_length=max_tokens, truncation=True)
            tokenized_rejected = tokenizer(
                rejected, max_length=max_tokens, truncation=True
            )
        else:
            tokenized_chosen = tokenizer(chosen)
            tokenized_rejected = tokenizer(rejected)

        data_processed["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        data_processed["attention_mask_chosen"].append(
            tokenized_chosen["attention_mask"]
        )
        data_processed["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        data_processed["attention_mask_rejected"].append(
            tokenized_rejected["attention_mask"]
        )

    return data_processed


class DatasetProceserReward(DatasetProceser):
    def __init__(
        self,
        data,
        train_samples=0,
        dataset_name="OpenAssistant/oasst1",
        model_name="",
        max_length=None,
        tokenizer=None,
        fold=0,
        seed=42,
        test_split=0.2,
        validation_split=0.15,
        subsample_percentage=1,
    ):
        super().__init__(
            data=data,
            train_samples=train_samples,
            dataset_name=dataset_name,
            model_name=model_name,
            tokenizer=tokenizer,
        )
        self.max_length = max_length
        self.seed = seed
        self.seed = 42
        self.test_split = test_split
        self.validation_split = validation_split
        self.subsample_percentage = subsample_percentage
        # -------- custom preprocessing----------#
        # if self.dataset_name == "OpenAssistant/oasst1":
        #     self.preprocess_oasst1()
        # elif "HelpSteer2" in self.dataset_name:
        #     self.preprocess_HelpSteer2()
        # elif "argilla" in self.dataset_name:
        #     self.preprocess_argilla()
        # -------- custom preprocessing----------#
        self.preprocess_general(max_length=self.max_length)
        if fold > 0:
            self.data = self.split_dataset_fold(fold=fold)
        if self.subsample_percentage > 0 and self.subsample_percentage < 1:
            self.subsample_train_data(subsample_percentage=self.subsample_percentage)
        if dataset_name != "allenai/reward-bench":
            self.split_validation()

        print(self.data)

    @classmethod
    def from_datasets(
        cls,
        dataset_name="OpenAssistant/oasst1",
        train_samples=0,
        model_name="",
        split="",
        tokenizer=None,
        fold=0,
        subsample_percentage=1,
        max_length=None,
    ):
        if split != "":
            data = load_dataset(dataset_name, split=split)
        else:
            data = load_dataset(dataset_name)
        return cls(
            data=data,
            dataset_name=dataset_name,
            train_samples=train_samples,
            model_name=model_name,
            tokenizer=tokenizer,
            fold=fold,
            subsample_percentage=subsample_percentage,
            max_length=max_length,
        )

    def split_dataset_fold(self, n_splits=5, shuffle=True, fold=1):
        # Assuming `dataset` is your DatasetDict with 'train' and 'validation' splits
        # dataset = load_dataset('your_dataset_name')

        # Merge train and validation datasets
        if isinstance(self.data, DatasetDict):
            print(len(self.data["train"]))
            print(len(self.data["test"]))

            combined_dataset = concatenate_datasets(
                [self.data["train"], self.data["test"]]
            )
        else:
            combined_dataset = self.data

        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=self.seed)

        dataset_np = combined_dataset.to_pandas().to_numpy()
        for i, (train_index, val_index) in enumerate(kf.split(dataset_np)):
            if i + 1 == int(fold):
                train_data = combined_dataset.select(train_index.tolist())
                val_data = combined_dataset.select(val_index.tolist())
                self.data = DatasetDict({"train": train_data, "test": val_data})
                break
        print(len(self.data["train"]))
        print(len(self.data["test"]))
        return self.data

    def subsample_train_data(self, subsample_percentage=0.10):
        # subsample the train data to not use everythin
        subsample_train = self.data["train"].train_test_split(
            test_size=1 - subsample_percentage, seed=self.seed
        )
        self.data["train"] = subsample_train["train"]

    def split_validation(self):
        # Split the dataset: 85% training, 15% validation
        train_test_split = self.data["train"].train_test_split(
            test_size=self.validation_split, seed=self.seed
        )
        data_dict = {
            "train": train_test_split["train"],
            "validation": train_test_split["test"],
        }
        if "validation" in list(self.data.keys()):
            data_dict["test"] = self.data["validation"]
        elif "test" in list(self.data.keys()):
            data_dict["test"] = self.data["test"]
        else:
            train_test_split2 = train_test_split["train"].train_test_split(
                test_size=self.validation_split, seed=self.seed
            )
            data_dict = {
                "train": train_test_split2["train"],
                "validation": train_test_split["test"],
                "test": train_test_split2["test"],
            }
        # Create a new DatasetDict with the split
        self.data = DatasetDict(data_dict)

    # def preprocess_oasst1(self):
    #     # check if self.data is DatasetDict type
    #     if isinstance(self.data, DatasetDict):
    #         data = {}
    #         splits = list(self.data.keys())
    #         for split in splits:
    #             data[split] = Dataset.from_pandas(
    #                 self._preprocess_oasst1_split(self.data.data[split])
    #             )
    #         self.data = DatasetDict(data)
    #     else:
    #         data_split = self._preprocess_oasst1_split(self.data)
    #         self.data = Dataset.from_pandas(data_split).train_test_split(
    #             test_size=self.test_split
    #         )

    # def preprocess_HelpSteer2(self, max_length=None):
    #     # check if self.data is DatasetDict type
    #     if isinstance(self.data, DatasetDict):
    #         data = {}
    #         splits = list(self.data.keys())
    #         for split in splits:
    #             data[split] = Dataset.from_pandas(
    #                 self._preprocess_HelpSteer2_split(self.data.data[split])
    #             )
    #         self.data = DatasetDict(data)
    #     else:
    #         data_split = self._preprocess_HelpSteer2_split(self.data, max_length)
    #         self.data = Dataset.from_pandas(data_split).train_test_split(
    #             test_size=self.test_split
    #         )

    def preprocess_general(self, max_length=None):
        # check if self.data is DatasetDict type
        if isinstance(self.data, DatasetDict):
            data = {}
            splits = list(self.data.keys())
            for split in splits:
                print("processing split: ", split)
                data_split = self._preprocess_general_split(
                    self.data.data[split], max_length
                )
                data[split] = Dataset.from_pandas(data_split)
            self.data = DatasetDict(data)
        else:
            data_split = self._preprocess_general_split(self.data, max_length)
            self.data = Dataset.from_pandas(data_split).train_test_split(
                test_size=self.test_split, seed=self.seed
            )

    def _preprocess_hhrlhf_split(self, data_split: Union[Dataset, pd.DataFrame]):
        if not isinstance(data_split, pd.DataFrame):
            data_split = data_split.to_pandas()
        data_split["chosen"] = data_split["chosen"].apply(
            lambda x: pd.Series(self.split_text_human_assistant(x))
        )
        data_split["rejected"] = data_split["rejected"].apply(
            lambda x: pd.Series(self.split_text_human_assistant(x))
        )
        return data_split

    def _preprocess_oasst1_split(
        self, data_split: Union[Dataset, pd.DataFrame]
    ) -> pd.DataFrame:
        if not isinstance(data_split, pd.DataFrame):
            data_split = data_split.to_pandas()
        data_split = self._filter_data_oasst1(data_split)
        self.prompter_data, self.assistant_data = self._split_data_prompterassistant(
            data_split
        )
        data_split = self._process_data_questionanswer(
            self.prompter_data, self.assistant_data
        )
        data_split = self._process_responses_chosen_rejected(data_split)
        # data_split.to_csv("openass.csv")
        # ----------------DEPRECATED----------------
        # substitudes the apply chat format for the one in the tokenizer
        # if "mistral" in self.model_name.lower():
        #     data_split = self._add_question_answer_mistral(data_split)
        # else:
        #     data_split = self._add_question_answer(data_split)
        # ----------------DEPRECATED----------------
        # data_split = self._preprocess_convert_chat(data_split, max_length)
        return data_split

    def _preprocess_HelpSteer2_split(self, data_split: Union[Dataset, pd.DataFrame]):
        """
        HelpSteer2 train set into a preference dataset by taking the response with the higher helpfulness score as the chosen response, with the remaining response being the rejected response. In cases where the helpfulness scores were identical, we discarded that pair entirely.
        """

        if not isinstance(data_split, pd.DataFrame):
            data_split = data_split.to_pandas()
        df_list = []
        # create a list with unique values in promp column
        for prompt in data_split["prompt"].unique():
            # filter the dataframe by prompt
            prompt_df = data_split[data_split["prompt"] == prompt]
            # sort the dataframe by the helpfulness score
            prompt_df = prompt_df.sort_values(by="helpfulness", ascending=False)
            # take the first row as the chosen response
            chosen = prompt_df.iloc[0]
            # take the second row as the rejected response
            try:
                rejected = prompt_df.iloc[1]
            except Exception as e:
                continue
            # check if bout have the same score
            if chosen["helpfulness"] == rejected["helpfulness"]:
                continue
            # append the chosen and rejected responses to the new dataframe
            df_list.append(
                {
                    "question": prompt,
                    "chosen": chosen["response"],
                    "chosen_score": chosen["helpfulness"],
                    "rejected": rejected["response"],
                    "rejected_score": rejected["helpfulness"],
                }
            )

        df = pd.DataFrame(df_list)
        return df

    def _preprocess_split(
        self, data_split: Union[Dataset, pd.DataFrame]
    ) -> pd.DataFrame:
        if not isinstance(data_split, pd.DataFrame):
            data_split = data_split.to_pandas()

        # data_split = self._add_question_answer(data_split)
        return data_split

    def _preprocess_allenai(
        self, data_split: Union[Dataset, pd.DataFrame]
    ) -> pd.DataFrame:
        if not isinstance(data_split, pd.DataFrame):
            data_split = data_split.to_pandas()
        # rename columns prompt to question
        # data_split = self._add_question_answer(data_split)
        data_split.rename(columns={"prompt": "question"}, inplace=True)
        return data_split

    @staticmethod
    def find_extreme_responses(responses):
        # Initialize with the first element
        if len(responses) == 0:
            return None, None, None, None
        min_entry = responses[0]
        max_entry = responses[0]

        for entry in responses[1:]:
            if entry["overall_score"] < min_entry["overall_score"]:
                min_entry = entry
            if entry["overall_score"] > max_entry["overall_score"]:
                max_entry = entry

        return (
            max_entry["response"],
            max_entry["overall_score"],
            min_entry["response"],
            min_entry["overall_score"],
        )

    def _preprocess_UltraFeedback_split(self, data_split: Union[Dataset, pd.DataFrame]):
        if not isinstance(data_split, pd.DataFrame):
            data_split = data_split.to_pandas()
        prompts = []
        for _, row in data_split.iterrows():
            chosen, chosen_score, rejected, rejected_score = (
                self.find_extreme_responses(row["completions"])
            )
            if chosen is not None:
                prompts.append(
                    {
                        "question": row["instruction"],
                        "chosen": chosen,
                        "chosen_score": chosen_score,
                        "rejected": rejected,
                        "rejected_score": rejected_score,
                    }
                )
        return pd.DataFrame(prompts)

    def _preprocess_general_split(
        self, data_split: Union[Dataset, pd.DataFrame], max_length: int = None
    ) -> pd.DataFrame:
        # -------- custom preprocessing----------#
        if not isinstance(data_split, pd.DataFrame):
            data_split = data_split.to_pandas()
        if "OpenAssistant/oasst1" in self.dataset_name:
            data_split = self._preprocess_oasst1_split(data_split)
        elif "HelpSteer2" in self.dataset_name:
            data_split = self._preprocess_HelpSteer2_split(data_split)
        elif "Anthropic/hh-rlhf" in self.dataset_name:
            data_split = self._preprocess_hhrlhf_split(data_split)
        elif "argilla" in self.dataset_name:
            data_split = self._preprocess_split(data_split)
        elif "openbmb/UltraFeedback" in self.dataset_name:
            data_split = self._preprocess_UltraFeedback_split(data_split)
        elif "allenai/reward-bench" in self.dataset_name:
            data_split = self._preprocess_allenai(data_split)
        # -------- custom preprocessing----------#
        # expects in each row a question, chosen, rejected
        data_split = self._preprocess_convert_chat(data_split, max_length)
        return data_split

    def _preprocess_convert_chat(self, data_split, max_length):
        """
        Prepare data for the input expected by the reward trainer
        """
        data_split = self.format_chat(
            data_split,
            remove_columns=False,
            question_name="question",
            answer_name="chosen",
            chat_name="chosen_chat",
        )
        data_split = self.format_chat(
            data_split,
            remove_columns=False,
            question_name="question",
            answer_name="rejected",
            chat_name="rejected_chat",
        )
        if max_length:
            data_split = self.filter_df_lenght_columns(
                df=data_split,
                column_names=["chosen_chat", "rejected_chat"],
                max_length=max_length,
            )
        return data_split

    @staticmethod
    def filter_instances_lenght(
        prompter: pd.DataFrame, assistant: pd.DataFrame, max_length: int = 350
    ):
        # count how many assistant responses has any prompt filterinf in the dataframe by prompter messague id = assistant parent id
        instances = {}
        for index, row in prompter.iterrows():
            instances[row["message_id"]] = {}
            instances[row["text"]] = row.text
            replies = assistant[assistant["parent_id"] == row["message_id"]]
            instances[row["message_id"]]["replies"] = replies
            replies = replies[replies["text"].str.len() < max_length]
            instances[row["message_id"]]["number_of_replies"] = replies.shape[0]
            instances[row["message_id"]]["max_lenght"] = replies.text.str.len().max()
            # filter by lenght < 350
            print("max lenght: ", instances[row["message_id"]]["max_lenght"])
            instances[row["message_id"]]["min_lenght"] = replies.text.str.len().min()
            print("min lenght: ", instances[row["message_id"]]["min_lenght"])

    @staticmethod
    def plot_histogram(instances: dict):
        # convert instances to dataframe
        instances = pd.DataFrame.from_dict(instances, orient="index")
        # Create a histogram for the 'column_name'
        plt.hist(instances["max_lenght"], bins=5, edgecolor="black")
        plt.title("Histogram of max_lenght")
        plt.xlabel("Values")
        plt.ylabel("Frequency")
        plt.show()

    @staticmethod
    def _process_responses_chosen_rejected(df: pd.DataFrame) -> pd.DataFrame:
        if "id" in df.columns:
            df["tup"] = list(zip(df["answer"], df["feedback"], df["id"]))
        else:
            df["tup"] = list(zip(df["answer"], df["feedback"]))
        df_g = df.groupby("question")["tup"].apply(list).reset_index()
        df_g["sorted_tup"] = df_g["tup"].apply(lambda x: sorted(x, key=itemgetter(1)))
        df_g["chosen"] = df_g["sorted_tup"].apply(lambda x: x[-1][0])
        df_g["chosen_score"] = df_g["sorted_tup"].apply(lambda x: x[-1][1])
        df_g["rejected"] = df_g["sorted_tup"].apply(lambda x: x[0][0])
        df_g["rejected_score"] = df_g["sorted_tup"].apply(lambda x: x[0][1])
        if "id" in df.columns:
            df_g["chosen_id"] = df_g["sorted_tup"].apply(lambda x: x[-1][2])
            df_g["rejected_id"] = df_g["sorted_tup"].apply(lambda x: x[0][2])
        df_g = df_g.dropna()
        # delete rows where the chosen_score is the same as the rejected_score
        df_g = df_g[df_g["chosen_score"] != df_g["rejected_score"]]
        df_g = df_g.drop(
            columns=["tup", "sorted_tup", "chosen_score", "rejected_score"]
        )
        # df_g = df_g[(df_g['chosen_score']>=4.0) & (df_g['rejected_score']<4.0)]
        return df_g

    @staticmethod
    # TODO CHANGE FOR CHAT TEMPLATE
    def _add_question_answer(df: pd.DataFrame) -> pd.DataFrame:
        df["chosen"] = "Human: " + df["question"] + "\n" + " Assistant: " + df["chosen"]
        df["rejected"] = (
            "Human: " + df["question"] + "\n" + " Assistant: " + df["rejected"]
        )
        return df

    def _add_question_answer_mistral(self, df: pd.DataFrame) -> pd.DataFrame:
        df["chosen"] = df.apply(
            lambda row: self.format_to_mistralft(row["question"], row["chosen"], ""),
            axis=1,
        )
        df["rejected"] = df.apply(
            lambda row: self.format_to_mistralft(row["question"], row["rejected"], ""),
            axis=1,
        )
        return df

    def preprocess_data_reward(
        self,
        tokenizer: AutoTokenizer = None,
        batch_size: int = 1000,
        chosen_name: str = "chosen_chat",
        rejected_name: str = "rejected_chat",
        eval_mode: bool = False,
        max_tokens=None,
    ):
        if tokenizer is None:
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required")
            else:
                tokenizer = self.tokenizer
        function_process = partial(
            preprocess_data_reward,
            tokenizer=tokenizer,
            chosen_name=chosen_name,
            rejected_name=rejected_name,
            max_tokens=max_tokens,
        )

        if "test" in self.data:
            self.data["test"] = self.data["test"].map(
                function_process,
                batched=True,
                batch_size=batch_size,
            )
        if eval_mode is True:
            return self.data

        if "validation" in self.data:
            self.data["validation"] = self.data["validation"].map(
                function_process,
                batched=True,
                batch_size=batch_size,
            )
        if "train" in self.data:
            self.data["train"] = self.data["train"].map(
                function_process,
                batched=True,
                batch_size=batch_size,
            )
        return self.data
