from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from transformers import AutoTokenizer
import re

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
B_TEXT, E_TEXT = "<s>", "</s>"


class DatasetProceser:
    def __init__(
        self,
        data,
        train_samples=0,
        dataset_name="OpenAssistant/oasst1",
        model_name="",
        tokenizer=None,
    ):
        self.train_samples = train_samples
        # self.data = self._read_dataset()
        self.data = data
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.tokenizer = tokenizer

    def _read_train_dataset(self) -> pd.DataFrame:
        data = load_dataset(self.dataset_name)
        data = data.data["train"]
        data = data.to_pandas()
        return data

    @staticmethod
    def _convert_to_dataset_dict(
        train: pd.DataFrame, validation: pd.DataFrame = None, test: pd.DataFrame = None
    ) -> DatasetDict:
        dataset_dict = {"train": Dataset.from_pandas(train)}

        if validation is not None:
            dataset_dict["validation"] = Dataset.from_pandas(validation)

        if test is not None:
            dataset_dict["test"] = Dataset.from_pandas(test)

        return DatasetDict(dataset_dict)

    @staticmethod
    def split_text_human_assistant(text):
        conversation = []
        text = re.sub(r"\n\nHuman:", "Human:", text)
        # Substitute \n\nAssistant: with Assistant:
        part2 = re.sub(r"\n\nAssistant:", "Assistant:", text)
        pattern = r"(.*?)(Human:\s*(.*?)\s*Assistant:\s*(.*))"
        match = re.match(pattern, part2, re.DOTALL)
        while match:
            initial_text, user, part2 = (
                match.group(1).strip(),
                match.group(3).strip(),
                match.group(4).strip(),
            )
            conversation.append({"role": "user", "content": user})
            match = re.match(pattern, part2, re.DOTALL)
            if match:
                # we save assistant response and go to next iteration
                conversation.append(
                    {"role": "assistant", "content": match.group(1).strip()}
                )
            else:
                conversation.append({"role": "assistant", "content": part2})
        return [conversation]

    @staticmethod
    def filter_df_lenght_columns(
        df: pd.DataFrame, column_names: list, max_length: int
    ) -> pd.DataFrame:
        # Define the columns to sum the lengths of the lists
        df["len_total"] = df[column_names].apply(
            lambda row: sum(len(row[col]) for col in column_names), axis=1
        )
        # -----------------------------------
        df_higher_than_maxlenght = df[df["len_total"] > max_length]
        percentage = (len(df_higher_than_maxlenght) / len(df)) * 100
        print(f"Percentage of rows rows higher than {max_length}: {percentage:.2f}%")
        # -----------------------------------
        df = df[df["len_total"] <= max_length]
        df = df.drop(columns=["len_total"])
        return df

    def save_example_dataset(self, name: str = "example_dataset.csv"):
        self.data.head(10).to_csv(name, sep=";")

    def format_chat(
        self,
        df: pd.DataFrame,
        tokenizer=None,
        remove_columns=False,
        question_name: str = "question",
        answer_name: str = "answer",
        chat_name: str = "text",
    ) -> pd.DataFrame:
        if tokenizer is None:
            if self.tokenizer is None:
                raise ValueError("Tokenizer is not defined")
            else:
                tokenizer = self.tokenizer

        # check if is already in the correct format
        if isinstance(df.iloc[0][answer_name][0], dict):
            if (
                "content" in df.iloc[0][answer_name][0].keys()
                and "role" in df.iloc[0][answer_name][0].keys()
            ):
                df["conversation"] = df[answer_name]
            else:
                df["conversation"] = df.apply(
                    lambda row: self._preformat_chat(
                        row[question_name], row[answer_name], ""
                    ),
                    axis=1,
                )
        else:
            df["conversation"] = df.apply(
                lambda row: self._preformat_chat(
                    row[question_name], row[answer_name], ""
                ),
                axis=1,
            )
        if remove_columns:
            remove_columns = [question_name, answer_name]
            df = df.drop(columns=remove_columns)

        df[chat_name] = df.apply(
            lambda row: self._format_chat(tokenizer, row["conversation"]),
            axis=1,
        )
        remove_columns = ["conversation"]
        df = df.drop(columns=remove_columns)
        return df

    @staticmethod
    def format_to_mistralft(
        instance: str, output: str, general_message: str = ""
    ) -> str:
        # deprecated, now use format chat
        # The function tok should never generate the EOS token, however FastChat (used in vLLM) sends the full prompt as a string which might lead to incorrect tokenization of the EOS token and prompt injection. Users are encouraged to send tokens instead as described above.
        if general_message == "":
            return B_TEXT + B_INST + instance + E_INST + output + E_TEXT
        else:
            return (
                B_TEXT + B_INST + general_message + instance + E_INST + output + E_TEXT
            )

    @staticmethod
    def _filter_data_oasst1(data: pd.DataFrame) -> pd.DataFrame:
        # filter by language = en
        data = data[data["lang"] == "en"]
        # filter by deleted = False
        data = data[data["deleted"] == False]
        return data

    @staticmethod
    def _split_data_prompterassistant(data: pd.DataFrame):
        # filter by role = 'prompter'
        prompter = data[data["role"] == "prompter"]
        # filter by role = 'assistant'
        assistant = data[data["role"] == "assistant"]
        return prompter, assistant

    @staticmethod
    def _process_data_questionanswer(
        prompter: pd.DataFrame, assistant: pd.DataFrame
    ) -> pd.DataFrame:
        # count how many assistant responses has any prompt filterinf in the dataframe by prompter messague id = assistant parent id
        instances = []
        for _, row in prompter.iterrows():
            replies = assistant[assistant["parent_id"] == row["message_id"]]
            for _, reply in replies.iterrows():
                instances.append(
                    {
                        "question": row["text"],
                        "answer": reply["text"],
                        "feedback": reply["rank"],
                        "id": reply["message_id"],
                    }
                )

        data = pd.DataFrame(instances)
        data = data.dropna(subset=["feedback"])
        return data

    @staticmethod
    def _preformat_chat(prompt: str, answer: str, system_message="") -> dict:
        if system_message == "":
            return [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
        else:
            return [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]

    @staticmethod
    def _format_chat(tokenizer: AutoTokenizer, data):
        # The function tok should never generate the EOS token, however FastChat (used in vLLM) sends the full prompt as a string which might lead to incorrect tokenization of the EOS token and prompt injection. Users are encouraged to send tokens instead as described above.
        return tokenizer.apply_chat_template(
            data, tokenize=False, add_generation_prompt=False
        )
