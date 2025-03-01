
import sys


# sys.path.append('/home/csp/repo/LLMs/eye_transformer/')
# print("CWD", os.getcwd(), "PATH", sys.path)
from utils.lmdb_storage import LMDBStorage
import pathlib
import hashlib

sys.path.append("../..")
import numpy as np

path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve())
sys.path.append(path)
path = str(
    pathlib.Path(__file__)
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
)
sys.path.append(path)
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from eyetrackpy.data_generator.fixations_predictor_trained_1.fixations_predictor_model_1 import (
    FixationsPredictor_1,
)

from eyetrackpy.data_generator.fixations_predictor_trained_2.fixations_predictor_model_2 import (
    FixationsPredictor_2,
)
from typing import (
    TypeVar,
)

T = TypeVar("T", bound="Module")
import re


class MyRewardBase:
    def __init__(
        self,
        model_name,
        features_used=[1, 1, 1, 1, 1],
        *argv,
        **karg,
    ):
        self.features_used = features_used
        self.model_name = model_name
        self.memory_storage = LMDBStorage(
            db_path="/data/alop/eye_transformer/rlhf_rw/buffer_train.lmdb"
        )

    def _load_tokenizer(self, load_local_folder_name=None):
        if load_local_folder_name:
            tokenizer = AutoTokenizer.from_pretrained(load_local_folder_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
        if tokenizer.chat_template is None:
            tokenizer.chat_template = tokenizer.default_chat_template
        tokenizer.add_eos_token = True
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        tokenizer.padding_side = "right"
        chat_tokens = list(set(re.findall(r"(<.*?>)", tokenizer.default_chat_template)))

        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": tokenizer.additional_special_tokens
                + chat_tokens
            }
        )
        self.tokenizer = tokenizer
        return self.tokenizer

    def load_fx_model_1(self, hidden_size, remap=False, fp_dropout=[0.0, 0.3]):
        p_1, p_2 = fp_dropout

        self.modelTokenizer = self.tokenizer
        self.FP_model = FixationsPredictor_1(
            hidden_dim=128,
            drop_out=0.2,
            modelTokenizer=self.modelTokenizer,
            remap=remap,
        )
        self.fixations_embedding_projector = nn.Sequential(
            nn.Linear(1, 128),
            # nn.BatchNorm1d(128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=p_1),
            nn.Linear(128, hidden_size),
            nn.Dropout(p=p_2),
        )
        self.norm_layer_fix = nn.LayerNorm(hidden_size)

    def load_fx_model_2(
        self,
        hidden_size,
        remap=False,
        fp_dropout=[0.0, 0.3],
        load_fix_model=True,
    ):
        p_1, p_2 = fp_dropout
        self.modelTokenizer = self.tokenizer
        if load_fix_model:
            self.FP_model = FixationsPredictor_2(
                modelTokenizer=self.modelTokenizer, remap=remap
            )
        num_features = int(sum(self.features_used))
        self.fixations_embedding_projector = nn.Sequential(
            nn.Linear(num_features, 128),
            # nn.BatchNorm1d(128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=p_1),
            nn.Linear(128, hidden_size),
            nn.Dropout(p=p_2),
        )
        self.norm_layer_fix = nn.LayerNorm(hidden_size)

    def _compute_fixations(
        self, input_ids, attention_mask, remap=False, fixations_model_version=1
    ):
        if fixations_model_version == 1:
            (
                fixations,
                fixations_attention_mask,
                mapped_fixations,
                text_tokenized_model,
                text_tokenized_fix,
                sentences,
            ) = self.FP_model._compute_mapped_fixations(input_ids)
        elif fixations_model_version == 2:
            (
                fixations,
                fixations_attention_mask,
                mapped_fixations,
                text_tokenized_model,
                text_tokenized_fix,
                sentences,
            ) = self.FP_model._compute_mapped_fixations(input_ids, attention_mask)
        if remap:
            fixations_attention_mask = attention_mask
        return (
            fixations,
            fixations_attention_mask,
            mapped_fixations,
            text_tokenized_model,
            text_tokenized_fix,
            sentences,
        )

    def compute_fixations(
        self, input_ids, attention_mask, remap=False, fixations_model_version=1
    ):
        # compute fixations you can use the cached funcion that will try to search from them on the cache.
        # there are not identical, because when computing you are using batch size, but when computing to save not
        # (
        #     fixations2,
        #     fixations_attention_mask2,
        #     mapped_fixations2,
        #     text_tokenized_model2,
        #     text_tokenized_fix2,
        #     sentences2,
        # ) = self._compute_fixations(
        #     input_ids, attention_mask, remap, fixations_model_version
        # )
        (
            fixations,
            fixations_attention_mask,
            mapped_fixations,
            text_tokenized_model,
            text_tokenized_fix,
            sentences,
        ) = self.compute_fixations_cached(
            input_ids, attention_mask, remap, fixations_model_version
        )
        del text_tokenized_fix, text_tokenized_model, sentences
        fixations_normalized, fixations_attention_mask = self.process_fixations(
            fixations,
            fixations_attention_mask,
            mapped_fixations,
            remap,
            fixations_model_version,
        )
        return fixations_normalized, fixations_attention_mask

    def process_fixations(
        self,
        fixations,
        fixations_attention_mask,
        mapped_fixations,
        remap,
        fixations_model_version,
    ):
        # compute fixations
        if remap:
            fixations = mapped_fixations
            # mapped_fixations = mapped_fixations.detach()
            del mapped_fixations
        # add noise compute fixations
        if self.training is False and self.noise_factor > 0:
            noise = torch.randn_like(fixations) * self.noise_factor
            fixations = fixations + noise
            noise = noise.detach()
            del noise
        if fixations_model_version == 1:
            # with second fixator you dont need to unsqueeze
            fixations = fixations.unsqueeze(2)
        # project to embedding size dimension
        fixations_projected = self.fixations_embedding_projector(fixations)
        # fixations = fixations.detach()
        # del fixations
        # normalize fixations
        fixations_normalized = self.norm_layer_fix(fixations_projected)
        # fixations_projected = fixations_projected.detach()
        # del fixations_projected
        torch.cuda.empty_cache()
        return fixations_normalized, fixations_attention_mask

    @staticmethod
    def hash_value(val):
        return hashlib.md5(str(val).encode()).hexdigest()

    @staticmethod
    def remove_padding_from_batch(batch_token_ids, pad_token_id=0):
        # Iterate over each sequence in the batch and remove padding
        return [
            list(filter(lambda token_id: token_id != pad_token_id, sequence))
            for sequence in batch_token_ids
        ]

    def compute_fixations_cached(
        self, input_ids_original, attention_mask, remap=False, fixations_model_version=1
    ):
        device = input_ids_original.device
        # Convert the input tensor to a list of lists and remove padding.
        input_ids_list = input_ids_original.cpu().numpy().tolist()
        filtered_ids = self.remove_padding_from_batch(
            input_ids_list, self.tokenizer.pad_token_id
        )
        fixations_all, fixations_attention_mask_all = [], []
        # Iterate over each sequence in the filtered list
        for seq in filtered_ids:
            # Compute a hash of the sequence for caching
            if remap is True:
                # we only care about the model if we are remapping

                hash_id = self.hash_value(
                    seq + [fixations_model_version] + ["remap"] + [self.model_name]
                )
            else:
                hash_id = self.hash_value(
                    seq + [fixations_model_version] + [self.model_name]
                )  # because we can call the same sequence with and without remap and diff fixations predictor model

            # Attempt to retrieve the result from the cache
            result = self.memory_storage.getItem(hash_id)

            if result is None:
                # If the result is not in the cache, compute the fixations
                print("result NOT found")
                torch_seq = torch.LongTensor(np.asarray(seq)).to(device).unsqueeze(0)
                (
                    fixations,
                    fixations_attention_mask,
                    mapped_fixations,
                    text_tokenized_model,
                    text_tokenized_fix,
                    sentences,
                ) = self._compute_fixations(
                    torch_seq,
                    attention_mask,
                    remap=remap,
                    fixations_model_version=fixations_model_version,
                )
                del text_tokenized_fix, text_tokenized_model, sentences
                if remap:
                    fixations = mapped_fixations
                    fixations_attention_mask = attention_mask
                fixation_outputs = {
                    "fixations": fixations.cpu(),
                    "fixations_attention_mask": fixations_attention_mask.cpu(),
                }
                self.memory_storage.add(hash_id, fixation_outputs)
            else:
                print("result found")
                # If the result is found in the cache, convert back to tensors
                fixations = result["fixations"].to(device)
                fixations_attention_mask = result["fixations_attention_mask"].to(device)
            if fixations_model_version == 2:
                idx = np.where(np.array(self.features_used) == 1)[0]
                fixations = fixations[:, :, idx]
            fixations_all.append(fixations.squeeze())
            fixations_attention_mask_all.append(fixations_attention_mask.squeeze())

        # ---------------
        # Pad and concatenate all outputs into the final result tensor
        fixations_all = self._pad_and_concat(fixations_all)
        if remap is False:
            fixations_attention_mask_all = self._pad_and_concat(
                fixations_attention_mask_all
            )
            return fixations_all, fixations_attention_mask_all, None, None, None, None
        else:
            try:
                fixations_attention_mask_all = self._pad_and_concat(
                    fixations_attention_mask_all
                )
            except:
                # enter here on the last of the batch, take a look
                print(
                    f"problema con el remapping con len {len(fixations_attention_mask_all)}, {len(fixations_attention_mask_all[0])}"
                )
                print(fixations_attention_mask_all)

            fixations_attention_mask_all = attention_mask
            return None, fixations_attention_mask_all, fixations_all, None, None, None

    @staticmethod
    def _pad_and_concat(list_of_tensors):
        def pad_tensor(tensor, max_length):
            """Pads a tensor to the specified max_length with the last value in the tensor."""
            padding_length = max_length - tensor.size(0)
            if padding_length > 0:
                # padding_tensor = torch.full((padding_length,), 0, dtype=tensor.dtype).to(tensor.device)
                if tensor.dim() == 1:  # 1D tensor
                    # Create a 1D padding tensor of zeros
                    padding_tensor = torch.zeros(padding_length).to(tensor.device)
                elif tensor.dim() == 2:  # 2D tensor
                    # Create a 2D padding tensor of zeros
                    padding_tensor = torch.zeros(padding_length, tensor.size(1)).to(
                        tensor.device
                    )
                else:
                    raise ValueError("Only 1D and 2D tensors are supported.")
                tensor = torch.cat([tensor, padding_tensor])
            return tensor

        # Determine the maximum size for padding for each tensor position
        max_length = max([len(i) for i in list_of_tensors])

        return torch.stack([pad_tensor(item, max_length) for item in list_of_tensors])
