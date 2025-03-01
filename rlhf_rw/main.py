import os
import sys
import pathlib
import torch
import argparse
import json

sys.path.append("../..")
path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve())
sys.path.append(path)

from rlhf_rw.trainers.reward_trainer_general import (
    RewardTrainerConstructorGeneral,
)
from transformers import set_seed

import os



def get_unique_folder_name(
    base_folder,
    model_name,
    dataset_name,
    concat,
    use_softprompt,
    batch_size,
    train_epochs,
    gradient_acum_steps,
    logging_steps,
    learning_rate,
    lr_scheduler_type,
    min_lr_ratio,
    weight_decay,
    seed,
    fixations_model_version,
    fp_dropout,
):
    # Start with the original folder name
    folder_name = create_folder_name(
        model_name,
        dataset_name,
        concat,
        use_softprompt,
        batch_size,
        train_epochs,
        gradient_acum_steps,
        logging_steps,
        learning_rate,
        lr_scheduler_type,
        min_lr_ratio,
        weight_decay,
        seed,
        fixations_model_version,
        fp_dropout,
    )
    folder_path = os.path.join(base_folder, folder_name)
    # Initialize version number
    version = 1
    # Check if the folder exists and add version suffix until it doesn't
    while os.path.exists(folder_path):
        new_folder_name = f"{folder_name}_v{version}"
        folder_path = os.path.join(base_folder, new_folder_name)
        version += 1

    return os.path.join(base_folder, folder_name), folder_path


def create_folder_name(
    model_name,
    dataset_name,
    concat,
    use_softprompt,
    batch_size,
    train_epochs,
    gradient_acum_steps,
    logging_steps,
    learning_rate,
    lr_scheduler_type,
    min_lr_ratio,
    weight_decay,
    seed,
    fixations_model_version,
    fp_dropout,
):
    # dataset_name, model name, concat and use_softpormpt i want to be folders
    folder_name = f"{model_name.replace('/', '-')}/{dataset_name.replace('/', '-')}/{concat}/{use_softprompt}"

    add_info = f"_batch_size_{batch_size}"
    add_info += f"_train_epochs_{train_epochs}"
    add_info += f"_gradient_acum_steps_{gradient_acum_steps}"
    add_info += f"_logging_steps_{logging_steps}"
    add_info += f"_learning_rate_{learning_rate}"
    add_info += f"_lr_scheduler_type_{lr_scheduler_type}"
    add_info += f"_min_lr_ratio_{min_lr_ratio}"
    add_info += f"_weight_decay_{weight_decay}"
    add_info += f"_seed_{seed}"
    add_info += f"_fmv_{fixations_model_version}"
    add_info += f"_fp_dropout_{fp_dropout[0]}_{fp_dropout[1]}"

    return folder_name + add_info


def create_model_name(model_name, dataset_name, concat, use_softprompt):
    model_name = model_name.split("/")[1] + "-rm-" + dataset_name.replace("/", "-")
    if concat and use_softprompt:
        model_name += "-concateye"
    return model_name


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CTORCH_USE_CUDA_DSA"] = "1"


if __name__ == "__main__":
    # Set the seed for reproducibility

    # We use our original fixations model for training but without fixations.
    # Difference between v1 and v2 is in the tokenization process
    # With v2 we use directly mistral token and with v1 tokens provided bu the fixations predictor
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-d",
        "--dataset_name",
        type=str,
        help="name of the dataset",
        default="OpenAssistant/oasst1",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        help="name of the model",
        default="mistralai/Mistral-7B-v0.1",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        help="dataset_split",
        default="",
    )
    parser.add_argument(
        "--subsample_percentage", type=float, help="subsample_percentage", default=1
    )
    parser.add_argument("--input_layer", help="train layer to add fixations", default=3)
    parser.add_argument("--freeze_layer", help="freeze_layer", default=False)
    parser.add_argument("--use_lora", help="use lora on trainig", default=True)
    parser.add_argument(
        "--use_quantization", help="use quantization on training", default=True
    )
    parser.add_argument("--concat", help="concatenate fixations", default=True)
    parser.add_argument("--use_softprompt", help="use_softprompt", default=True)
    parser.add_argument("--seed", type=int, help="seed", default=42)
    parser.add_argument("--batch_size", type=int, help="batch size", default=16)
    parser.add_argument("--fold", type=int, help="fold", default=0)
    parser.add_argument("--train_epochs", type=int, help="train_epochs", default=2)
    parser.add_argument("--max_length", type=int, help="max_length", default=10000)
    parser.add_argument(
        "--learning_rate", type=float, help="learning rate", default=5e-5
    )
    parser.add_argument("--weight_decay", type=float, help="weight_decay", default=0.1)
    parser.add_argument("--min_lr_ratio", type=float, help="min_lr_ratio", default=0.7)
    parser.add_argument("--noise_factor", type=float, help="noise_factor", default=0.0)
    parser.add_argument(
        "--lr_scheduler_type", help="lr_scheduler_type", default="constant_with_warmup"
    )
    parser.add_argument("--grid_search", help="grid_search", default=False)
    parser.add_argument(
        "--gradient_acum_steps", type=int, help="gradient_acum_steps", default=1
    )
    parser.add_argument(
        "--gradient_checkpointing", help="gradient_checkpointing", default=True
    )
    parser.add_argument("--logging_steps", type=int, help="logging_steps", default=0)
    parser.add_argument("--train_samples", type=int, help="train_samples", default=0)
    parser.add_argument("--freeze", help="freeze", default=0)
    parser.add_argument("--mode", help="train or evaluate", default="train")
    parser.add_argument("--fp_dropout", help="train or evaluate", default="0.1,0.3")
    parser.add_argument(
        "-fmv", "--fixations_model_version", help="fixations_model_version", default="1"
    )
    parser.add_argument("--features_used", help="features used", default="1,1,1,1,1")

    args = parser.parse_args()
    seed = int(args.seed)
    set_seed(seed)
    model_name = args.model_name
    dataset_name = args.dataset_name
    train_samples = int(args.train_samples)
    subsample_percentage = float(args.subsample_percentage)
    fold = int(args.fold)
    dataset_split = str(args.dataset_split)
    batch_size = int(args.batch_size)
    train_epochs = int(args.train_epochs)
    gradient_acum_steps = int(args.gradient_acum_steps)
    logging_steps = int(args.logging_steps)
    gradient_checkpointing = str(args.gradient_checkpointing).lower() == "true"
    learning_rate = float(args.learning_rate)
    weight_decay = float(args.weight_decay)
    lr_scheduler_type = args.lr_scheduler_type
    min_lr_ratio = float(args.min_lr_ratio)
    noise_factor = float(args.noise_factor)
    input_layer = [int(element) for element in str(args.input_layer).split(",")]
    freeze = str(args.freeze).lower() == "true"
    use_softprompt = str(args.use_softprompt).lower() == "true"
    if args.freeze_layer == 0 and freeze is True:
        freeze_layer = input_layer
    elif args.freeze_layer == 0 and freeze is False:
        freeze_layer = None
    else:
        freeze_layer = int(args.freeze_layer)

    use_lora = str(args.use_lora).lower() == "true"
    use_quantization = str(args.use_quantization).lower() == "true"
    concat = str(args.concat).lower() == "true"
    if concat is False:
        use_softprompt = False
    grid_search = str(args.grid_search).lower() == "true"
    mode = str(args.mode).lower()
    fp_dropout = [float(element) for element in str(args.fp_dropout).split(",")]
    fixations_model_version = int(args.fixations_model_version)
    features_used = [int(element) for element in str(args.features_used).split(",")]
    max_length = int(args.max_length)
    max_tokens = None
    if fixations_model_version == 2 and concat is True:
        # if fixations_model_version == 2 and use_softprompt is True:
        max_tokens = 1350
        max_length = 10000

    # create folder name with all args
    name = create_model_name(model_name, dataset_name, concat, use_softprompt)
    reward_trainer = RewardTrainerConstructorGeneral(
        model_name=model_name,
        use_lora=use_lora,
        use_quantization=use_quantization,
        concat=concat,
        use_softprompt=use_softprompt,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        fold=fold,
        subsample_percentage=subsample_percentage,
        noise_factor=noise_factor,
        input_layer=input_layer,
        freeze_layer=freeze_layer,
        freeze=freeze,
        batch_size=batch_size,
        train_epochs=train_epochs,
        gradient_acum_steps=gradient_acum_steps,
        logging_steps=logging_steps,
        gradient_checkpointing=gradient_checkpointing,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        min_lr_ratio=min_lr_ratio,
        weight_decay=weight_decay,
        fp_dropout=fp_dropout,
        fixations_model_version=fixations_model_version,
        seed=seed,
        grid_search=grid_search,
        load_fix_model=True,
        features_used=features_used,
        max_length=max_length,
        max_tokens=max_tokens,
    )
    folder_name_path, folder_name_unique_path = get_unique_folder_name(
        str(pathlib.Path(__file__).parent.resolve().parent.resolve()) + "/models_save/",
        model_name,
        dataset_name,
        concat,
        use_softprompt,
        batch_size,
        train_epochs,
        gradient_acum_steps,
        logging_steps,
        learning_rate,
        lr_scheduler_type,
        min_lr_ratio,
        weight_decay,
        seed,
        fixations_model_version,
        fp_dropout,
    )
    if mode == "train":
        # print(f'Model used {reward_trainer.model_name_log}')
        reward_trainer.train_model(
            train_samples=train_samples, save_folder=folder_name_unique_path
        )

        # save_model
        reward_trainer.trainer.save_model(folder_name_unique_path)
        if concat is False or use_softprompt is True:
            torch.save(
                reward_trainer.model.base_model.model.fixations_embedding_projector.state_dict(),
                folder_name_unique_path + "/fixations_projector_state_dict2.bin",
            )
            torch.save(
                reward_trainer.model.base_model.model.norm_layer_fix.state_dict(),
                folder_name_unique_path + "/layer_norm_state_dic2.bin",
            )

        # reward_trainer.model.base_model.save_pretrained(folder_name_unique_path)
        reward_trainer.model.tokenizer.save_pretrained(folder_name_unique_path)
        args_dict = vars(args)
        # Save the arguments to a file (e.g., args.json)
        with open(folder_name_unique_path + "/args.json", "w") as f:
            json.dump(args_dict, f, indent=4)
        # does not work push to hub
        # reward_trainer.trainer.push_to_hub(name)
        # reward_trainer.tokenizer.push_to_hub(name)
        results = reward_trainer.eval_model_v2()
        with open(folder_name_unique_path + "/results_dataset_test.json", "w") as f:
            json.dump(results, f, indent=4)
        print("results_dataset_test", results)


    else:
        results = reward_trainer.eval_model(folder_name=folder_name_path)
        with open(folder_name_path + "/results_dataset_test.json", "w") as f:
            json.dump(results, f, indent=4)

