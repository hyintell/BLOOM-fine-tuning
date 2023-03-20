import torch
import transformers
from transformers import BloomTokenizerFast, BloomForCausalLM, TrainingArguments
from datasets import load_dataset
from utils import ModifiedTrainer, tokenise_data, data_collator
from utils import ModelArguments, DataArguments


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_name = model_args.model_name_or_path
    tokeniser = BloomTokenizerFast.from_pretrained(
        f"{model_name}", add_prefix_space=True
    )
    model = BloomForCausalLM.from_pretrained(f"{model_name}").to(device)

    data_name = data_args.data_name_or_path
    dataset = load_dataset(data_name)
    input_ids = tokenise_data(dataset, tokeniser)

    model.gradient_checkpointing_enable()
    model.is_parallelizable = True
    model.model_parallel = True

    # train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=input_ids,
        args=training_args,
        data_collator=data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    main()