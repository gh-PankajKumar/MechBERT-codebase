from datasets import load_dataset
from pathlib import Path
from transformers import AutoTokenizer
import os



def get_training_corpus():
    """
    Retrieve training corpus segmented into chunks of 1000.

    Returns:
        generator: A generator yielding chunks of 1000 texts from the training corpus.

    """
    return (
        dataset["train"][i : i + 1000]["text"] for i in range(0, len(dataset["train"]), 1000)
    )


## Load variables from submission command:
train_file = os.environ["TRAIN_FILE"]
model_name = os.environ["MODEL_NAME"] 
vocab_size = int(os.environ["VOCAB_SIZE"])
out_name = os.environ["OUT_NAME"]


## Load corpus for training
dataset = load_dataset("text", data_files={"train": [train_file]})
training_corpus = get_training_corpus()


## Train tokenizer
old_tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, vocab_size)
tokenizer.save_pretrained(out_name)
