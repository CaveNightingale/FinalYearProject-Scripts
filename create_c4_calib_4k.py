import os
import datasets
from datasets import load_dataset, Dataset
from itertools import islice

dataset = load_dataset(
    "allenai/c4",
    "en",
    split="train",
    streaming=True
)

subset = list(islice(dataset, 4096))

small = Dataset.from_list(subset)

small.save_to_disk("data/c4_calib_4k")

