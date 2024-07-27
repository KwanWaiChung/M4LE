import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter("ignore", ResourceWarning)

import json
import os
import numpy as np
import warnings
import random
from tqdm import tqdm
from strictfire import StrictFire
from typing import List

ROOT_FOLDER = os.path.abspath(__file__)
for _ in range(2):
    ROOT_FOLDER = os.path.dirname(ROOT_FOLDER)


OUTPUT_FOLDER = os.path.join(
    ROOT_FOLDER,
    "data",
    "summarization",
)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def booksum(buckets: List[int], one_shot: bool, seed: int, n_examples: int):
    seeder = random.Random(seed)
    data_path = os.path.join(
        ROOT_FOLDER,
        "booksum/alignments/paragraph-level-summary-alignments/chapter_summary_aligned_train_split.jsonl.gathered",
    )
    if one_shot:
        instruction = """Summarize the content extracted from a book.
Content: The committee had initially planned a series of hearings in 1972 on dietary fat, cholesterol, and heart disease, but the plans changed because McGovern ran for president. When the committee returned to the diet-and-chronic-disease issue after McGovern's defeat, the subject that seemed most urgent—thanks in part to the publication of John Yudkin's Sweet and Dangerous—was sugar in the diet, diabetes, and heart disease. 
Summary: In 1972, a committee planned hearings on dietary fat and heart disease, but pivoted due to McGovern's presidential run. Post-McGovern's defeat, they refocused on sugar's role in diet, diabetes, and heart disease. 
"""
    else:
        instruction = "Summarize the following extract from a book."

    data = {bucket: {} for bucket in buckets}

    with open(data_path, "r") as f:
        prev_bid = ""
        text = []
        summaries = []
        for line in tqdm(f, total=8315):
            line = json.loads(line)
            next_text_str = " ".join(line["text"])
            next_summary_str = " ".join(line["summary"])

            if line["bid"] != prev_bid or any(
                len(instruction.split())
                + len(" ".join(text).split())
                + len(" ".join(summaries).split())
                + len(next_text_str.split())
                + len(next_summary_str.split())
                > bucket
                for bucket in buckets
            ):
                if text:
                    text_str = "\n".join(text)
                    input_str = f"Content: {text_str}\nSummary:"
                    summary_str = " ".join(summaries)
                    bid = line["book_id"]
                    input_l = len((instruction + " " + input_str).split())
                    total_l = len(
                        (
                            instruction + " " + input_str + " " + summary_str
                        ).split()
                    )

                    for bucket in buckets:
                        if total_l < bucket:
                            if bid in data[bucket]:
                                if data[bucket][bid]["input"] != input_str:
                                    for tmp in range(10):
                                        _bid = bid + str(tmp)
                                        if _bid not in data[bucket]:
                                            break
                                    data[bucket][_bid] = {
                                        "instruction": instruction,
                                        "input": input_str,
                                        "answers": [summary_str],
                                        "input_length": input_l,
                                        "total_length": total_l,
                                        "book_id": line["book_id"],
                                        "length_bucket": bucket,
                                    }
                                else:
                                    data[bucket][bid]["answers"].append(
                                        summary_str
                                    )
                                    if (
                                        total_l
                                        > data[bucket][bid]["total_length"]
                                    ):
                                        data[bucket][bid][
                                            "total_length"
                                        ] = total_l
                            else:
                                data[bucket][bid] = {
                                    "instruction": instruction,
                                    "input": input_str,
                                    "answers": [summary_str],
                                    "input_length": input_l,
                                    "total_length": total_l,
                                    "book_id": line["book_id"],
                                    "length_bucket": bucket,
                                }
                text = []
                summaries = []
                prev_bid = line["bid"]

            text.append(next_text_str)
            summaries.append(next_summary_str)

    out_data = []
    for bucket in buckets:
        bucket_data = list(data[bucket].values())
        if len(bucket_data) == 0:
            print(f"Booksum data doesn't support length up to {bucket}.")
        if len(bucket_data) > n_examples:
            bucket_data = seeder.sample(bucket_data, k=n_examples)
        print(
            f"Booksum has {len(bucket_data)} samples within the"
            f" {bucket} bucket."
        )
        out_data += bucket_data

    out_filename = os.path.join(
        OUTPUT_FOLDER,
        f"booksum.jsonl",
    )
    with open(out_filename, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in out_data]
            )
        )
    print(
        f"Booksum retrieval data saved to {out_filename} with"
        f" {len(out_data)} samples."
    )


def main(buckets: List[int], one_shot: bool, seed: int, n_examples: int):
    booksum(
        buckets=buckets, one_shot=one_shot, seed=seed, n_examples=n_examples
    )


if __name__ == "__main__":
    StrictFire(main)
