# randomly create N topics with wow_data and inference with random string prompt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import secrets
import random
import json
import numpy as np
from strictfire import StrictFire
from opencc import OpenCC
from typing import List
from utils import int_to_cn, int_to_en

ROOT_FOLDER = os.path.abspath(__file__)
for _ in range(2):
    ROOT_FOLDER = os.path.dirname(ROOT_FOLDER)
DATA_FOLDER = os.path.join(ROOT_FOLDER, "raw_data")
OUTPUT_FOLDER = os.path.join(
    ROOT_FOLDER,
    "data",
    "topic_retrieval",
)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def en_topic_retrieval(
    buckets: List[int],
    n_topics: List[int],
    n_examples: int,
    article_length: int,
    seed: int,
    one_shot: bool,
):
    instruction = (
        "You are given multiple pairs of topic phrase and the related article."
        " Memorize the topic phrases. I will ask you to return one specific"
        " topic phrase at the end."
    )
    seeder = random.Random(seed)
    woz_data = json.load(
        open(os.path.join(DATA_FOLDER, "wow", "data.json"), "r")
    )
    # maps topic to passage
    pairs = {}
    for _data in woz_data:
        if _data["chosen_topic"] not in pairs:
            pairs[_data["chosen_topic"]] = "\n".join(
                _data["chosen_topic_passage"]
            )
    pairs = list(pairs.items())

    buckets = {bucket: [] for bucket in buckets}
    for n_topic in n_topics:
        td_bucket_count = {bucket: [] for bucket in buckets}
        n_topics2 = int_to_en(n_topic)
        if n_topic >= 200:
            gold_indexes = [0] + list(range(9, n_topic, 10))
        else:
            gold_indexes = [0] + list(range(4, n_topic, 5))
        data = []
        for gold_index in gold_indexes:
            for _ in range(-(-n_examples // len(gold_indexes))):
                keys = [secrets.token_hex(3).upper() for _ in range(n_topic)]
                prompts = []
                subpairs = seeder.choices(pairs, k=n_topic)
                for i in range(n_topic):
                    prompts.append(
                        f"Phrase {keys[i]} is {subpairs[i][0]}, the article"
                        " is as"
                        f" follow:\n{' '.join(subpairs[i][1].split()[:article_length])}"
                    )
                #             if n_shots == 1:
                #                 example_key = secrets.token_hex(3).upper()
                #                 _article = """LLaMA (Large Language Model Meta AI) is a family of large language models (LLMs), released by Meta AI starting in February 2023.
                # For the first version of LLaMa, four model sizes were trained: 7, 13, 33 and 65 billion parameters. LLaMA's developers reported that the 13B parameter model's performance on most NLP benchmarks exceeded that of the much larger GPT-3 (with 175B parameters) and that the largest model was competitive with state of the art models such as PaLM and Chinchilla. Whereas the most powerful LLMs have generally been accessible only through limited APIs (if at all), Meta released LLaMA's model weights to the research community under a noncommercial license. Within a week of LLaMA's release, its weights were leaked to the public on 4chan via BitTorrent.
                # In July 2023, Meta released several models as Llama 2, using 7, 13 and 70 billion parameters. """
                #                 prompts.append(
                #                     f"Phrase {example_key} is LLaMA, the article is as"
                #                     f" follow:\n{tokenizer.decode(tokenizer(_article)['input_ids'][:article_length])}"
                #                 )
                # prompt = f"Below are the wikipedia articles of {N2} phrases.\n\n" + "\n\n".join(prompts[:N]) + f"\n\nAmong the {N2} phrases mentioned previously, phrase {keys[0]} is"
                input = (
                    "Below are the wikipedia articles of"
                    f" {n_topics2} phrases.\n\n"
                    + "\n\n".join(prompts[:n_topic])
                    + f"\n\nAmong the {n_topics2} phrases mentioned previously"
                )
                if one_shot:
                    input += f", phrase {keys[-1]} is {subpairs[-1][0]}"
                input += f", phrase {keys[gold_index]} is"
                # f"phrase {keys[-1]} is {subpairs[-1][0]}"

                input_l = len((instruction + " " + input).split())
                answers = [subpairs[gold_index][0]]
                total_l = len(
                    (instruction + " " + input + " " + answers[0]).split()
                )
                for bucket in buckets:
                    if total_l < bucket:
                        input_example = {
                            "instruction": instruction,
                            "input": input,
                            "answers": answers,
                            "input_length": input_l,
                            "total_length": total_l,
                            "gold_index": gold_index,
                            "n_topics": n_topic,
                            "length_bucket": bucket,
                        }
                        buckets[bucket].append(input_example)
                        td_bucket_count[bucket].append(
                            input_example["total_length"]
                        )
                        break
        td_bucket_str = {
            b: f"{np.mean(rows):.0f}" for b, rows in td_bucket_count.items()
        }
        print(
            f"n_topics={n_topic} has the following length distribution:"
            f" {td_bucket_str}"
        )
    out_data = []
    out_filename = os.path.join(
        OUTPUT_FOLDER,
        f"wow.jsonl",
    )
    for bucket, data in buckets.items():
        if len(data) == 0:
            print(
                "English topic retrieval doesn't support length up to"
                f" {bucket}."
            )
            continue
        if len(data) > n_examples:
            data = seeder.sample(data, k=n_examples)
        print(
            f"English topic retrieval {len(data)} samples within the"
            f" {bucket} bucket."
        )
        out_data += data
    with open(out_filename, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in out_data]
            )
        )
    print(
        f"English topic retrieval data saved to {out_filename} with"
        f" {len(out_data)} samples."
    )


def cn_topic_retrieval(
    buckets: List[int],
    n_topics: List[int],
    n_examples: int,
    article_length: int,
    seed: int,
    one_shot: bool,
):
    instruction = (
        "我會給你多組主題短語和相關文章的配對。記住這些主題短語。"
        "最後我會要求你回答一個特定的主題短語。"
    )
    seeder = random.Random(seed)
    input_path = os.path.join(
        DATA_FOLDER,
        "qa",
        f"DRCD",
        f"DRCD_test.json",
    )
    drcd_data = json.load(open(input_path, "r"))
    # maps topic to passage
    pairs = {}
    for row in drcd_data["data"]:
        title = row["title"]
        paragraph: str = seeder.choice(row["paragraphs"])["context"]
        if title in pairs:
            print(f"{title} exists already")
            continue
        pairs[title] = paragraph
    pairs = list(pairs.items())

    buckets = {bucket: [] for bucket in buckets}
    for n_topic in n_topics:
        td_bucket_count = {bucket: [] for bucket in buckets}
        n_topics2 = int_to_cn(n_topic)
        if n_topic >= 200:
            gold_indexes = [0] + list(range(9, n_topic, 10))
        else:
            gold_indexes = [0] + list(range(4, n_topic, 5))
        data = []
        for gold_index in gold_indexes:
            for _ in range(-(-n_examples // len(gold_indexes))):
                keys = [secrets.token_hex(3).upper() for _ in range(n_topic)]
                prompts = []
                subpairs = seeder.choices(pairs, k=n_topic)
                for i in range(n_topic):
                    prompts.append(
                        f"{keys[i]}主題是{subpairs[i][0]}， 文章如下"
                        f":\n{subpairs[i][1][:article_length]}"
                    )
                # prompt = f"Below are the wikipedia articles of {N2} phrases.\n\n" + "\n\n".join(prompts[:N]) + f"\n\nAmong the {N2} phrases mentioned previously, phrase {keys[0]} is"
                input = (
                    f"以下是{n_topics2}個主題對應的維基百科文章。\n\n"
                    + "\n\n".join(prompts[:n_topic])
                    + f"\n\n在以上提及到的{n_topics2}個主題當中"
                )
                if one_shot:
                    input += f"，{keys[-1]}主題是{subpairs[-1][0]}"
                input += f"，{keys[gold_index]}主題是"

                answers = [subpairs[gold_index][0]]
                input_l = len(instruction + input)
                total_l = len(instruction + input + answers[0])

                for bucket in buckets:
                    if total_l < bucket:
                        input_example = {
                            "instruction": instruction,
                            "input": input,
                            "answers": answers,
                            "input_length": input_l,
                            "total_length": total_l,
                            "gold_index": gold_index,
                            "n_topics": n_topic,
                            "length_bucket": bucket,
                        }
                        buckets[bucket].append(input_example)
                        td_bucket_count[bucket].append(
                            input_example["total_length"]
                        )
                        break
        td_bucket_str = {
            b: f"{np.mean(rows):.0f}" for b, rows in td_bucket_count.items()
        }
        print(
            f"n_topics={n_topic} has the following length distribution:"
            f" {td_bucket_str}"
        )
    out_data = []
    out_filename = os.path.join(
        OUTPUT_FOLDER,
        f"drcd_explicit-single.jsonl",
    )
    for bucket, data in buckets.items():
        if len(data) == 0:
            print(
                "Chinese topic retrieval doesn't support length up to"
                f" {bucket}."
            )
            continue
        if len(data) > n_examples:
            data = seeder.sample(data, k=n_examples)
        print(
            f"Chinese topic retrieval {len(data)} samples within the"
            f" {bucket} bucket."
        )
        out_data += data
    with open(out_filename, "w") as f:
        f.write(
            cc.convert(
                "\n".join(
                    [json.dumps(row, ensure_ascii=False) for row in out_data]
                )
            )
        )
    print(
        f"Chinese topic retrieval data saved to {out_filename} with"
        f" {len(out_data)} samples."
    )


def main(
    buckets: List[int],
    n_examples: int,
    en_article_length: int = 110,
    cn_article_length: int = 110,
    seed: int = 111,
    one_shot: bool = False,
):
    en_n_topics_dict = {
        1000: 6,
        2000: 11,
        4000: 21,
        6000: 41,
        8000: 61,
        12000: 81,
        32000: 251,
        64000: 501,
        128000: 1001,
    }
    cn_n_topics_dict = {
        1000: 6,
        2000: 11,
        4000: 21,
        6000: 41,
        8000: 51,
        12000: 81,
        32000: 221,
        64000: 441,
        128000: 881,
    }

    en_n_topics = [en_n_topics_dict[b] for b in buckets]
    cn_n_topics = [cn_n_topics_dict[b] for b in buckets]
    en_topic_retrieval(
        buckets, en_n_topics, n_examples, en_article_length, seed, one_shot
    )
    cn_topic_retrieval(
        buckets, cn_n_topics, n_examples, cn_article_length, seed, one_shot
    )


if __name__ == "__main__":
    cc = OpenCC("t2s")
    StrictFire(main)
