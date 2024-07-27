import os
import sys

ROOT_FOLDER = os.path.abspath(__file__)
for _ in range(2):
    ROOT_FOLDER = os.path.dirname(ROOT_FOLDER)
sys.path.append(ROOT_FOLDER)
from typing import List
from strictfire import StrictFire
from tqdm import tqdm
from typing import List, Tuple
import json
import random
import secrets

OUTPUT_FOLDER = os.path.join(ROOT_FOLDER, "data", "nli")
DATA_FOLDER = os.path.join(ROOT_FOLDER, "raw_data", "nli")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def wiki_nli(buckets: List[int], seed: int, one_shot: bool, n_examples: int):
    # topic to List of paragraphs
    text_dict = {}
    folder = os.path.join(DATA_FOLDER, "wikitext-2-raw")
    with open(os.path.join(folder, "wiki.test.raw")) as f:
        topic = ""
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line[0] + line[-1] == "==" and line.count("=") == 2:
                # topic
                assert line not in text_dict
                topic = line
                text_dict[topic] = []
                continue
            if line[0] + line[-1] == "==" and line.count("=") > 2:
                # some section name
                continue
            if len(line.split()) > 30:
                text_dict[topic].append(line)
    topics = list(text_dict.keys())
    for topic in topics:
        if len(text_dict[topic]) == 0:
            text_dict.pop(topic)

    instruction = (
        "You are given a query paragraph and multiple candidate paragraphs."
        " One candidate paragraph is the continuation of the query paragraph."
        " You need to return the paragraph ID of the candidate paragraph that"
        " continues the query paragraph."
    )
    example = """Query Paragraph: In 2006 , Boulter starred alongside Whishaw in the play Citizenship written by Mark Ravenhill .
Candidate Paragraph C29BD6: The An Lushan Rebellion began in December 755 , and was not completely suppressed for almost eight years .
Candidate Paragraph 37B155: He appeared on a 2006 episode of the television series , Doctors , followed by a role in the 2007 theatre production of How to Curse directed by Josie Rourke .
Answer: 37B155."""
    seeder = random.Random(seed)
    pbar = tqdm(desc="Processing NLI", total=len(buckets) * n_examples)
    topics = list(text_dict.keys())
    prev_bucket = -1000
    out_data = []
    for bucket in buckets:
        n_paras = -(-bucket // 230)
        indexes = [0] + list(range(4, n_paras, 5))
        data = []
        while len(data) < n_examples:
            for gold_index in indexes:
                trial = []
                while True:
                    input_str = []
                    chosen_topics = seeder.choices(topics, k=n_paras + 1)
                    for i, gt_topic in enumerate(chosen_topics):
                        if chosen_topics.count(gt_topic) == 1:
                            break
                    # first one is query
                    paras = text_dict[gt_topic]
                    i = random.randint(0, len(paras) - 2)
                    query_para = paras[i]
                    input_str.append(f"Query Paragraph: {query_para}")
                    ans = secrets.token_hex(3).upper()
                    ans_para = f"Candidate Paragraph {ans}: {paras[i+1]}"
                    candidate_paras = [
                        f"Candidate Paragraph {secrets.token_hex(3).upper()}:"
                        f" {seeder.choice(text_dict[topic])}"
                        for topic in chosen_topics
                    ]
                    candidate_paras.insert(gold_index, ans_para)
                    input_str += candidate_paras

                    input_str = "\n".join(input_str + ["Answer:"])
                    if one_shot:
                        input_str = example + "\n\n" + input_str

                    input_l = len(input_str.split())
                    total_l = input_l + 1
                    if total_l < prev_bucket or input_l > bucket:
                        if len(trial) > 200:
                            raise ValueError(
                                f"Can't create data within the {prev_bucket} <"
                                f" {bucket}. The lengths are {trial}"
                            )
                        trial.append(total_l)
                        continue
                    data.append(
                        {
                            "instruction": instruction,
                            "input": input_str,
                            "answers": [ans],
                            "input_length": input_l,
                            "total_length": total_l,
                            "length_bucket": bucket,
                            "gold_index": gold_index,
                        }
                    )
                    break
                pbar.update(1)
        if len(data) > n_examples:
            data = seeder.sample(data, k=n_examples)
        out_data += data
        prev_bucket = bucket
    pbar.close()
    out_filename = os.path.join(OUTPUT_FOLDER, "wikitext-103.jsonl")
    with open(out_filename, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in out_data]
            )
        )
    print(
        f"Wikitext-103 data saved to {out_filename} with"
        f" {len(out_data)} samples."
    )


def zh_wiki_nli(
    buckets: List[int], seed: int, n_examples: int, one_shot: bool = True
):
    # topic to List of paragraphs
    folder = os.path.join(DATA_FOLDER, "wiki_zh")
    article_len = 140

    seeder = random.Random(seed)
    processed_data = []
    for subfolder in os.listdir(folder):
        for f in os.listdir(os.path.join(folder, subfolder)):
            p = os.path.join(folder, subfolder, f)
            data = [json.loads(line) for line in open(p)]
            for row in seeder.sample(data, k=20):
                row["text"] = row["text"].replace("\n\n", "\n")
                if len(row["text"]) < article_len * 2:
                    continue
                paragraphs = []
                tmp_para = ""
                for line in row["text"].split("\n"):
                    if tmp_para:
                        tmp_para += "\n"
                    tmp_para += line
                    if len(tmp_para) >= article_len:
                        paragraphs.append(tmp_para)
                        tmp_para = ""
                # if tmp_para:
                #     paragraphs.append(tmp_para)
                if len(paragraphs) < 2:
                    continue
                processed_data.append(
                    {
                        "path": p,
                        "paragraphs": paragraphs,
                        "title": row["title"],
                    }
                )

    instruction = (
        "请在以下数个候选段落中选取一个能正确衔接主段落的段落。"
        "你只需要提供该段落的编号为答案。"
    )
    example = """主段落：结城正美（，）是日本知名男性漫画家，本名为佐藤修治，日本北海道札幌人，而确切的出生地是在与札幌邻近的虻田郡倶知安町。
根据作品《机动警察》大然版书皮作者介绍处，结城正美念小学时因为长相较帅而时常遭到欺负，在漫画技巧的学习阶段时曾担任过另一名日本知名漫画家新谷薰（男性）的工作室助手。
候选段落C29BD6：脱烷基是和烷基化相反的一种化工单元过程，是从有机化合物分子中脱去烷基的单元过程，一般是脱去和碳原子链接的烷基。例如从甲苯中脱去甲基生成苯，从甲基萘中脱去甲基生成萘等过程。
候选段落37B155：结城正美的出道作品《The Rival，劲敌》是在1980年4月号的OUT漫画月刊（）中刊载，之后他边工作边画漫画，然而逐渐难以负荷同时性的两边工作量，因此选择工作而放弃漫画，不过在一年后他重新选择漫画而放弃工作，成为全职的漫画家。附带一提的是他个人坦承经常有拖稿（延误交稿时间）的表现。
答案：37B155。"""
    seeder = random.Random(seed)
    pbar = tqdm(desc="Processing ZH NLI", total=len(buckets) * n_examples)
    prev_bucket = -1000
    out_data = []
    for bucket in buckets:
        window_length = bucket - len(instruction)
        if one_shot:
            window_length -= len(example)
        n_paras = window_length // 240
        indexes = [0] + list(range(4, n_paras, 5))
        data = []
        while len(data) < n_examples:
            for gold_index in indexes:
                trial = []
                while True:
                    input_str = []
                    chosen_paragraphs = seeder.choices(
                        processed_data, k=n_paras
                    )
                    # first one is query
                    paras = chosen_paragraphs[0]["paragraphs"]
                    i = seeder.randint(0, len(paras) - 2)
                    query_para = paras[i]
                    input_str.append(f"主段落: {query_para.strip()}")
                    ans = secrets.token_hex(3).upper()
                    ans_para = f"候选段落{ans}: {paras[i+1]}"
                    candidate_paras = [
                        f"候选段落{secrets.token_hex(3).upper()}:"
                        f" {seeder.choice(para['paragraphs']).strip()}"
                        for para in chosen_paragraphs[1:]
                    ]
                    candidate_paras.insert(gold_index, ans_para)
                    input_str += candidate_paras
                    input_str = "\n".join(input_str + ["答案:"])
                    if one_shot:
                        input_str = example + "\n\n" + input_str
                    input_l = len(input_str)
                    total_l = input_l + 1
                    if total_l < prev_bucket or total_l > bucket:
                        if len(trial) > 200:
                            raise ValueError(
                                f"Can't create data within the {prev_bucket} <"
                                f" {bucket}. The lengths are {trial}"
                            )
                        trial.append(total_l)
                        continue
                    data.append(
                        {
                            "instruction": instruction,
                            "input": input_str,
                            "answers": [ans],
                            "input_length": input_l,
                            "total_length": total_l,
                            "length_bucket": bucket,
                            "gold_index": gold_index,
                        }
                    )
                    break
                pbar.update(1)
        if len(data) > n_examples:
            data = seeder.sample(data, k=n_examples)
        out_data += data
        prev_bucket = bucket
    pbar.close()
    out_filename = os.path.join(OUTPUT_FOLDER, "wiki2019zh.jsonl")
    with open(out_filename, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in out_data]
            )
        )
    print(
        f"Wiki2019zh data saved to {out_filename} with"
        f" {len(out_data)} samples."
    )


def main(buckets: List[int], one_shot: bool, seed: int, n_examples: int):
    buckets = sorted(buckets)
    wiki_nli(buckets, seed, one_shot=one_shot, n_examples=n_examples)
    zh_wiki_nli(buckets, seed, one_shot=one_shot, n_examples=n_examples)


if __name__ == "__main__":
    StrictFire(main)
