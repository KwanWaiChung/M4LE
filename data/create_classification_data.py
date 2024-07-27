import os
import sys

ROOT_FOLDER = os.path.abspath(__file__)
for _ in range(2):
    ROOT_FOLDER = os.path.dirname(ROOT_FOLDER)
sys.path.append(ROOT_FOLDER)
from typing import List
from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.error import HTTPError
from strictfire import StrictFire
from tqdm import tqdm
from typing import List, Tuple
from statistics import median
from utils import int_to_en
import json
import gzip
import random
import pandas as pd
import secrets
import numpy as np

DATA_FOLDER = os.path.join(
    ROOT_FOLDER,
    "raw_data",
)
OUTPUT_FOLDER = os.path.join(ROOT_FOLDER, "data", "classification")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def bigpatent(buckets: List[int], seed: int, n_examples: int = 1000):
    """
    - A (Human Necessities),
    - B (Performing Operations; Transporting),
    - C (Chemistry; Metallurgy),
    - D (Textiles; Paper),
    - E (Fixed Constructions),
    - F (Mechanical Engineering; Lightning; Heating; Weapons; Blasting),
    - G (Physics),
    - H (Electricity), and
    - Y (General tagging of new or cross-sectional technology)
    """
    classes = [
        "human necessities",
        "performing operations, transporting",
        "chemistry, metallurgy",
        "textiles, paper",
        "fixed constructions",
        "mechanical engineering, lightning, heating, weapons, blasting",
        "physics",
        "electricity",
        "general tagging of new or cross-sectional technology",
    ]
    instruction = (
        "Classify the patent document below into one of the following"
        " category by providing the numerical index and the class label. "
        + "1: human necessities. 2: performing operations, transporting. 3:"
        " chemistry, metallurgy. 4: textiles, paper. 5: fixed constructions."
        " 6: mechanical engineering, lightning, heating, weapons, blasting."
        " 7: physics. 8: electricity. 9: general tagging of new or"
        " cross-sectional technology."
        + "\nDescription: to best understand the principles of the present"
        " invention , the following example is provided for illustrative"
        " purposes only . 25 g . of sodium aluminate ( na 2 al 2 o 4 ) is"
        " dissolved in a solution of 25 g . of triethanolamine and 75 g . of"
        " water . the resulting solution was clear and stable for a period"
        " of 30 days at which time it was used in the preparation of a"
        " gelled hydrocarbon . as is the case for all highly caustic"
        " solutions , the stabilized sodium aluminate solution of the"
        " present invention must be stored in such a manner as to avoid"
        " absorption of carbon dioxide , thus preventing the formation of"
        " carbonates , which result in an unstable solution . while this"
        " invention has been described in connection with a certain specific"
        " embodiment thereof , it is to be understood that this is by way of"
        " illustration and not by way of limitation ; and the scope of the"
        " appended claims should be construed as broadly as the prior art"
        " will permit .\nClass: 2: performing operations, transporting.\n"
    )
    test_folder = os.path.join(
        DATA_FOLDER, "summarization", "bigPatentData", "test"
    )
    data = [[] for _ in buckets]
    n_files = 0
    for folder in os.listdir(test_folder):
        for _ in os.listdir(os.path.join(test_folder, folder)):
            n_files += 1
    pbar = tqdm(total=n_files, desc="Processing bigpatent")
    seeder = random.Random(seed)
    for i, folder in enumerate(sorted(os.listdir(test_folder))):
        for file in os.listdir(os.path.join(test_folder, folder)):
            with gzip.open(os.path.join(test_folder, folder, file), "r") as f:
                for line in f:
                    line = json.loads(line)
                    input_str = f"Description: {line['description']}\nClass:"
                    input_l = len((instruction + " " + input_str).split())
                    total_l = len(
                        (instruction + " " + input_str + " " + str(i)).split()
                    )
                    for j, bucket in enumerate(buckets):
                        if total_l < bucket:
                            data[j].append(
                                {
                                    "instruction": instruction,
                                    "input": input_str,
                                    "answers": [str(i + 1)],
                                    "input_length": input_l,
                                    "total_length": total_l,
                                    "length_bucket": bucket,
                                }
                            )
                            break
            pbar.update(1)
    pbar.close()
    out_data = []
    out_filename = os.path.join(OUTPUT_FOLDER, "bigpatent_global_cls.jsonl")
    for j, bucket in enumerate(buckets):
        if len(data[j]) > n_examples:
            data[j] = seeder.sample(data[j], k=n_examples)
        if len(data[j]) > 0:
            out_data += data[j]
            print(
                f"Bigpatent data has {len(data[j])} samples within the"
                f" {bucket} bucket."
            )
        else:
            print(f"Bigpatent doesn't support length of {bucket}.")
    with open(out_filename, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in out_data]
            )
        )
    print(
        f"Bigpatent data saved to {out_filename} with {len(out_data)} samples."
    )


def mnds_news_retrieval(buckets: List[int], seed: int, n_examples: int):
    in_filename = os.path.join(
        DATA_FOLDER, "classification", "MN-DS-news-classification.csv"
    )
    df = pd.read_csv(in_filename)
    instruction = (
        "You are given multiple news articles below where each of them belongs"
        " to one of the 17 categories. Each article is prefixed with a article"
        " id. You will be asked to return the article ids of all articles"
        " belong to a particular category."
    )
    examples = [
        (
            (
                "Rarely do the worlds of art and science intersect, but they"
                " did with famed Dutch artist Escher.\nEven if you do not"
                " recognize his name, it is likely you have seen his work"
                " without knowing it.\nOne of the largest collections of his"
                " work is now on display in the US."
            ),
            "arts, culture, entertainment and media.",
            7526,  # dataid
        ),
        (
            (
                "On Sunday, NBC’s Meet The Press will air an interview with"
                " President Donald Trump, conducted by the network’s political"
                " director, Chuck Todd. While Todd’s interviews with 2020"
                " Democratic contenders have consisted largely of challenges"
                " from the left interspersed with the odd softball, Trump is"
                " unlikely to receive the same friendly treatment."
            ),
            "arts, culture, entertainment and media.",
            113188,
        ),
        (
            (
                "The full extent of the ferry disaster in the Iraqi city of"
                " Mosul is becoming clearer.\nCivil Defence says the number"
                " of dead is now at least 120, while 100 people are still"
                " missing.\nIraq's Prime Minister Adel Abdul Mahdi is"
                " formally requesting a local governor be sacked over the"
                " incident."
            ),
            "disaster, accident and emergency incident",
            "11210",
        ),
    ]
    example_prompt = []
    article_ids = [secrets.token_hex(3).upper() for _ in examples]
    for i, example in enumerate(examples):
        example_prompt.append(
            f"The article id is {article_ids[i]}. {example[0]}"
        )
    example_prompt = "\n\n".join(example_prompt)
    example_prompt += (
        "\nQuestion: Provide me the article id of all the news articles"
        " related to 'arts, culture, entertainment and media.'.\nAnswer:"
        f" {', '.join(article_ids[:2])}."
    )

    df["length"] = df["content"].map(lambda x: len(x.split()))
    seeder = random.Random(seed)
    pbar = tqdm(
        desc="Processing mnds_news retrieval", total=len(buckets) * n_examples
    )
    prev_bucket = 0
    n_docs = {16000: 30, 32000: 60, 64000: 110, 128000: 220}
    out_data = []
    for bucket in buckets:
        data = []
        if bucket in n_docs:
            n_doc = n_docs[bucket]
        else:
            n_doc: int = int(-(-bucket // df["length"].median()))
        for _ in range(n_examples):
            trial = []
            while True:
                article_ids = [
                    secrets.token_hex(3).upper() for _ in range(n_doc)
                ]
                rows = df.sample(
                    n=n_doc, random_state=seeder.randint(0, 999999999)
                )
                input_str = []
                class_map = {}
                for i, (_, row) in enumerate(rows.iterrows()):
                    _content = row["content"]
                    while "\n\n" in _content:
                        _content = _content.replace("\n\n", "\n")
                    input_str.append(
                        f"The article id {article_ids[i]}. {_content}"
                    )
                    class_map.setdefault(row["category_level_1"], []).append(
                        article_ids[i]
                    )
                input_str = "\n\n".join(input_str)
                class_map2 = {k: v for k, v in class_map.items() if len(v) > 1}
                if class_map2:
                    class_target = seeder.choice(list(class_map2.keys()))
                else:
                    class_target = seeder.choice(list(class_map.keys()))
                input_str += (
                    "\nQuestion: Provide me the article id of all the news"
                    f" articles related to '{class_target}'.\nAnswer:"
                )
                input_str = example_prompt + "\n\n" + input_str
                answer = ", ".join(class_map[class_target])
                input_l = len((instruction + " " + input_str).split())
                total_l = len((instruction + " " + input_str + answer).split())
                if total_l > prev_bucket and total_l <= bucket:
                    data.append(
                        {
                            "instruction": instruction,
                            "input": input_str,
                            "answers": [answer],
                            "input_length": input_l,
                            "total_length": total_l,
                            "length_bucket": bucket,
                            "labels": class_map,
                            "n_para": n_doc,
                        }
                    )
                    break
                else:
                    if len(trial) > 200:
                        raise ValueError(
                            f"Can't create data within the {prev_bucket} <"
                            f" {bucket}. The lengths are {trial}"
                        )
                    trial.append(total_l)
            pbar.update(1)
        prev_bucket = bucket
        if len(data) > n_examples:
            data = seeder.sample(data, k=n_examples)
        out_data += data
    pbar.close()

    out_filename = os.path.join(
        OUTPUT_FOLDER,
        "mnds-news_explicit-multiple.jsonl",
    )
    with open(out_filename, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in out_data]
            )
        )
    print(
        f"MNDS news retrieval data saved to {out_filename} with"
        f" {len(out_data)} samples."
    )


def marc(buckets: List[int], seed: int, n_examples: int):
    folder = os.path.join(DATA_FOLDER, "classification", "marc")
    # en
    en_d = [
        json.loads(l)
        for l in open(os.path.join(folder, "dataset_en_test.json"))
    ]
    zh_d = [
        json.loads(l)
        for l in open(os.path.join(folder, "dataset_zh_test.json"))
    ]

    # 1-stars and 5-stars
    en_d = [row for row in en_d if row["stars"] in [1, 5]]
    zh_d = [row for row in zh_d if row["stars"] in [1, 5]]
    d = en_d + zh_d
    seeder = random.Random(seed)

    instruction = (
        "You are given multiple customer reviews in Chinese or English below."
        " Please provide the review IDs of all the positive reviews."
    )
    example = """Review 2BCC61: Defective. Does not work. It does not work. When i plugged in it shows the status light but it does not work. very disappointed with the purchase.
Review F02EBB: 这简直就是太差了！出版社怎么就能出版吗？我以为是百度摘录呢！这到底是哪个鱼目混珠的教授啊？！能给点干货吗？！总算应验了一句话，一本书哪怕只有一句花你感到有意义也算是本好书。哇为了找这本书哪怕一句不是废话的句子都费了我整整一天时间。。
Review 7BD363: The light is very helpful for dark colored yarns.Very comfortable to use.
Review F6F80F: 开本，用纸，画风都太棒了。世图这个版本是良心之作。希望引进更多，省得我费心力淘原版了。后面附有的星球日报报纸也挺有趣。
Positive Reviews: 7BD363,F6F80F."""

    example_length = len(
        (
            "Review 2BCC61: Defective. Does not work. It does not work. When i"
            " plugged in it shows the status light but it does not work. very"
            " disappointed with the purchase. Review 7BD363: The light is very"
            " helpful for dark colored yarns.Very comfortable to use."
        ).split()
    )
    example_length += len(
        "Review F02EBB:"
        " 这简直就是太差了！出版社怎么就能出版吗？我以为是百度摘录呢！这到底是哪个鱼目混珠的教授啊？！能给点干货吗？！总算应验了一句话，"
        "一本书哪怕只有一句花你感到有意义也算是本好书。"
        "哇为了找这本书哪怕一句不是废话的句子都费了我整整一天时间。。Review"
        " F6F80F:"
        " 开本，用纸，画风都太棒了。世图这个版本是良心之作。希望引进更多，"
        "省得我费心力淘原版了。后面附有的星球日报报纸也挺有趣。"
    )
    pbar = tqdm(desc="Processing marc", total=len(buckets) * n_examples)
    prev_bucket = -1000
    all_data = []
    for bucket in buckets:
        n_reviews: int = -(-bucket // 55)
        data = []

        for _ in range(n_examples):
            trials = []
            while True:
                rows = seeder.sample(d, k=n_reviews)
                reviews = []
                # class_count = {1: 0, 5: 0}
                input_l = len(instruction.split()) + example_length
                answers = []
                for row in rows:
                    review_id: str = secrets.token_hex(3).upper()
                    if row["language"] == "en":
                        text = row["review_title"] + " " + row["review_body"]
                        input_l += len(text.split())
                    elif row["language"] == "zh":
                        text = row["review_title"] + row["review_body"]
                        input_l += len(text)
                    reviews.append(f"Review {review_id}: {text}")
                    if row["stars"] == 5:
                        answers.append(review_id)

                    # class_count[row["stars"]] += 1
                if input_l < prev_bucket or input_l > bucket:
                    if len(trials) > 100:
                        raise ValueError(
                            f"Can't create data within the {prev_bucket} <"
                            f" {bucket}. The lengths are {trials}"
                        )
                    trials.append(input_l)
                    continue
                input_str = "\n".join(reviews)
                input_str += "\nPositive Reviews:"
                input_str = example + "\n\n" + input_str
                answer = ",".join(answers) + "."
                data.append(
                    {
                        "instruction": instruction,
                        "input": input_str,
                        "answers": [answer],
                        "input_length": input_l,
                        "total_length": input_l + len(answers),
                        "length_bucket": bucket,
                    }
                )
                break
            pbar.update(1)
        if len(data) > n_examples:
            data = seeder.sample(data, k=n_examples)
        all_data += data
    pbar.close()
    out_filename = os.path.join(
        OUTPUT_FOLDER,
        f"marc.jsonl",
    )
    with open(out_filename, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in all_data]
            )
        )
    print(f"Marc data saved to {out_filename} with {len(all_data)} samples.")


def online_shopping(
    buckets: List[int], seed: int, n_examples: int, one_shot: bool = True
):
    in_filename = os.path.join(
        DATA_FOLDER,
        "classification/online_shopping_10_cats/online_shopping_10_cats.csv",
    )
    df = pd.read_csv(in_filename)
    d = df.to_dict("records")
    seeder = random.Random(seed)
    examples_idx = [18879, 41707, 32206]
    for i in examples_idx:
        d.pop(i)
    d = [row for row in d if type(row["review"]) == str]
    instruction = "以下是多个客户的评论。请提供所有正面评价的评论编号。"
    example = """评论2BCC61: 送货快服务好，包装严格大小均匀味道不错值得购买。 
评论F02EBB: 东西很不错，穿起来很合身舒服 
评论7BD363: 不好用 用完头屑很多 头痒 差评
答案: 2BCC61，F02EBB。"""

    example_length = len(example)
    pbar = tqdm(
        desc="Processing online shopping", total=len(buckets) * n_examples
    )
    prev_bucket = -1000
    n_docs = {128000: 1800}
    all_data = []
    for bucket in buckets:
        if bucket in n_docs:
            n_reviews: int = n_docs[bucket]
        else:
            n_reviews: int = -(-bucket // 70)
        data = []

        for _ in range(n_examples):
            trials = []
            while True:
                rows = seeder.sample(d, k=n_reviews)
                reviews = []
                # class_count = {1: 0, 5: 0}
                input_l = len(instruction.split()) + example_length
                answers = []
                for row in rows:
                    review_id: str = secrets.token_hex(3).upper()
                    reviews.append(f"评论{review_id}: {row['review']}")
                    if row["label"] == 1:
                        answers.append(review_id)

                    # class_count[row["stars"]] += 1
                input_str = "\n".join(reviews) + "\n答案:"
                answer = "，".join(answers) + "。"
                if one_shot:
                    input_str = example + "\n\n" + input_str
                input_l = len(input_str)
                total_l = input_l + len(answer)
                if total_l < prev_bucket or total_l > bucket:
                    if len(trials) > 100:
                        raise ValueError(
                            f"Can't create data within the {prev_bucket} <"
                            f" {bucket}. The lengths are {trials}"
                        )
                    trials.append(total_l)
                    continue
                data.append(
                    {
                        "instruction": instruction,
                        "input": input_str,
                        "answers": [answer],
                        "input_length": input_l,
                        "total_length": input_l + len(answers),
                        "length_bucket": bucket,
                    }
                )
                break
            pbar.update(1)
        if len(data) > n_examples:
            data = seeder.sample(data, k=n_examples)
        all_data += data
        prev_bucket = bucket
    pbar.close()
    out_filename = os.path.join(
        OUTPUT_FOLDER,
        f"online-shopping.jsonl",
    )
    with open(out_filename, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in all_data]
            )
        )
    print(
        f"Online shopping data saved to {out_filename} with"
        f" {len(all_data)} samples."
    )


def thucnews_classify_one(
    buckets: List[int], seed: int, n_examples: int, one_shot: bool = True
):
    in_folder = os.path.join(DATA_FOLDER, "classification", "THUCNews")
    seeder = random.Random(seed)
    data = []
    for label in os.listdir(in_folder):
        filenames = os.listdir(os.path.join(in_folder, label))
        filenames = seeder.sample(filenames, k=100)
        for filename in filenames:
            lines = (
                open(os.path.join(in_folder, label, filename))
                .read()
                .splitlines()
            )
            line = " ".join([l.strip().replace("\u3000", "") for l in lines])
            if len(line) > 20:
                data.append((line, label))

    instruction = (
        "请细阅读以下几篇新闻。每篇新闻都属于其中一个类别:体育、娱乐、家居、"
        "彩票、房产、教育、时尚、时政、星座、游戏、社会、科技、股票、财经。"
        "最后我会请您返回其中一篇新闻的类别。"
    )
    example = """新闻FE6806：NBA漫画-身在江湖姚明有绝招穿只泰哥派防弹衣新浪体育讯点击欣赏大嘴NBA漫画博客
新闻43B294：图文-英超20强升班马巡礼诺维奇球员庆祝升入英超 新浪体育讯图文为诺维奇球员庆祝升入英超。
新闻4F2B51：组图：陈豪助阵商业活动 表现羞涩不敢牵女友手 组图：陈豪助阵商业活动 羞涩不敢牵女友手
问题：请返回新闻FE6806的类别。
回答：体育。"""

    pbar = tqdm(
        desc="Processing thuc_news classify one",
        total=len(buckets) * n_examples,
    )
    median_len = median([len(l[0]) for l in data])
    prev_bucket = 0
    n_docs = {16000: 15, 32000: 30, 64000: 60, 128000: 120}
    all_data = []
    for bucket in buckets:
        out_data = []
        while len(out_data) < n_examples:
            if bucket in n_docs:
                n_doc = n_docs[bucket]
            else:
                n_doc: int = int(-(-bucket // median_len))
            gold_indexes = [0] + list(range(4, n_doc, 5))
            for gold_index in gold_indexes:
                trials = []
                while True:
                    article_ids = [
                        secrets.token_hex(3).upper() for _ in range(n_doc)
                    ]
                    rows = seeder.sample(data, k=n_doc)
                    input_str = []
                    for i, row in enumerate(rows):
                        input_str.append(f"新闻{article_ids[i]}：{row[0]}")
                    input_str = "\n".join(input_str)
                    input_str += (
                        f"\n问题：请返回新闻{article_ids[gold_index]}的类别。\n回答："
                    )
                    if one_shot:
                        input_str = example + "\n\n" + input_str
                    answer = rows[gold_index][1]
                    input_l = len((instruction + " " + input_str))
                    total_l = len((instruction + " " + input_str + answer))
                    if total_l < prev_bucket or total_l > bucket:
                        if len(trials) > 100:
                            raise ValueError(
                                f"Can't create data within the {prev_bucket} <"
                                f" {bucket}. The lengths are {trials}"
                            )
                        trials.append(total_l)
                        continue
                    out_data.append(
                        {
                            "instruction": instruction,
                            "input": input_str,
                            "answers": [answer],
                            "input_length": input_l,
                            "total_length": total_l,
                            "length_bucket": bucket,
                            "n_para": n_doc,
                            "gold_index": gold_index,
                        }
                    )
                    break
                pbar.update(1)
        prev_bucket = bucket
        if len(out_data) > n_examples:
            out_data = seeder.sample(out_data, k=n_examples)
        all_data += out_data
    pbar.close()
    out_filename = os.path.join(
        OUTPUT_FOLDER,
        "thucnews_explicit-single.jsonl",
    )
    with open(out_filename, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in all_data]
            )
        )
    print(
        f"Thucnews classify one data saved to {out_filename} with"
        f" {len(all_data)} samples."
    )


def thucnews_retrieval(
    buckets: List[int], seed: int, n_examples: int, one_shot: bool = True
):
    in_folder = os.path.join(DATA_FOLDER, "classification", "THUCNews")
    seeder = random.Random(seed)
    data = []
    for label in os.listdir(in_folder):
        filenames = os.listdir(os.path.join(in_folder, label))
        filenames = seeder.sample(filenames, k=100)
        for filename in filenames:
            lines = (
                open(os.path.join(in_folder, label, filename))
                .read()
                .splitlines()
            )
            line = " ".join([l.strip().replace("\u3000", "") for l in lines])
            if len(line) > 20:
                data.append((line, label))

    instruction = (
        "请细阅以下几篇新闻。最后我会请你返回所有属于某类别的新闻编号。"
    )
    example = """新闻 FE6806：NBA漫画-身在江湖姚明有绝招只穿泰哥派防弹衣新浪体育讯点击欣赏大嘴NBA漫画博客
新闻 43B294：图文-英超20强升班马巡礼诺维奇球员庆祝升入英超 新浪体育讯图为诺维奇球员庆祝升入英超。
新闻 4F2B51：组图：陈豪助阵商业活动 表现羞涩不敢牵女友手 组图：陈豪助阵商业活动 羞涩不敢牵女方手
问题：请返回所有体育新闻的编号。
回答：FE6806，43B294。"""

    pbar = tqdm(
        desc="Processing thuc_news retrieval", total=len(buckets) * n_examples
    )
    median_len = median([len(l[0]) for l in data])
    prev_bucket = 0

    n_docs = {16000: 15, 32000: 30, 64000: 60, 128000: 120}
    all_data = []
    for bucket in buckets:
        out_data = []
        if bucket in n_docs:
            n_doc = n_docs[bucket]
        else:
            n_doc: int = int(-(-bucket // median_len))
        for _ in range(n_examples):
            trials = []
            while True:
                article_ids = [
                    secrets.token_hex(3).upper() for _ in range(n_doc)
                ]
                rows = seeder.sample(data, k=n_doc)
                input_str = []
                class_map = {}
                for i, row in enumerate(rows):
                    input_str.append(f"新闻 {article_ids[i]}：{row[0]}")
                    class_map.setdefault(row[1], []).append(article_ids[i])
                input_str = "\n".join(input_str)
                class_map2 = {k: v for k, v in class_map.items() if len(v) > 1}
                if class_map2:
                    class_target = seeder.choice(list(class_map2.keys()))
                else:
                    class_target = seeder.choice(list(class_map.keys()))
                input_str += (
                    f"\n问题：请返回所有{class_target}新闻的编号。\n回答："
                )
                if one_shot:
                    input_str = example + "\n\n" + input_str
                answer = "，".join(class_map[class_target])
                input_l = len((instruction + " " + input_str))
                total_l = len((instruction + " " + input_str + answer))
                if total_l < prev_bucket or total_l > bucket:
                    if len(trials) > 100:
                        raise ValueError(
                            f"Can't create data within the {prev_bucket} <"
                            f" {bucket}. The lengths are {trials}"
                        )
                    trials.append(total_l)
                    continue
                out_data.append(
                    {
                        "instruction": instruction,
                        "input": input_str,
                        "answers": [answer],
                        "input_length": input_l,
                        "total_length": total_l,
                        "length_bucket": bucket,
                        "labels": class_map,
                        "n_para": n_doc,
                    }
                )
                break
            pbar.update(1)
        prev_bucket = bucket
        if len(out_data) > n_examples:
            out_data = seeder.sample(out_data, k=n_examples)
        all_data += out_data

    pbar.close()
    out_filename = os.path.join(
        OUTPUT_FOLDER,
        f"thucnews_explicit-multiple.jsonl",
    )
    with open(out_filename, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in all_data]
            )
        )
    print(
        f"Thucnews retrieval data saved to {out_filename} with"
        f" {len(all_data)} samples."
    )


def thucnews_count(
    buckets: List[int], seed: int, n_examples: int, one_shot: bool = True
):
    in_folder = os.path.join(DATA_FOLDER, "classification", "THUCNews")
    seeder = random.Random(seed)
    data = {}
    all_data = []
    for label in os.listdir(in_folder):
        filenames = os.listdir(os.path.join(in_folder, label))
        filenames = seeder.sample(filenames, k=100)
        for filename in filenames:
            lines = (
                open(os.path.join(in_folder, label, filename))
                .read()
                .splitlines()
            )
            line = " ".join([l.strip().replace("\u3000", "") for l in lines])
            if len(line) > 20:
                all_data.append((line, label))
                data.setdefault(label, []).append((line, label))

    instruction = (
        "请细阅以下几篇新闻。最后我会请你返回属于某类别的新闻的数目。"
    )
    example = """新闻：NBA漫画-身在江湖姚明有绝招只穿泰哥派防弹衣新浪体育讯点击欣赏大嘴NBA漫画博客
新闻：图文-英超20强升班马巡礼诺维奇球员庆祝升入英超 新浪体育讯图为诺维奇球员庆祝升入英超。
新闻：组图：陈豪助阵商业活动 表现羞涩不敢牵女友手 组图：陈豪助阵商业活动 羞涩不敢牵女方手
问题：以上有多少篇新闻是关于体育的？
回答：2。"""

    median_length = median(
        [len(d[0]) for d_list in data.values() for d in d_list]
    )
    pbar = tqdm(
        desc="Processing thucnews count", total=len(buckets) * n_examples
    )
    prev_bucket = 0
    n_docs = {16000: 15, 32000: 30, 64000: 60, 128000: 120}
    all_out_data = []
    for bucket in buckets:
        out_data = []
        if bucket in n_docs:
            n_doc = n_docs[bucket]
        else:
            n_doc: int = int(-(-bucket // median_length))

        # 2 <= max_class <= 6
        max_class = max(min(6, n_doc - 2), 2)
        n_class = 1
        for _ in range(n_examples):
            trials = []
            while True:
                class_target: str = seeder.choice(list(data.keys()))
                rows1 = seeder.sample(data[class_target], k=n_class)
                rows2 = seeder.sample(all_data, k=n_doc - n_class)
                if any([row[1] == class_target for row in rows2]):
                    continue
                rows = rows1 + rows2
                input_str = []
                class_count = {}
                seeder.shuffle(rows)
                for i, row in enumerate(rows):
                    _content = row[0]
                    input_str.append(f"新闻：{_content.strip()}")
                    class_count.setdefault(row[1], []).append(i)
                input_str = "\n".join(input_str)
                input_str += (
                    f"\n问题：以上有多少篇新闻是关于{class_target}的？\n回答："
                )
                if one_shot:
                    input_str = example + "\n\n" + input_str
                assert len(class_count[class_target]) == n_class
                answer_int = n_class
                answer = f"{answer_int}。"
                input_l = len((instruction + " " + input_str))
                total_l = len((instruction + " " + input_str + answer))
                if total_l < prev_bucket or total_l > bucket:
                    if len(trials) > 100:
                        raise ValueError(
                            f"Can't create data within the {prev_bucket} <"
                            f" {bucket}. The lengths are {trials}"
                        )
                    trials.append(total_l)
                    continue
                out_data.append(
                    {
                        "instruction": instruction,
                        "input": input_str,
                        "answers": [answer],
                        "input_length": input_l,
                        "total_length": total_l,
                        "length_bucket": bucket,
                        "labels": class_count,
                        "n_para": n_doc,
                    }
                )
                n_class = n_class % max_class + 1
                break
            pbar.update(1)
        prev_bucket = bucket
        if len(out_data) > n_examples:
            out_data = seeder.sample(out_data, k=n_examples)
        all_out_data += out_data
    pbar.close()
    out_filename = os.path.join(
        OUTPUT_FOLDER,
        "thucnews_semantic-multiple.jsonl",
    )
    with open(out_filename, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in all_out_data]
            )
        )
    print(
        f"Thucnews count data saved to {out_filename} with"
        f" {len(all_out_data)} samples."
    )


def mnds_news_count(buckets: List[int], seed: int, n_examples: int):
    in_filename = os.path.join(
        DATA_FOLDER, "classification", "MN-DS-news-classification.csv"
    )
    df = pd.read_csv(in_filename)
    instruction = (
        "You are given multiple news articles below where each of them belongs"
        " to one of the 17 categories. You will be asked to return the number"
        " of articles belong to a particular category."
    )
    examples = [
        (
            (
                "Rarely do the worlds of art and science intersect, but they"
                " did with famed Dutch artist Escher.\nEven if you do not"
                " recognize his name, it is likely you have seen his work"
                " without knowing it.\nOne of the largest collections of his"
                " work is now on display in the US."
            ),
            "arts, culture, entertainment and media.",
            7526,  # dataid
        ),
        (
            (
                "On Sunday, NBC’s Meet The Press will air an interview with"
                " President Donald Trump, conducted by the network’s political"
                " director, Chuck Todd. While Todd’s interviews with 2020"
                " Democratic contenders have consisted largely of challenges"
                " from the left interspersed with the odd softball, Trump is"
                " unlikely to receive the same friendly treatment."
            ),
            "arts, culture, entertainment and media.",
            113188,
        ),
        (
            (
                "The full extent of the ferry disaster in the Iraqi city of"
                " Mosul is becoming clearer.\nCivil Defence says the number"
                " of dead is now at least 120, while 100 people are still"
                " missing.\nIraq's Prime Minister Adel Abdul Mahdi is"
                " formally requesting a local governor be sacked over the"
                " incident."
            ),
            "disaster, accident and emergency incident",
            "11210",
        ),
    ]
    example_prompt = []
    for i, example in enumerate(examples):
        example_prompt.append(f"Article: {example[0]}")
    example_prompt = "\n".join(example_prompt)
    example_prompt += (
        f"\nQuestion: Provide me the number of articles"
        f" related to 'arts, culture, entertainment and media.'.\nAnswer:"
        f" 2."
    )

    df["length"] = df["content"].map(lambda x: len(x.split()))
    median_length = df["length"].median()
    df = df.to_dict("records")
    data = {}
    for i, row in enumerate(df):
        row["id"] = i
        data.setdefault(row["category_level_1"], []).append(row)

    seeder = random.Random(seed)
    pbar = tqdm(
        desc="Processing mnds_news count", total=len(buckets) * n_examples
    )
    prev_bucket = 0
    n_docs = {16000: 30, 32000: 60, 64000: 110, 128000: 210}
    all_data = []
    for bucket in buckets:
        out_data = []
        if bucket in n_docs:
            n_doc = n_docs[bucket]
        else:
            n_doc: int = int(-(-bucket // median_length))

        # 2 <= max_class <= 6
        max_class = max(min(6, n_doc - 2), 2)
        n_class = 1
        for _ in range(n_examples):
            trial = []
            while True:
                class_target: str = seeder.choice(list(data.keys()))
                rows1 = seeder.sample(data[class_target], k=n_class)
                rows2 = seeder.sample(df, k=n_doc - n_class)
                if any(
                    [row["category_level_1"] == class_target for row in rows2]
                ):
                    continue
                rows = rows1 + rows2
                input_str = []
                class_count = {}
                seeder.shuffle(rows)
                for i, row in enumerate(rows):
                    _content = row["content"]
                    while "\n\n" in _content:
                        _content = _content.replace("\n\n", "\n")
                    input_str.append(f"Article: {_content.strip()}")
                    class_count[row["category_level_1"]] = (
                        class_count.setdefault(row["category_level_1"], 0) + 1
                    )
                input_str = "\n".join(input_str)
                input_str += (
                    "\nQuestion: Provide me the number of"
                    f" articles related to '{class_target}'.\nAnswer:"
                )
                input_str = example_prompt + "\n\n" + input_str
                assert class_count[class_target] == n_class
                answer_int = n_class
                answer = f"{answer_int}."
                input_l = len((instruction + " " + input_str).split())
                total_l = len((instruction + " " + input_str + answer).split())
                if total_l > prev_bucket and total_l <= bucket:
                    out_data.append(
                        {
                            "instruction": instruction,
                            "input": input_str,
                            "answers": [answer],
                            "input_length": input_l,
                            "total_length": total_l,
                            "length_bucket": bucket,
                            "labels": class_count,
                            "n_para": n_doc,
                        }
                    )
                    n_class = n_class % max_class + 1
                    break
                else:
                    if len(trial) > 200:
                        raise ValueError(
                            f"Can't create data within the {prev_bucket} <"
                            f" {bucket}. The lengths are {trial}"
                        )
                    trial.append(total_l)
            pbar.update(1)
        prev_bucket = bucket
        if len(out_data) > n_examples:
            out_data = seeder.sample(out_data, k=n_examples)
        all_data += out_data
    pbar.close()
    out_filename = os.path.join(
        OUTPUT_FOLDER,
        f"mnds-news_semantic-multiple.jsonl",
    )
    with open(out_filename, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in all_data]
            )
        )
    print(
        f"MNDS news count data saved to {out_filename} with"
        f" {len(all_data)} samples."
    )


def mnds_news_classify_one(
    buckets: List[int], seed: int, n_examples: int, one_shot: bool = True
):
    in_filename = os.path.join(
        DATA_FOLDER, "classification", "MN-DS-news-classification.csv"
    )
    df = pd.read_csv(in_filename)
    df["length"] = df["content"].map(lambda x: len(x.split()))
    median_len = df["length"].median()
    data = df.to_dict("records")
    seeder = random.Random(seed)

    instruction = """You are given multiple news articles below. Each of them belongs to one of the following categories:
1. crime, law and justice
2. arts, culture, entertainment and media
3. economy, business and finance
4. disaster, accident and emergency incident
5. environment
6. education
7. health
8. human interest
9. lifestyle and leisure
10. politics
11. labour
12. religion and belief
13. science and technology
14. society
15. sport
16. conflict, war and peace
17. weather
You will be asked to return the category of a news article I specified at the end."""
    examples = [
        (
            (
                "Rarely do the worlds of art and science intersect, but they"
                " did with famed Dutch artist Escher.\nEven if you do not"
                " recognize his name, it is likely you have seen his work"
                " without knowing it.\nOne of the largest collections of his"
                " work is now on display in the US."
            ),
            "arts, culture, entertainment and media.",
            7526,  # dataid
        ),
        (
            (
                "On Sunday, NBC’s Meet The Press will air an interview with"
                " President Donald Trump, conducted by the network’s political"
                " director, Chuck Todd. While Todd’s interviews with 2020"
                " Democratic contenders have consisted largely of challenges"
                " from the left interspersed with the odd softball, Trump is"
                " unlikely to receive the same friendly treatment."
            ),
            "arts, culture, entertainment and media.",
            113188,
        ),
        (
            (
                "The full extent of the ferry disaster in the Iraqi city of"
                " Mosul is becoming clearer.\nCivil Defence says the number"
                " of dead is now at least 120, while 100 people are still"
                " missing.\nIraq's Prime Minister Adel Abdul Mahdi is"
                " formally requesting a local governor be sacked over the"
                " incident."
            ),
            "disaster, accident and emergency incident",
            "11210",
        ),
    ]
    example_prompt = []
    article_ids = [secrets.token_hex(3).upper() for _ in examples]
    for i, example in enumerate(examples):
        example_prompt.append(f"Article {article_ids[i]}: {example[0]}")
    example = "\n".join(example_prompt)
    example += (
        "\nQuestion: What is the category of article"
        f" {article_ids[0]}?\nAnswer: arts, culture, entertainment and media"
    )

    pbar = tqdm(
        desc="Processing mnds_news classify one",
        total=len(buckets) * n_examples,
    )
    prev_bucket = 0
    n_docs = {16000: 30, 32000: 60, 64000: 110, 128000: 210}
    all_data = []
    for bucket in buckets:
        out_data = []
        if bucket in n_docs:
            n_doc = n_docs[bucket]
        else:
            n_doc: int = int(-(-bucket // median_len))
        while len(out_data) < n_examples:
            gold_indexes = [0] + list(range(4, n_doc, 5))
            for gold_index in gold_indexes:
                trial = []
                while True:
                    article_ids = [
                        secrets.token_hex(3).upper() for _ in range(n_doc)
                    ]
                    rows = seeder.sample(data, k=n_doc)
                    input_str = []
                    for i, row in enumerate(rows):
                        _content = row["content"]
                        while "\n\n" in _content:
                            _content = _content.replace("\n\n", "\n")
                        input_str.append(
                            f"Article {article_ids[i]}: {_content.strip()}"
                        )
                    input_str = "\n".join(input_str)
                    input_str += (
                        "\nQuestion: What is the category of article"
                        f" {article_ids[gold_index]}?\nAnswer:"
                    )
                    if one_shot:
                        input_str = example + "\n\n" + input_str
                    answer = rows[gold_index]["category_level_1"]
                    input_l = len((instruction + " " + input_str).split())
                    total_l = len(
                        (instruction + " " + input_str + answer).split()
                    )

                    if total_l > prev_bucket and total_l <= bucket:
                        out_data.append(
                            {
                                "instruction": instruction,
                                "input": input_str,
                                "answers": [answer],
                                "input_length": input_l,
                                "total_length": total_l,
                                "length_bucket": bucket,
                                "n_para": n_doc,
                                "gold_index": gold_index,
                            }
                        )
                        break
                    else:
                        if len(trial) > 200:
                            raise ValueError(
                                f"Can't create data within the {prev_bucket} <"
                                f" {bucket}. The lengths are {trial}"
                            )
                        trial.append(total_l)
                pbar.update(1)
        prev_bucket = bucket
        if len(out_data) > n_examples:
            out_data = seeder.sample(out_data, k=n_examples)
        all_data += out_data
    pbar.close()

    out_filename = os.path.join(
        OUTPUT_FOLDER,
        f"mnds-news_explicit-single.jsonl",
    )
    with open(out_filename, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in all_data]
            )
        )
    print(
        f"MNDS news classify one data saved to {out_filename} with"
        f" {len(all_data)} samples."
    )


def main(buckets: List[int], seed: int, n_examples: int):
    buckets = sorted(buckets)
    bigpatent(buckets=buckets, seed=seed, n_examples=n_examples)
    mnds_news_retrieval(buckets, seed=seed, n_examples=n_examples)
    mnds_news_count(buckets, seed=seed, n_examples=n_examples)
    mnds_news_classify_one(buckets, seed=seed, n_examples=n_examples)
    thucnews_retrieval(buckets, seed=seed, n_examples=n_examples)
    thucnews_count(buckets, seed=seed, n_examples=n_examples)
    thucnews_classify_one(buckets, seed=seed, n_examples=n_examples)
    marc(buckets, seed, n_examples=n_examples)
    online_shopping(buckets, seed, n_examples=n_examples)


if __name__ == "__main__":
    StrictFire(main)
