import sys
import os

ROOT_FOLDER = os.path.abspath(__file__)
for _ in range(2):
    ROOT_FOLDER = os.path.dirname(ROOT_FOLDER)
sys.path.append(ROOT_FOLDER)
import json
import gzip
import pandas as pd
import re
import random
import secrets
from typing import List, Dict, Any
from strictfire import StrictFire
from tqdm import tqdm
from opencc import OpenCC
from statistics import median
from bs4 import BeautifulSoup

DATA_FOLDER = os.path.join(
    ROOT_FOLDER,
    "raw_data/summarization",
)
OUTPUT_FOLDER = os.path.join(ROOT_FOLDER, "data", "summarization")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def arxiv(buckets: List[int], seed: int, n_examples: int = 1000):
    instruction = "Write an abstract for the following acadamic paper."
    example = """Paper: we have recorded the flight of a fly during take off and landing using digital high speed photography . it is shown that the dynamics of flexible wings are different for these procedures . during this observation fly flew freely in a big box and it was not tethered .
Abstract: in this fluid dynamics video , we demonstrated take off and landing of a fly . the deformation of wings is in focus in this video ."""
    lines = (
        open(
            os.path.join(
                DATA_FOLDER,
                "arxiv-dataset",
                "test.txt",
            ),
            "r",
        )
        .read()
        .splitlines()
    )
    seeder = random.Random(seed)
    data = [[] for _ in buckets]
    for line in tqdm(lines, desc="Processing arxiv"):
        line = json.loads(line)
        answer = " ".join(
            [
                l.split("<S> ")[1].split(" </S>")[0].strip() if l else l
                for l in line["abstract_text"]
            ]
        )
        input_str = f"Paper: {' '.join(line['article_text'])}\nAbstract:"
        input_str = f"{example}\n\n{input_str}"
        input_l = len((instruction + " " + input_str).split())
        total_l = len((instruction + " " + input_str + " " + answer).split())
        for j, bucket in enumerate(buckets):
            if total_l < bucket:
                data[j].append(
                    {
                        "instruction": instruction,
                        "input": input_str,
                        "answers": [answer],
                        "input_length": input_l,
                        "total_length": input_l,
                        "length_bucket": bucket,
                    }
                )
                break
    out_data = []
    for j, bucket in enumerate(buckets):
        if len(data[j]) > n_examples:
            data[j] = seeder.sample(data[j], k=n_examples)
        if len(data[j]) > 0:
            out_data += data[j]
            print(
                f"Arxiv data has {len(data[j])} samples within the"
                f" {bucket} bucket."
            )
        else:
            print(f"Arxiv doesn't support length of {bucket}.")
    out_filename = os.path.join(OUTPUT_FOLDER, "arxiv.jsonl")
    with open(out_filename, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in out_data]
            )
        )
    print(f"Arxiv data saved to {out_filename} with {len(out_data)} samples.")


def bigpatent(buckets: List[int], seed: int, n_examples: int = 1000):
    instruction = "Summarize the following description of a patent."
    example = """Description: to best understand the principles of the present invention , the following example is provided for illustrative purposes only . 25 g . of sodium aluminate ( na 2 al 2 o 4 ) is dissolved in a solution of 25 g . of triethanolamine and 75 g . of water . the resulting solution was clear and stable for a period of 30 days at which time it was used in the preparation of a gelled hydrocarbon . as is the case for all highly caustic solutions , the stabilized sodium aluminate solution of the present invention must be stored in such a manner as to avoid absorption of carbon dioxide , thus preventing the formation of carbonates , which result in an unstable solution . while this invention has been described in connection with a certain specific embodiment thereof , it is to be understood that this is by way of illustration and not by way of limitation ; and the scope of the appended claims should be construed as broadly as the prior art will permit .
Summary: aqueous solutions of sodium aluminate are stabilized by the addition of triethanolamine ."""
    test_folder = os.path.join(DATA_FOLDER, "bigPatentData", "test")
    data = [[] for _ in buckets]
    n_files = 0
    seeder = random.Random(seed)
    for folder in os.listdir(test_folder):
        for file in os.listdir(os.path.join(test_folder, folder)):
            n_files += 1
    pbar = tqdm(total=n_files, desc="Processing bigpatent")
    for folder in os.listdir(test_folder):
        for file in os.listdir(os.path.join(test_folder, folder)):
            with gzip.open(os.path.join(test_folder, folder, file), "r") as f:
                for line in f:
                    line = json.loads(line)
                    input_str = f"Description: {line['description']}\nSummary:"
                    input_str = example + "\n\n" + input_str
                    answer = line["abstract"]
                    input_l = len((instruction + " " + input_str).split())
                    total_l = len(
                        (instruction + " " + input_str + " " + answer).split()
                    )
                    for j, bucket in enumerate(buckets):
                        if total_l < bucket:
                            data[j].append(
                                {
                                    "instruction": instruction,
                                    "input": input_str,
                                    "answers": [answer],
                                    "input_length": input_l,
                                    "total_length": total_l,
                                    "length_bucket": bucket,
                                }
                            )
                            break
                pbar.update(1)
    pbar.close()
    out_data = []
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
    out_filename = os.path.join(OUTPUT_FOLDER, "bigpatent_global_sum.jsonl")
    with open(out_filename, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in out_data]
            )
        )
    print(
        f"Bigpatent data saved to {out_filename} with {len(out_data)} samples."
    )


def pubmed(
    buckets: List[int],
    seed: int,
    one_shot: bool = True,
    n_examples: int = 1000,
):
    instruction = "Write an abstract for the following medical paper."
    example = """Paper: cd45.1 or thy1.1 congenic ot-1 tcr transgenic nave t cells were isolated using the cd8 untouched isolation kit ( miltenyi biotech ) plus biotin conjugated anti - cd44 . cells were transferred into c57/bl6 ( jackson laboratories ) mice 1 day prior to infection . recombinant listeria monocytogenes strains were generated that stably express chicken ovalbumin ( aa134387 ) containing either the native ligand siinfekl257264 or the apl listed in supp . 2 . the designation of the apl indicates the substituted amino acid and the position within the siinfekl epitope . a previously described cassette24 encoding for the expression of secreted ova was manipulated by site - directed mutagenesis to insert the apl . the cassettes were cloned into a vector ( ppl2 ) and stable listeria recombinants were made as described25 . spleens were digested with blendzyme 2 and dnase 1 ( both roche ) for 1 hour at 37c and mashed through a 100 m cell strainer ( becton dickinson ) to obtain single cell suspensions . flow cytometry , intracellular cytokine staining ( ics ) , and cfse labeling were performed using standard procedures . for ics , the cells were first stimulated at 37c for 30 min with peptide , then 7 m brefeldin was added and the incubation was continued for another 4.5 hours . for ccr7 staining , cells were incubated for 1 hour at 37c and then stained on ice with ccl19-igg fusion supernatant26 . to assess the level of non - specific binding of ccl19-igg to ot-1 , soluble ccl19 ( r&d systems ) was added to control samples . for immunofluorescent microscopy , 1020 m sections , were cut and stained using standard procedures and analysed on a zeiss 510 meta confocal microscope .
Abstract: following an infection , cd8 + t cells are activated and undergo a characteristic kinetic sequence of rapid expansion , subsequent contraction and formation of memory cells13 . the pool of nave t cell clones is diverse and contains cells bearing t cell antigen receptors ( tcr ) that differ in their affinity for the same antigen4,5 . how these differences in affinity impact the function and the response kinetics of individual t cell clones was previously unknown . here we show that during the in vivo response to microbial infection , even very weak tcr - ligand interactions are sufficient to activate nave t cells , induce rapid initial proliferation and generate effector and memory cells . the strength of the tcr - ligand interaction critically impacts when expansion stops , when the cells exit lymphoid organs and when contraction begins , i.e. strongly stimulated t cells contract and exit lymphoid organs later than do weakly stimulated cells . our data challenges the prevailing view that strong tcr ligation is a prerequisite for cd8 + t cell activation . instead , very weak interactions are sufficient for activation , but strong tcr ligation is required to sustain t cell expansion . we propose that in response to microbial challenge , t cell clones with a broad range of avidities for foreign ligands are initially recruited , and that the pool of t cells subsequently matures in affinity due to the more prolonged expansion of high affinity t cell clones ."""
    lines = (
        open(
            os.path.join(
                DATA_FOLDER,
                "pubmed-dataset",
                "test.txt",
            ),
            "r",
        )
        .read()
        .splitlines()
    )
    seeder = random.Random(seed)
    data = [[] for _ in buckets]
    for line in tqdm(lines, desc="Processing pubmed"):
        line = json.loads(line)
        answer = " ".join(
            [
                l.split("<S> ")[1].split(" </S>")[0].strip() if l else l
                for l in line["abstract_text"]
            ]
        )
        input_str = " ".join(line["article_text"])
        input_str = f"Paper: {input_str}\nAbstract:"
        if one_shot:
            input_str = example + "\n\n" + input_str
        input_l = len((instruction + " " + input_str).split())
        total_l = len((instruction + " " + input_str + " " + answer).split())

        for j, bucket in enumerate(buckets):
            if total_l < bucket:
                data[j].append(
                    {
                        "instruction": instruction,
                        "input": input_str,
                        "answers": [answer],
                        "input_length": input_l,
                        "total_length": total_l,
                        "length_bucket": bucket,
                    }
                )
                break
    out_data = []
    for j, bucket in enumerate(buckets):
        if len(data[j]) > n_examples:
            data[j] = seeder.sample(data[j], k=n_examples)
        if len(data[j]) > 0:
            out_data += data[j]
            print(
                f"Pubmed data has {len(data[j])} samples within the"
                f" {bucket} bucket."
            )
        else:
            print(f"Pubmed doesn't support length of {bucket}.")
    out_filename = os.path.join(OUTPUT_FOLDER, "pubmed.jsonl")
    with open(out_filename, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in out_data]
            )
        )
    print(f"Pubmed data saved to {out_filename} with {len(out_data)} samples.")


def ncls(buckets, seed, n_examples):
    instruction = (
        "You are provided with multiple articles below. You will be asked to"
        " write a Chinese summary for a particular article."
    )
    en_example = """Article 2A41C1: West Ham host Swansea in what is the final Saturday kick-off in the 112-year history of the famous Boleyn Ground. The Hammers will bring the curtain down for good, prior to their move to the Olympic Stadium, when they face Manchester United on Tuesday night. Sportsmail brings you the best pictures from what promises to be an emotional day for all those associated with the London club.
Article 25BBC5: (CNN) -- The Internet is captivated by the news that Amazon is considering a future system that uses unmanned aerial vehicles, or drones, to deliver packages. Some people spent hours examining the feasibility of such a plan and earnestly pondering the effects of weather, thieves and other factors. Others, of course, just cracked jokes. Twitter, as usual, had a field day with the idea. Here are some of the funniest comments, memes and parody "drone" accounts.
Chinese Summary of Article 2A41C1: 周六下午，西汉姆联在厄普顿公园倒数第二场比赛中主场迎战斯旺西·城。"""
    # Chinese Summary of Article 25BBC5: 开玩笑的人蜂拥到推特上取笑亚马逊的无人机交付计划。"""

    en_file = os.path.join(DATA_FOLDER, "NCLS-Data/EN2ZHSUM/EN2ZHSUM_test.txt")
    data = []
    with open(en_file) as f:
        read_article = False
        read_en_reference = False
        read_zh_reference = False
        en_reference = ""
        zh_reference = ""
        article = ""
        for line in f:
            line = line.strip()
            if "<Article" in line:
                read_article = True
            elif "<EN-REF" in line and not en_reference:
                read_en_reference = True
            elif (
                "<EN-REF" in line
                and "human-corrected" in line
                and en_reference
            ):
                read_en_reference = True
            elif "<ZH-REF" in line and not zh_reference:
                read_zh_reference = True
            elif (
                "<ZH-REF" in line
                and "human-corrected" in line
                and zh_reference
            ):
                read_zh_reference = True
            elif read_article:
                article = line
                read_article = False
            elif read_en_reference:
                en_reference = line
                read_en_reference = False
            elif read_zh_reference:
                zh_reference = line
                read_zh_reference = False
            elif "</doc>" in line:
                assert article != ""
                assert en_reference != ""
                assert zh_reference != ""
                en_len = len(en_reference.split())
                zh_len = len(zh_reference)
                article_len = len(article.split())
                data.append(
                    {
                        "article": article,
                        "en_summary": en_reference,
                        "zh_summary": zh_reference,
                        "article_len": article_len,
                        "en_len": en_len,
                        "zh_len": zh_len,
                    }
                )
                article = ""
                en_reference = ""
                zh_reference = ""

    seeder = random.Random(seed)
    pbar = tqdm(desc="Processing ncls", total=len(buckets) * n_examples)
    prev_bucket = -1000
    all_data = []
    for bucket in buckets:
        out_data = []
        n_article = -(-bucket // 800)
        gold_indexes = [0] + list(range(4, n_article, 5))
        while len(out_data) < n_examples:
            for gold_index in gold_indexes:
                trial = []
                while True:
                    sampled_articles: List[Dict[str, Any]] = seeder.sample(
                        data, k=n_article
                    )
                    keys = [
                        secrets.token_hex(3).upper() for _ in range(n_article)
                    ]
                    input_str = "\n".join(
                        [
                            f"Article {key}: {passage['article']}"
                            for key, passage in zip(keys, sampled_articles)
                        ]
                    )
                    input_str = (
                        en_example
                        + "\n\n"
                        + input_str
                        + f"\nChinese Summary of Article {keys[gold_index]}:"
                    )
                    answer = sampled_articles[gold_index]["zh_summary"]
                    input_l = len((instruction + " " + input_str).split())
                    total_l = input_l + len(answer)
                    if total_l > prev_bucket and total_l <= bucket:
                        out_data.append(
                            {
                                "instruction": instruction,
                                "input": input_str,
                                "answers": [answer],
                                "input_length": input_l,
                                "total_length": total_l,
                                "gold_index": gold_index,
                                "length_bucket": bucket,
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
        all_data += out_data
        prev_bucket = bucket
    out_filename = os.path.join(OUTPUT_FOLDER, "ncls.jsonl")
    with open(out_filename, "w") as f:
        f.write(
            cc.convert(
                "\n".join(
                    [json.dumps(row, ensure_ascii=False) for row in all_data]
                )
            )
        )
    print(f"NCLS data saved to {out_filename} with {len(all_data)} samples.")


def clts(
    buckets: List[int],
    seed: int,
    one_shot: bool = True,
    n_examples: int = 1000,
):
    # CLTS plus
    src_path = os.path.join(
        DATA_FOLDER,
        "clts/test.src",
    )
    tgt_path = os.path.join(
        DATA_FOLDER,
        "clts/test.tgt",
    )
    srcs = open(src_path).read().splitlines()
    tgts = open(tgt_path).read().splitlines()

    seeder = random.Random(seed)
    instruction = "请为下列新闻写一个摘要。"
    example = """新闻: 新华社华盛顿1月30日消息，当地时间1月30日上午，新一轮中美经贸高级别磋商开幕式在美国白宫艾森豪威尔行政办公楼举行，中共中央政治局委员、国务院副总理、中美全面经济对话中方牵头人刘鹤，美国贸易代表莱特希泽，财政部长姆努钦等出席。本轮磋商于30日至31日在华盛顿进行。（原题为《中美经贸高级别磋商在华盛顿开幕》）
摘要: 当地时间1月30日上午，新一轮中美高层经贸磋商开幕式在白宫艾森豪威尔行政办公楼举行。此轮谈判将于30日至31日在华盛顿举行。"""

    data = [[] for _ in buckets]
    for src, tgt in tqdm(zip(srcs, tgts), desc="processing clts"):
        article = "".join(src.split())
        summary = "".join(tgt.split())

        input_str = f"新闻: {article.strip()}\n摘要:"
        if one_shot:
            input_str = f"{example}\n\n{input_str}"
        input_l = len((instruction + input_str))
        total_l = len((instruction + input_str + summary))
        for j, bucket in enumerate(buckets):
            if total_l < bucket:
                data[j].append(
                    {
                        "instruction": instruction,
                        "input": input_str,
                        "answers": [summary],
                        "input_length": input_l,
                        "total_length": total_l,
                        "length_bucket": bucket,
                    }
                )
                break
    out_data = []
    for j, bucket in enumerate(buckets):
        if len(data[j]) > n_examples:
            data[j] = seeder.sample(data[j], k=n_examples)
        if len(data[j]) > 0:
            out_data += data[j]
            print(
                f"CLTS data has {len(data[j])} samples within the"
                f" {bucket} bucket."
            )
        else:
            print(f"CLTS data doesn't support length of {bucket}.")
    out_filename = os.path.join(OUTPUT_FOLDER, "clts.jsonl")
    with open(out_filename, "w") as f:
        f.write(
            cc.convert(
                "\n".join(
                    [json.dumps(row, ensure_ascii=False) for row in out_data]
                )
            )
        )
    print(f"CLTS data saved to {out_filename} with {len(out_data)} samples.")


def cnewsum(
    buckets: List[int], seed: int, one_shot: bool = True, n_examples=1000
):
    in_filename = os.path.join(
        DATA_FOLDER,
        "CNewSum_v2/final/test.simple.label.jsonl",
    )
    seeder = random.Random(seed)
    lines = [json.loads(l) for l in open(in_filename)]

    instruction = "请为下列新闻写一个摘要。"
    example = """新闻: 昨天上午,有市民发现一名男子在宝山淞滨路、同济路东侧的一个绿化公园内已经死亡,警方介绍:已经排除他杀。
摘要: 绿化公园内一男子死亡,被市民发现"""
    data = [[] for _ in buckets]
    for line in tqdm(lines, desc="cnewsum"):
        article = "".join(["".join(l.split()) for l in line["article"]])
        summary = "".join(["".join(l.split()) for l in line["summary"]])

        if len(article) < 60:
            continue
        input_str = f"新闻: {article.strip()}\n摘要:"
        if one_shot:
            input_str = f"{example}\n\n{input_str}"
        input_l = len((instruction + input_str))
        total_l = len((instruction + input_str + summary))
        for j, bucket in enumerate(buckets):
            if total_l < bucket:
                data[j].append(
                    {
                        "instruction": instruction,
                        "input": input_str,
                        "answers": [summary],
                        "input_length": input_l,
                        "total_length": total_l,
                        "length_bucket": bucket,
                    }
                )
                break
    out_data = []
    for j, bucket in enumerate(buckets):
        if len(data[j]) > n_examples:
            data[j] = seeder.sample(data[j], k=n_examples)
        if len(data[j]) > 0:
            out_data += data[j]
            print(
                f"Cnewsum data has {len(data[j])} samples within the"
                f" {bucket} bucket."
            )
        else:
            print(f"Cnewsum doesn't support length of {bucket}.")
    out_filename = os.path.join(OUTPUT_FOLDER, "cnewsum.jsonl")
    with open(out_filename, "w") as f:
        f.write(
            cc.convert(
                "\n".join(
                    [json.dumps(row, ensure_ascii=False) for row in out_data]
                )
            )
        )
    print(
        f"Cnewsum data saved to {out_filename} with {len(out_data)} samples."
    )


def cepsum(
    buckets: List[int],
    seed: int,
    n_examples: int = 1000,
    one_shot: bool = True,
):
    example = """产品01959A描述: 范思哲范瑟丝女士红色牛皮拉链长款钱包，材质展示，LOGO展示，结构展示，工艺展示，厚约2cm，拉链，开和方式，光滑，其他，厚薄指数，柔软指数，手感指数，适中，按扣，柔软，偏硬
产品E25387描述: 蔻驰奢侈品女士黑色皮革单肩手提包戴妃包，商品正面，商品展示，商品背面，商品侧面，产地，单肩手提包，商品货号，适用背法，皮革，菲律宾，包包皮质，品牌名称，内部结构，拉链暗袋/证件袋
产品01959A摘要: 复古魅力的红色相机包型，以经典亮泽标志点缀细腻牛皮革，时髦与功能性兼具。包身容量适中，以五金双拉链开合，收纳实用便捷。柔软皮革双手带，出街轻便大方。包底走线，牢固耐磨。"""
    instruction = "以下是多个产品的描述。我会在最后让你写一个产品的摘要。"
    in_filepath = os.path.join(DATA_FOLDER, "CEPSUM/train.jsonl")
    example_len = len(example)
    instruction_len = len(instruction)
    data = [json.loads(line) for line in open(in_filepath, "r")]
    processed_data = []
    for row in data:
        if len(row["source"]) < 60:
            continue
        processed_data.append(
            {
                "desc": row["source"],
                "summary": row["targets"],
            }
        )

    median_len = median([len(row["desc"]) for row in processed_data])
    seeder = random.Random(seed)
    prev_bucket = 0
    all_data = []
    for bucket in buckets:
        out_data = []
        if one_shot:
            n_doc = -(-(bucket - example_len - instruction_len) // median_len)
        else:
            n_doc = (bucket - instruction_len) // median_len
        n_doc = int(n_doc)
        pbar = tqdm(
            desc=f"Processing cepsum bucket={bucket}", total=n_examples
        )
        while len(out_data) < n_examples:
            gold_indexes = [0] + list(range(4, n_doc, 5))
            for gold_index in gold_indexes:
                trial = []
                while True:
                    docs = seeder.sample(processed_data, k=n_doc)
                    keys = [secrets.token_hex(3).upper() for _ in range(n_doc)]
                    input_str = "\n".join(
                        [
                            f"产品{key}描述: {doc['desc']}"
                            for key, doc in zip(keys, docs)
                        ]
                    )
                    input_str += f"\n产品{keys[gold_index]}摘要:"
                    answers: List[str] = docs[gold_index]["summary"]
                    if one_shot:
                        input_str = example + "\n\n" + input_str
                    input_l = len(input_str)
                    total_l = input_l + len(max(answers, key=len))
                    if total_l > prev_bucket and total_l <= bucket:
                        out_data.append(
                            {
                                "instruction": instruction,
                                "input": input_str,
                                "answers": answers,
                                "input_length": input_l,
                                "total_length": total_l,
                                "length_bucket": bucket,
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
        pbar.close()
        prev_bucket = bucket
        all_data += out_data

    out_filename = os.path.join(OUTPUT_FOLDER, "cepsum.jsonl")
    with open(out_filename, "w") as f:
        f.write(
            cc.convert(
                "\n".join(
                    [json.dumps(row, ensure_ascii=False) for row in all_data]
                )
            )
        )
    print(f"Cepsum data saved to {out_filename} with {len(all_data)} samples.")


def lcsts(
    buckets: List[int],
    seed: int,
    n_examples: int = 1000,
    one_shot: bool = True,
):
    example = """段落D2B24B: #专业课#要唱响中国，最关键的是快速、布局、谋划的商业魔式，并逐步攀登多层次的顶峰：最低级的策划叫“广告策划”；第二层叫“营销策划”；第三层叫“产品策划”；第四层叫“产业策划”；第五层叫商业社会最高明的策划——“商业魔式策划”。
段落F4B965: #乌鲁木齐身边事#三碑屯牛羊肉批发市场，作为乌鲁木齐唯一一家牛羊肉一级批发市场，春节后将搬迁至新建成的华凌清真牛羊肉批发市场。 位于卡子湾华凌畜牧基地的新市场——华凌清真牛羊肉批发市场室内面积9000平方米、拥有320个固定摊位。
问题: 请为段落F4B965写一篇摘要。
摘要: 乌鲁木齐三屯碑牛羊肉批发市场要搬迁了"""
    instruction = (
        "以下是多则新浪微博段落。我会在最后让你为其中一个段落写一个摘要。"
    )
    example_len = len(example)
    instruction_len = len(instruction)
    in_filepath = os.path.join(DATA_FOLDER, "lcsts/PART_II.txt")
    soup = BeautifulSoup(open(in_filepath).read(), features="html.parser")
    data = [
        {
            "summary": row.summary.text.strip('"\n '),
            "text": row.short_text.text.strip('"\n '),
        }
        for row in soup.find_all("doc")
        if row.human_label.text == "5"
    ]
    indexes = [
        i
        for i, row in enumerate(data)
        if row["summary"]
        in ["商业魔式之五层次策划理论", "乌鲁木齐三屯碑牛羊肉批发市场要搬迁了"]
    ]
    # example indices
    for i in indexes:
        data.pop(i)

    seeder = random.Random(seed)
    prev_bucket = 0
    all_data = []
    for bucket in buckets:
        out_data = []
        if one_shot:
            n_doc = (bucket - example_len - instruction_len) // 120
        else:
            n_doc = (bucket - instruction_len) // 120
        n_doc = int(n_doc)
        pbar = tqdm(desc=f"Processing lcsts bucket={bucket}", total=n_examples)
        while len(out_data) < n_examples:
            gold_indexes = [0] + list(range(4, n_doc, 5))
            for gold_index in gold_indexes:
                trial = []
                while True:
                    docs = seeder.sample(data, k=n_doc)
                    keys = [secrets.token_hex(3).upper() for _ in range(n_doc)]
                    input_str = "\n".join(
                        [
                            f"段落{key}: {doc['text']}"
                            for key, doc in zip(keys, docs)
                        ]
                    )
                    input_str += f"\n问题: 请为段落{keys[gold_index]}写一篇摘要。\n摘要:"
                    answers: str = docs[gold_index]["summary"]
                    if one_shot:
                        input_str = example + "\n\n" + input_str
                    input_l = len(input_str)
                    total_l = input_l + len(max(answers, key=len))
                    if total_l > prev_bucket and total_l <= bucket:
                        out_data.append(
                            {
                                "instruction": instruction,
                                "input": input_str,
                                "answers": [answers],
                                "input_length": input_l,
                                "total_length": total_l,
                                "length_bucket": bucket,
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
        pbar.close()
        prev_bucket = bucket
        all_data += out_data
    out_filename = os.path.join(OUTPUT_FOLDER, "lcsts.jsonl")
    with open(out_filename, "w") as f:
        f.write(
            cc.convert(
                "\n".join(
                    [json.dumps(row, ensure_ascii=False) for row in all_data]
                )
            )
        )
    print(f"lcsts data saved to {out_filename} with {len(all_data)} samples.")


def cnnnews(
    buckets: List[int],
    seed: int,
    n_examples: int = 1000,
    one_shot: bool = True,
):
    def _get_art_abs(story_file):
        """Get abstract (highlights) and article from a story file path."""
        # Based on https://github.com/abisee/cnn-dailymail/blob/master/
        #     make_datafiles.py

        lines = open(story_file).read().splitlines()
        END_TOKENS = [
            ".",
            "!",
            "?",
            "...",
            "'",
            "`",
            '"',
            "\u2019",
            "\u201d",
            ")",
        ]

        # The github code lowercase the text and we removed it in 3.0.0.

        # Put periods on the ends of lines that are missing them
        # (this is a problem in the dataset because many image captions don't end in
        # periods; consequently they end up in the body of the article as run-on
        # sentences)
        def fix_missing_period(line):
            """Adds a period to a line that is missing a period."""
            if "@highlight" in line:
                return line
            if not line:
                return line
            if line[-1] in END_TOKENS:
                return line
            return line + " ."

        lines = [fix_missing_period(line) for line in lines]

        # Separate out article and abstract sentences
        article_lines = []
        highlights = []
        next_is_highlight = False
        for line in lines:
            if not line:
                continue  # empty line
            elif line.startswith("@highlight"):
                next_is_highlight = True
            elif next_is_highlight:
                highlights.append(line)
            else:
                article_lines.append(line)

        # Make article into a single string
        article = " ".join(article_lines)

        abstract = " ".join(highlights)
        return article, abstract

    folder = os.path.join(DATA_FOLDER, "cnn/stories")
    data = [_get_art_abs(os.path.join(folder, f)) for f in os.listdir(folder)]
    data = [
        row for row in data if row[0] and row[1] and len(row[0].split()) > 30
    ]
    example_idxes = [81846, 36799]
    data = [row for i, row in enumerate(data) if i not in example_idxes]
    instruction = (
        "You are provided with multiple news articles below. You will be asked"
        " to write a highlight for a particular news article."
    )
    example = """Article FE6806: World-renowned chef, author and Emmy winning television personality Anthony Bourdain visits Libya in the next episode of "Anthony Bourdain: Parts Unknown," airing Sunday, May 19, at 9 p.m. ET. Follow the show on Twitter and Facebook.
Article 43B294: (CNN) -- Qatar plans to build nine fully air-conditioned open-air stadiums to stage matches at the 2022 FIFA World Cup. Click through the gallery above to see how the stadiums will look.
Highlight FE6806: Which American food is a hit in Libya? Bourdain meets Libyan Boy Scouts . Bourdain visits Libya's Misrata War Museum ."""

    pbar = tqdm(desc="Processing cnnnews", total=len(buckets) * n_examples)
    median_len = median([len(l[0].split()) for l in data])
    prev_bucket = 0
    seeder = random.Random(seed)
    n_docs = {128000: 200}
    all_data = []
    for bucket in buckets:
        out_data = []
        if bucket in n_docs:
            n_doc = n_docs[bucket]
        else:
            n_doc: int = int(-(-bucket // median_len))
        gold_indexes = [0] + list(range(4, n_doc, 5))
        while len(out_data) < n_examples:
            for gold_index in gold_indexes:
                trials = []
                while True:
                    article_ids = [
                        secrets.token_hex(3).upper() for _ in range(n_doc)
                    ]
                    rows = seeder.sample(data, k=n_doc)
                    input_str = []
                    for i, row in enumerate(rows):
                        input_str.append(f"Article {article_ids[i]}: {row[0]}")
                    input_str = "\n".join(input_str)
                    input_str += f"\nHighlight {article_ids[gold_index]}:"
                    if one_shot:
                        input_str = example + "\n\n" + input_str
                    answer = rows[gold_index][1]
                    input_l = len((instruction + " " + input_str).split())
                    total_l = len(
                        (instruction + " " + input_str + answer).split()
                    )
                    if total_l < prev_bucket or total_l > bucket:
                        if len(trials) > 200:
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
                            "n_doc": n_doc,
                        }
                    )
                    break
                pbar.update(1)
        prev_bucket = bucket
        all_data += out_data

    pbar.close()
    out_filename = os.path.join(OUTPUT_FOLDER, "cnnnews.jsonl")
    with open(out_filename, "w") as f:
        f.write(
            cc.convert(
                "\n".join(
                    [json.dumps(row, ensure_ascii=False) for row in all_data]
                )
            )
        )
    print(
        f"Cnnnews data saved to {out_filename} with {len(all_data)} samples."
    )


def news2016(
    buckets: List[int],
    seed: int,
    n_examples: int = 1000,
    one_shot: bool = True,
):
    in_filename = os.path.join(DATA_FOLDER, "news2016zh/news2016zh_valid.json")
    data = [json.loads(line) for line in open(in_filename)]
    # keywords: '高德，租车'
    # title: '高德开放LBS+平台 与神州租车达成战略合作'
    # content: '...'
    data = [
        row
        for row in data
        if row["keywords"] not in row["title"]
        and row["keywords"] not in row["content"]
        and row["title"] not in row["content"]
        and len(row["content"]) > 200
        and len(row["content"]) < 800
    ]

    instruction = (
        "以下有几段新闻。请根据关键词找到最相关的一段新闻，"
        "并为其写一个约三十字的标题。"
    )
    example = """新闻1: 南安普顿右路传中，国米中卫穆里略头球争顶并未能解围成功，反而将球蹭逻辑点。等候多时的罗德里格斯飞身铲射险些得手！换个角度再看一遍！差之毫厘，国米险些站立。
新闻2: 欢迎收看搜狐体育带来的精彩视频；北京时间10月22日，观澜湖明星赛首轮比赛在黑石球场展开较量。在观澜湖明星赛上姚明和艾弗森两位NBA名人堂球员决赛投篮了，结果让人大跌眼镜。
关键词: 险情，乌龙
答案: 最相关的新闻是1。标题是"边路传中造险情，穆里略险些乌龙辅助"。"""
    median_len = median([len(row["content"]) for row in data])

    seeder = random.Random(seed)
    prev_bucket = 0
    all_data = []
    for bucket in buckets:
        out_data = []
        if one_shot:
            n_doc = max(
                (bucket - len(example) - len(instruction)) // median_len,
                2,
            )
        else:
            n_doc = max((bucket - len(instruction)) // median_len, 2)
        pbar = tqdm(
            desc=f"Processing news2016 bucket={bucket}", total=n_examples
        )
        while len(out_data) < n_examples:
            gold_indexes = [0] + list(range(4, n_doc, 5))
            for gold_index in gold_indexes:
                trial = []
                while True:
                    docs = seeder.sample(data, k=n_doc)
                    input_str = "\n".join(
                        [
                            f"新闻{i}: {doc['content']}"
                            for i, doc in enumerate(docs, 1)
                        ]
                    )
                    input_str += (
                        f"\n关键词: {docs[gold_index]['keywords']}\n答案: "
                    )
                    if one_shot:
                        input_str = example + "\n\n" + input_str
                    answer = (
                        f"最相关的新闻是{gold_index+1}。"
                        f"\"{docs[gold_index]['title']}\""
                    )
                    input_l = len(input_str)
                    total_l = input_l + len(answer)
                    if total_l > prev_bucket and total_l <= bucket:
                        out_data.append(
                            {
                                "instruction": instruction,
                                "input": input_str,
                                "answers": [answer],
                                "input_length": input_l,
                                "total_length": total_l,
                                "length_bucket": bucket,
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
        pbar.close()
        all_data += out_data
        prev_bucket = bucket

    out_filename = os.path.join(OUTPUT_FOLDER, "news2016.jsonl")
    with open(out_filename, "w") as f:
        f.write(
            cc.convert(
                "\n".join(
                    [json.dumps(row, ensure_ascii=False) for row in out_data]
                )
            )
        )
    print(
        f"News2016 data saved to {out_filename} with {len(out_data)} samples."
    )


def wikihow(
    buckets: List[int],
    seed: int,
    n_examples: int = 1000,
    one_shot: bool = True,
):
    instruction = (
        "Below are some articles from wikihow. I will ask you to summarize a"
        " particular article at the end."
    )
    example = """Article 1: It's style that will leave you looking classy and feminine. This is a twist on the classic — messier and more mermaid-like. This more obscure braid requires skill, but results in an interesting look. It's cute, classic, and easy to do. It looks bit medieval and very eye-catching. It's ideal for weddings or other elegant occasions. 
Article 2: It's a green app that contains a white phone icon inside a white text bubble. It's at the top-center of the screen. Select the chat with the attachment you wish to download. Select the attachment you wish to download. It's in the upper-right corner of the screen. The attachment has been saved to your Android device.
Question: Summarize the article related to "How to Style Very Long Hair" using a few instructive sentences.
Summary: Do a French braid. Make an intricate fishtail braid. Try a Dutch braid. Do a triple braid. Make a crazy braid. Do a cascading waterfall braid.  """
    example_titles = [
        "How to Download on WhatsApp",
        "How to Style Very Long Hair",
    ]
    in_filename = os.path.join(DATA_FOLDER, "wikihow/wikihowAll.csv")
    df = pd.read_csv(in_filename)
    df = [
        row
        for row in df.to_dict("records")
        if type(row["text"]) == str
        and len(row["text"].split()) > 60
        and row["title"] not in example_titles
    ]
    for row in df:
        row["headline"] = (
            row["headline"].replace(".,", ".").replace("\n", " ").strip()
        )
        row["text"] = re.sub(r"[.]+[\n]+[,]", ".\n", row["text"])
        row["text"] = re.sub(r"\n\s+", " ", row["text"]).strip()
        while row["title"][-1].isdigit():
            row["title"] = row["title"][:-1]
    median_len = median([len(row["text"].split()) for row in df])

    seeder = random.Random(seed)
    prev_bucket = 0
    n_docs = {16000: 35, 32000: 70, 64000: 140, 128000: 280}
    all_data = []
    for bucket in buckets:
        out_data = []
        if bucket in n_docs:
            n_doc = n_docs[bucket]
        elif one_shot:
            n_doc = max(
                (bucket - len(example) - len(instruction)) // median_len,
                2,
            )
        else:
            n_doc = max((bucket - len(instruction)) // median_len, 2)
        n_doc = int(n_doc)
        pbar = tqdm(
            desc=f"Processing wikihow bucket={bucket}", total=n_examples
        )
        while len(out_data) < n_examples:
            gold_indexes = [0] + list(range(4, n_doc, 5))
            for gold_index in gold_indexes:
                trial = []
                while True:
                    docs = seeder.sample(df, k=n_doc)
                    input_str = "\n".join(
                        [
                            f"Article {i}: {doc['text']}"
                            for i, doc in enumerate(docs, 1)
                        ]
                    )
                    input_str += (
                        "\nQuestion: Summarize the article related to"
                        f" \"{docs[gold_index]['title']}\" using a few"
                        " instructive sentences.\nSummary:"
                    )
                    if one_shot:
                        input_str = example + "\n\n" + input_str
                    answer = docs[gold_index]["headline"]
                    input_l = len(input_str.split())
                    total_l = input_l + len(answer.split())
                    if total_l > prev_bucket and total_l <= bucket:
                        out_data.append(
                            {
                                "instruction": instruction,
                                "input": input_str,
                                "answers": [answer],
                                "input_length": input_l,
                                "total_length": total_l,
                                "length_bucket": bucket,
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
        pbar.close()
        prev_bucket = bucket
        all_data += out_data
    out_filename = os.path.join(OUTPUT_FOLDER, "wikihow.jsonl")
    with open(out_filename, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in out_data]
            )
        )
    print(
        f"Wikihow data saved to {out_filename} with {len(out_data)} samples."
    )


def main(buckets: List[int], seed: int, n_examples=1000):
    buckets = sorted(buckets)
    cnewsum(buckets, seed, n_examples=n_examples)
    clts(buckets, seed, n_examples=n_examples)
    arxiv(buckets, seed=seed, n_examples=n_examples)
    bigpatent(buckets, seed=seed, n_examples=n_examples)
    pubmed(buckets, seed=seed, n_examples=n_examples)
    ncls(buckets, seed, n_examples=n_examples)
    cepsum(buckets, seed, n_examples=n_examples)
    lcsts(buckets, seed, n_examples=n_examples)
    cnnnews(buckets, seed, n_examples=n_examples)
    news2016(buckets, seed, n_examples=n_examples)
    wikihow(buckets, seed, n_examples=n_examples)


if __name__ == "__main__":
    cc = OpenCC("t2s")
    StrictFire(main)
