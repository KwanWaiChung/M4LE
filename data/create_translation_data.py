import sys
import os

ROOT_FOLDER = os.path.abspath(__file__)
for _ in range(2):
    ROOT_FOLDER = os.path.dirname(ROOT_FOLDER)
sys.path.append(ROOT_FOLDER)
import json
import random
import xml.etree.ElementTree as ET
import re
from xml.etree.ElementTree import ParseError

from typing import Dict, Any
from typing import List
from strictfire import StrictFire
from tqdm import tqdm
from statistics import median
from opencc import OpenCC

OUTPUT_FOLDER = os.path.join(ROOT_FOLDER, "data", "translation")
DATA_FOLDER = os.path.join(ROOT_FOLDER, "raw_data", "translation")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def int_to_en(num):
    d = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
        13: "thirteen",
        14: "fourteen",
        15: "fifteen",
        16: "sixteen",
        17: "seventeen",
        18: "eighteen",
        19: "nineteen",
        20: "twenty",
        30: "thirty",
        40: "forty",
        50: "fifty",
        60: "sixty",
        70: "seventy",
        80: "eighty",
        90: "ninety",
    }
    k = 1000
    m = k * 1000
    b = m * 1000
    t = b * 1000

    assert 0 <= num

    if num < 20:
        return d[num]

    if num < 100:
        if num % 10 == 0:
            return d[num]
        else:
            return d[num // 10 * 10] + " " + d[num % 10]

    if num < k:
        if num % 100 == 0:
            return d[num // 100] + " hundred"
        else:
            return d[num // 100] + " hundred and " + int_to_en(num % 100)

    if num < m:
        if num % k == 0:
            return int_to_en(num // k) + " thousand"
        else:
            return int_to_en(num // k) + " thousand, " + int_to_en(num % k)

    if num < b:
        if (num % m) == 0:
            return int_to_en(num // m) + " million"
        else:
            return int_to_en(num // m) + " million, " + int_to_en(num % m)

    if num < t:
        if (num % b) == 0:
            return int_to_en(num // b) + " billion"
        else:
            return int_to_en(num // b) + " billion, " + int_to_en(num % b)

    if num % t == 0:
        return int_to_en(num // t) + " trillion"
    else:
        return int_to_en(num // t) + " trillion, " + int_to_en(num % t)


def news_commentary(
    seed: int, buckets: List[int], n_examples: int, article_len: int
):
    en_prompt = """Paragraph 1: The recent events in Tibet and adjoining provinces are causes for deep concern . 
Paragraph 2: The selection of Beijing to organize and host the 2008 Olympic Games was accompanied by the Chinese government ’ s pledges of visible progress on respect for human rights . 
Question: Can you identitfy the paragraph related to 'Playing for Human Rights' out of the two paragraphs provided above and translate it into Chinese?
Answer: 段落2: 当北京被挑选主办2008年奥运会的时候，中国政府承诺在尊重人权上取得明显的进步。"""
    en_prompt_length = (
        len(
            """ News 1: The recent events in Tibet and adjoining provinces are causes for deep concern .
News 2: The selection of Beijing to organize and host the 2008 Olympic Games was accompanied by the Chinese government ’ s pledges of visible progress on respect for human rights .
Question: Can you identitfy the paragraph related to 'Playing for Human Rights' out of the two paragraphs provided above and translate it into Chinese?
Answer: """.split()
        )
        + len(
            "新闻2: 当北京被挑选主办2008年奥运会的时候，"
            "中国政府承诺在尊重人权上取得明显的进步。"
        )
    )
    zh_prompt = """段落1: 西藏及其临近省份最近的事态令人极为担忧。
段落2: 当北京被挑选主办2008年奥运会的时候，中国政府承诺在尊重人权上取得明显的进步。
问题: 请你从上例2个段落中选出有关「为人权而赛」的段落，并把它翻译成英文。
答案: Paragraph 2: The selection of Beijing to organize and host the 2008 Olympic Games was accompanied by the Chinese government ’ s pledges of visible progress on respect for human rights ."""
    zh_prompt_length = (
        len(zh_prompt.split("答案:")[0])
        + 3
        + len(zh_prompt.split("答案:")[1].split())
    )
    folder = os.path.join(
        DATA_FOLDER,
        "News-Commentary_v16",
    )
    en_folder = os.path.join(folder, "en", "News-Commentary", "xml", "en")
    zh_folder = os.path.join(folder, "zh", "News-Commentary", "xml", "zh")
    alignments = ET.parse(os.path.join(folder, "en-zh.xml")).getroot()

    data = []
    for element in tqdm(alignments, desc="Reading data from news commentary"):
        xtargets = [e.attrib["xtargets"] for e in element]
        if not xtargets:
            continue
        en_doc = element.attrib["fromDoc"].split("/")[1].split(".gz")[0]
        zh_doc = element.attrib["toDoc"].split("/")[1].split(".gz")[0]
        # chosen as example
        if en_doc in [
            "tibet-s-peace-of-the-grave.xml",
            "playing-for-human-rights.xml",
        ]:
            continue
        try:
            en_tree = ET.parse(os.path.join(en_folder, en_doc)).getroot()
            zh_tree = ET.parse(os.path.join(zh_folder, zh_doc)).getroot()
        except ParseError:
            continue
        en_headline = " ".join(
            [ele.text for ele in en_tree.findall(".HEADLINE/s/chunk/w")]
        )
        zh_headline = "".join(
            [ele.text for ele in zh_tree.findall(".HEADLINE/s/w")]
        )
        en_paragraphs2 = [child for child in en_tree][1:]
        zh_paragraphs2 = [child for child in zh_tree][1:]
        en_paragraphs = {}
        zh_paragraphs = {}
        for para in en_paragraphs2:
            for w in para.iter("w"):
                _id = int(w.attrib["id"][1:].split(".")[0])
                en_paragraphs.setdefault(_id, []).append(w.text)
        for para in zh_paragraphs2:
            for w in para.iter("w"):
                _id = int(w.attrib["id"][1:].split(".")[0])
                zh_paragraphs.setdefault(_id, []).append(w.text)

        en_text = []
        zh_text = []
        total_l = 0
        next_en_text = ""
        next_zh_text = ""
        next_total_l = 0
        for xtarget in xtargets[1:]:
            _ids = xtarget.split(";")
            next_en_text = " ".join(
                sum(
                    [en_paragraphs[int(en_id)] for en_id in _ids[0].split()],
                    [],
                )
            )

            if _ids[1]:
                next_zh_text = "".join(
                    sum(
                        [
                            zh_paragraphs[int(zh_id)]
                            for zh_id in _ids[1].split()
                        ],
                        [],
                    )
                )
                next_total_l += len(next_zh_text)
            next_total_l += len(next_en_text.split())
            # must end with both translions
            if total_l + next_total_l > article_len and _ids[1]:
                if total_l > 0:
                    en_text = " ".join(en_text)
                    zh_text = "".join(zh_text)
                    data.append(
                        {
                            "en_headline": en_headline,
                            "en_text": en_text,
                            "zh_headline": zh_headline,
                            "zh_text": zh_text,
                            "en_length": len(en_text.split()),
                            "zh_length": len(zh_text),
                            "total_length": total_l,
                        }
                    )
                    en_text = []
                    zh_text = []
                    total_l = 0
            if _ids[1]:
                en_text.append(next_en_text)
                zh_text.append(next_zh_text)
                total_l += next_total_l
                next_total_l = 0
        if total_l > 0:
            en_text = " ".join(en_text)
            zh_text = "".join(zh_text)
            data.append(
                {
                    "en_headline": en_headline,
                    "en_text": en_text,
                    "zh_headline": zh_headline,
                    "zh_text": zh_text,
                    "en_length": len(en_text.split()),
                    "zh_length": len(zh_text),
                    "total_length": total_l,
                }
            )

    # only use passage length under 400. 385 matches (excluded 2 examples)
    seeder = random.Random(seed)
    prev_bucket = 0
    median_en_len = median([tmp["en_length"] for tmp in data])
    median_cn_len = median([tmp["zh_length"] for tmp in data])
    all_data = []
    for bucket in buckets:
        n_passage = int(
            -(
                -(bucket - en_prompt_length - median_cn_len)
                // (median_en_len * 1.2)
            )
        )
        if n_passage < 5:
            gold_indexes = [0, n_passage - 1]
        else:
            gold_indexes = [0] + list(range(4, n_passage, 5))
        instruction = (
            f"I will provide you {n_passage} paragraphs below. I will"
            " ask you to identify one of them and translate it into"
            " Chinese.\n"
        )
        out_data = []
        pbar = tqdm(
            desc=(
                f"Creating news commentary en2zh examples for bucket={bucket}"
            ),
            total=n_examples,
        )
        while len(out_data) < n_examples:
            for gold_index in gold_indexes:
                while True:
                    sampled_para: List[Dict[str, Any]] = seeder.sample(
                        data, k=n_passage
                    )
                    input_str = "\n".join(
                        [
                            f"Paragraph {i}: {passage['en_text']}"
                            for i, passage in enumerate(sampled_para, 1)
                        ]
                    )
                    input_l = en_prompt_length
                    input_str += (
                        "\nQuestion: Can you identify the paragraph related"
                        f" to '{sampled_para[gold_index]['en_headline']}' out"
                        f" of the {n_passage} paragraphs provided"
                        " above and translate it into Chinese?\nAnswer:"
                    )
                    input_l += len(input_str.split())
                    input_str = f"{en_prompt}\n\n" + input_str
                    answer = (
                        f"段落{gold_index+1}:"
                        f" {sampled_para[gold_index]['zh_text']}"
                    )
                    total_l = input_l + len(answer)
                    if (
                        total_l <= bucket
                        and total_l >= prev_bucket + (bucket - prev_bucket) / 2
                    ):
                        break
                out_data.append(
                    {
                        "instruction": instruction,
                        "input": input_str,
                        "answers": [answer],
                        "input_length": input_l,
                        "total_length": total_l,
                        "length_bucket": bucket,
                        "gold_index": gold_index,
                        "n_passages": n_passage,
                    }
                )
                pbar.update(1)
        pbar.close()
        prev_bucket = bucket
        if len(out_data) > n_examples:
            out_data = seeder.sample(out_data, k=n_examples)
        all_data += out_data
    out_filename = os.path.join(
        OUTPUT_FOLDER,
        f"news-commentary-en2zh.jsonl",
    )
    with open(out_filename, "w") as f:
        f.write(
            cc.convert(
                "\n".join(
                    [json.dumps(row, ensure_ascii=False) for row in all_data]
                )
            )
        )
    print(
        f"News commentary en2zh data has been stored in {out_filename} with"
        f" {len(all_data)} data."
    )

    # cn to en now.
    prev_bucket = 0
    all_data = []
    for bucket in buckets:
        # n_passage = -(
        #     -(bucket - en_prompt_length - median_cn_len) // median_en_len
        # )
        n_passage = int(
            -(
                -(bucket - zh_prompt_length - median_en_len)
                // (median_cn_len * 1.2)
            )
        )
        if n_passage < 5:
            gold_indexes = [0, n_passage - 1]
        else:
            gold_indexes = [0] + list(range(4, n_passage, 5))
        instruction = f"我将在下面给您提供{n_passage}个段落。我会请您找出其中一个并将其翻译成英文。"
        out_data = []
        pbar = tqdm(
            desc=(
                f"Creating news commentary zh2en examples for bucket={bucket}"
            ),
            total=n_examples,
        )
        while len(out_data) < n_examples:
            for gold_index in gold_indexes:
                while True:
                    sampled_para: List[Dict[str, Any]] = seeder.sample(
                        data, k=n_passage
                    )
                    input_str = "\n".join(
                        [
                            f"段落{i}: {passage['zh_text']}"
                            for i, passage in enumerate(sampled_para, 1)
                        ]
                    )
                    input_l = zh_prompt_length
                    input_str += (
                        "\n问题:"
                        f" 请你从上例{n_passage}个段落中选出有关「{sampled_para[gold_index]['zh_headline']}」的段落，"
                        "并将其翻译成英文。\n答案:"
                    )
                    input_l += len(input_str)
                    input_str = f"{zh_prompt}\n\n" + input_str
                    answer = (
                        f"Paragraph {gold_index+1}:"
                        f" {sampled_para[gold_index]['en_text']}"
                    )
                    total_l = input_l + len(answer.split())
                    if (
                        total_l > prev_bucket + (bucket - prev_bucket) / 2
                        and total_l <= bucket
                    ):
                        break
                out_data.append(
                    {
                        "instruction": instruction,
                        "input": input_str,
                        "answers": [answer],
                        "input_length": input_l,
                        "total_length": total_l,
                        "length_bucket": bucket,
                        "gold_index": gold_index,
                        "n_passages": n_passage,
                    }
                )
                pbar.update(1)
        pbar.close()
        if len(out_data) > n_examples:
            out_data = seeder.sample(out_data, k=n_examples)
        all_data += out_data
        prev_bucket = bucket

    out_filename = os.path.join(
        OUTPUT_FOLDER,
        f"news-commentary-zh2en.jsonl",
    )
    with open(out_filename, "w") as f:
        f.write(
            cc.convert(
                "\n".join(
                    [json.dumps(row, ensure_ascii=False) for row in all_data]
                )
            )
        )
    print(
        f"News commentary zh2en data has been stored in {out_filename} with"
        f" {len(all_data)} data."
    )


def ted_talk(
    seed: int,
    buckets: List[int],
    n_examples: int = 1000,
    one_shot: bool = True,
):
    en_instruction = (
        "Below are few TED Talks English transcripts. You have to translate a"
        " particular TED Talk related to a given title into Chinese."
    )
    zh_instruction = (
        "以下是一些TED演讲的中文字幕。您需要将特定标题相关的一个TED演讲翻译成英文。"
    )
    en_example = """Talk 1: Hello everyone. And so the two of us are here to give you an example of creation. And I'm going to be folding one of Robert Lang's models. And this is the piece of paper it will be made from, and you can see all of the folds that are needed for it. And Rufus is going to be doing some improvisation on his custom, five-string electric cello, and it's very exciting to listen to him. Are you ready to go? OK. Just to make it a little bit more exciting. All right. Take it away, Rufus. (Music) All right. There you go. (Laughter) (Applause)
Talk 2: Tonight we're going to play you two songs. We're three brothers from New Jersey, and the funny thing is that, believe it or not, we are hooked on bluegrass and we're excited to play it for you tonight. (Music) (Applause) TM: Thank you, thank you. (Applause) Robbie Mizzone: Thank you. I'm Robbie Mizzone. I'm 13, and I play the fiddle. This is my brother, Jonny. He's 10, and he plays the banjo. And on guitar is my 14-year-old brother, Tommy. (Applause) We call ourselves the Sleepy Man Banjo Boys. (Music) (Applause) TM: Thank you. JM: Thank you all. TM: Thank you very much.
Question: Translate the talk related to 'Sleepy Man Banjo Boys: Teen wonders play bluegrass' into Chinese.
Translation: 演讲2: 今晚我们要给大家演奏两首曲子我们是来自新泽西的三兄弟有意思的是我们对蓝草音乐很着迷同时我们也很兴奋能够在这里演出（音乐）（掌声）TM：谢谢，谢谢大家（掌声）RM：谢谢我叫Robbie Mizzone，今年十三岁，我负责小提琴这是我的弟弟，Jonny，他十岁了，他弹奏班卓琴吉他部分则是我的14岁哥哥，Tommy（鼓掌）我们是Sleepy Man Banjo Boys（音乐）（掌声）再次感谢感谢大家非常荣幸。"""

    zh_example = """演讲1: 大家好。今天我们俩将为大家展示一项创作。我将把罗伯特朗的其中一个模型完整地重现在大家的面前。就用这张纸。所有的褶痕你都看得见。而在此期间,鲁弗斯将为大家做一些即兴的创作。他的拿手好戏，五弦电子大提琴演奏。值得一听！准备好了吗？要开始咯！为了让这一切更有趣……好了，鲁弗斯，把它拿走（音乐）好了，开始咯！（笑声）（鼓掌
演讲2: 今晚我们要给大家演奏两首曲子我们是来自新泽西的三兄弟有意思的是我们对蓝草音乐很着迷同时我们也很兴奋能够在这里演出（音乐）（掌声）TM：谢谢，谢谢大家（掌声）RM：谢谢我叫Robbie Mizzone，今年十三岁，我负责小提琴这是我的弟弟，Jonny，他十岁了，他弹奏班卓琴吉他部分则是我的14岁哥哥，Tommy（鼓掌）我们是Sleepy Man Banjo Boys（音乐）（掌声）再次感谢感谢大家非常荣幸。
问题： 请将有关'蓝草爵士三兄弟'的演讲翻译成英文。
翻译: Talk 2: Tonight we're going to play you two songs. We're three brothers from New Jersey, and the funny thing is that, believe it or not, we are hooked on bluegrass and we're excited to play it for you tonight. (Music) (Applause) TM: Thank you, thank you. (Applause) Robbie Mizzone: Thank you. I'm Robbie Mizzone. I'm 13, and I play the fiddle. This is my brother, Jonny. He's 10, and he plays the banjo. And on guitar is my 14-year-old brother, Tommy. (Applause) We call ourselves the Sleepy Man Banjo Boys. (Music) (Applause) TM: Thank you. JM: Thank you all. TM: Thank you very much."""

    en_example_len = (
        len(
            """Talk 1: Hello everyone. And so the two of us are here to give you an example of creation. And I'm going to be folding one of Robert Lang's models. And this is the piece of paper it will be made from, and you can see all of the folds that are needed for it. And Rufus is going to be doing some improvisation on his custom, five-string electric cello, and it's very exciting to listen to him. Are you ready to go? OK. Just to make it a little bit more exciting. All right. Take it away, Rufus. (Music) All right. There you go. (Laughter) (Applause)
Talk 2: Tonight we're going to play you two songs. We're three brothers from New Jersey, and the funny thing is that, believe it or not, we are hooked on bluegrass and we're excited to play it for you tonight. (Music) (Applause) TM: Thank you, thank you. (Applause) Robbie Mizzone: Thank you. I'm Robbie Mizzone. I'm 13, and I play the fiddle. This is my brother, Jonny. He's 10, and he plays the banjo. And on guitar is my 14-year-old brother, Tommy. (Applause) We call ourselves the Sleepy Man Banjo Boys. (Music) (Applause) TM: Thank you. JM: Thank you all. TM: Thank you very much.
Question: Translate the talk related to 'Sleepy Man Banjo Boys: Teen wonders play bluegrass' into Chinese.
Translation:""".split()
        )
        + len(
            "演讲2: 今晚我们要给大家演奏两首曲子我们是来自新泽西的三兄弟有意思的是我们对蓝草音乐很着迷同时我们也很兴奋能够在这里演出（音乐）（掌声）TM：谢谢，"
            "谢谢大家（掌声）RM：谢谢我叫Robbie Mizzone，今年十三岁，"
            "我负责小提琴这是我的弟弟，Jonny，他十岁了，"
            "他弹奏班卓琴吉他部分则是我的14岁哥哥，Tommy（鼓掌）我们是Sleepy"
            " Man Banjo Boys（音乐）（掌声）再次感谢感谢大家非常荣幸。"
        )
    )
    zh_example_len = (
        len(
            """演讲1: 大家好。今天我们俩将为大家展示一项创作。我将把罗伯特朗的其中一个模型完整地重现在大家的面前。
        演讲2: 今晚我们要给大家演奏两首曲子我们是来自新泽西的三兄弟有意思的是我们对蓝草音乐很着迷同时我们也很兴奋能够在这里演出（音乐）（掌声）TM：谢谢，谢谢大家（掌声）RM：谢谢我叫Robbie Mizzone，今年十三岁，我负责小提琴这是我的弟弟，Jonny，他十岁了，他弹奏班卓琴吉他部分则是我的14岁哥哥，Tommy（鼓掌）我们是Sleepy Man Banjo Boys（音乐）（掌声）再次感谢感谢大家非常荣幸。
        问题： 请将有关'蓝草爵士三兄弟'的演讲翻译成英文。
        翻译:"""
        )
        + len(
            (
                "Talk 2: Tonight we're going to play you two songs. We're"
                " three brothers from New Jersey, and the funny thing is that,"
                " believe it or not, we are hooked on bluegrass and we're"
                " excited to play it for you tonight. (Music) (Applause) TM:"
                " Thank you, thank you. (Applause) Robbie Mizzone: Thank you."
                " I'm Robbie Mizzone. I'm 13, and I play the fiddle. This is"
                " my brother, Jonny. He's 10, and he plays the banjo. And on"
                " guitar is my 14-year-old brother, Tommy. (Applause) We call"
                " ourselves the Sleepy Man Banjo Boys. (Music) (Applause) TM:"
                " Thank you. JM: Thank you all. TM: Thank you very much."
            ).split()
        )
    )
    en_file = os.path.join(DATA_FOLDER, "tedtalk/ted_en-20160408.xml")
    zh_file = os.path.join(DATA_FOLDER, "tedtalk/ted_zh-cn-20160408.xml")
    en_tree = ET.parse(en_file)
    zh_tree = ET.parse(zh_file)
    en_root = en_tree.getroot()
    zh_root = zh_tree.getroot()

    en_talks = [elem for elem in en_root]
    zh_talks = [elem for elem in zh_root]
    en_talk_dict = {elem[0].find("talkid").text: elem[0] for elem in en_talks}
    zh_talk_dict = {elem[0].find("talkid").text: elem[0] for elem in zh_talks}
    data = []
    for talkid in zh_talk_dict:
        en_talk = en_talk_dict[talkid]
        zh_talk = zh_talk_dict[talkid]
        title_node = zh_talk.find("title")
        if title_node is None:
            continue
        zh_talk_title: str = title_node.text
        en_talk_title: str = en_talk.find("title").text
        if talkid in ["1619", "322"]:  # chosen as examples.
            continue
        en_text = []
        zh_text = []
        next_total_l = 0
        for en_ele, zh_ele in zip(
            en_talk.findall("transcription/seekvideo"),
            zh_talk.findall("transcription/seekvideo"),
        ):
            _en_text = en_ele.text
            _zh_text = zh_ele.text
            next_total_l += len(_en_text.split()) + len(_zh_text)
            if next_total_l >= 300:
                break
            en_text.append(_en_text)
            zh_text.append(_zh_text)
        en_text = " ".join(en_text)
        zh_text = "".join(zh_text)
        if len(en_text.split()) > 50:
            data.append(
                (talkid, en_talk_title, zh_talk_title, en_text, zh_text)
            )
    seeder = random.Random(seed)
    prev_bucket = 0
    all_en_data = []
    all_zh_data = []
    for bucket in buckets:
        en_data = []
        zh_data = []
        if one_shot:
            n_en_talks = (bucket - 500) // 120
            n_zh_talks = (bucket - 500) // 190
        else:
            n_en_talks = bucket // 150
            n_zh_talks = bucket // 190
        # construct en2zh
        pbar = tqdm(
            desc=f"Processing ted talk bucket={bucket}", total=2 * n_examples
        )
        while len(en_data) < n_examples:
            gold_indexes = [0] + list(range(4, n_en_talks, 5))
            for gold_index in gold_indexes:
                trial = []
                while True:
                    talks = seeder.sample(data, k=n_en_talks)
                    input_str = [
                        f"Talk {i}: {talk[3]}"
                        for i, talk in enumerate(talks, 1)
                    ]
                    input_str.append(
                        "Question: Translate the talk related to"
                        f" '{talks[gold_index][1]}' into Chinese."
                    )
                    input_str = "\n".join(input_str)
                    input_str += "\nTranslation:"
                    input_l = len(en_instruction.split()) + len(
                        input_str.split()
                    )
                    if one_shot:
                        input_str = en_example + "\n\n" + input_str
                        input_l += en_example_len
                    answer = f"演讲{gold_index+1}: {talks[gold_index][4]}"
                    total_l = input_l + len(answer)
                    if (
                        total_l > bucket
                        or total_l < prev_bucket + (bucket - prev_bucket) / 2
                    ):
                        trial.append(total_l)
                        continue
                    en_data.append(
                        {
                            "instruction": en_instruction,
                            "input": input_str,
                            "answers": [answer],
                            "input_length": input_l,
                            "total_length": total_l,
                            "length_bucket": bucket,
                            "gold_index": gold_index,
                            "titles": [talk[1] for talk in talks],
                        }
                    )
                    break
                pbar.update(1)
                # zh2en
        while len(zh_data) < n_examples:
            gold_indexes = [0] + list(range(4, n_zh_talks, 5))
            for gold_index in gold_indexes:
                trial = []
                while True:
                    talks = seeder.sample(data, k=n_zh_talks)
                    input_str = [
                        f"演讲{i}: {talk[4]}"
                        for i, talk in enumerate(talks, 1)
                    ]
                    input_str.append(
                        f"问题: 请将有关'{talks[gold_index][2]}'的演讲翻译成英文。"
                    )
                    input_str = "\n".join(input_str)
                    input_str += "\n翻译:"
                    input_l = len(zh_instruction) + len(input_str)
                    zero_shot_input_l = len(zh_instruction) + len(input_str)
                    if one_shot:
                        input_str = zh_example + "\n\n" + input_str
                        input_l += zh_example_len
                    answer = f"Talk {gold_index+1}: {talks[gold_index][3]}"
                    total_l = input_l + len(answer.split())
                    if (
                        total_l > bucket
                        or total_l < prev_bucket + (bucket - prev_bucket) / 2
                    ):
                        trial.append(total_l)
                        continue
                    zh_data.append(
                        {
                            "instruction": en_instruction,
                            "input": input_str,
                            "answers": [answer],
                            "input_length": input_l,
                            "total_length": total_l,
                            "length_bucket": bucket,
                            "gold_index": gold_index,
                            "titles": [talk[2] for talk in talks],
                        }
                    )
                    break
                pbar.update(1)
        pbar.close()
        if len(en_data) > n_examples:
            en_data = seeder.sample(en_data, k=n_examples)
        all_en_data += en_data
        if len(zh_data) > n_examples:
            zh_data = seeder.sample(zh_data, k=n_examples)
        all_zh_data += zh_data
        prev_bucket = bucket
    out_filename = os.path.join(OUTPUT_FOLDER, "tedtalks-en2zh.jsonl")
    with open(out_filename, "w") as f:
        f.write(
            cc.convert(
                "\n".join(
                    [
                        json.dumps(row, ensure_ascii=False)
                        for row in all_en_data
                    ]
                )
            )
        )
    print(
        f"Tedtalks en2zh data saved to {out_filename} with"
        f" {len(all_en_data)} samples."
    )

    out_filename = os.path.join(OUTPUT_FOLDER, "tedtalks-zh2en.jsonl")
    with open(out_filename, "w") as f:
        f.write(
            cc.convert(
                "\n".join(
                    [
                        json.dumps(row, ensure_ascii=False)
                        for row in all_zh_data
                    ]
                )
            )
        )
    print(
        f"Tedtalks zh2en data saved to {out_filename} with"
        f" {len(all_zh_data)} samples."
    )

    # out_filename = os.path.join(
    #     ROOT_FOLDER,
    #     "output",
    #     "translation",
    #     f"tedtalks_en2zh_len={bucket}_shots=1.json",
    # )
    # os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    # # if len(en_data) > 1000:
    # #     indexes = seeder.sample(list(range(len(en_data))), k=1000)
    # #     en_data = [en_data[i] for i in indexes]
    # #     zh_data = [zh_data[i] for i in indexes]

    # en_data = json.loads(
    #     cc.convert(json.dumps(en_data, ensure_ascii=False))
    # )
    # json.dump(
    #     en_data, open(out_filename, "w"), ensure_ascii=False, indent=4
    # )
    # print(f"data saved to {out_filename}")
    # out_filename = os.path.join(
    #     ROOT_FOLDER,
    #     "output",
    #     "translation",
    #     f"tedtalks_zh2en_len={bucket}_shots=1.json",
    # )
    # zh_data = json.loads(
    #     cc.convert(json.dumps(zh_data, ensure_ascii=False))
    # )
    # json.dump(
    #     zh_data, open(out_filename, "w"), ensure_ascii=False, indent=4
    # )
    # print(f"data saved to {out_filename}")


def open_subtitles(
    seed: int,
    buckets: List[int],
    n_examples: int,
):
    en_instruction = (
        "Translate movie subtitles from English to Chinese. The first one is"
        " an example."
    )
    zh_instruction = (
        "Translate movie subtitles from Chinese to English. The first one is"
        " an example."
    )
    en_example = """English: The family of a woman who died after a botched police raid intend to sue the city for $ 500 million , naming the city , the police department , and the officers involved in the raid .
Chinese:  在警方一次拙劣的突击行动中意外死亡的妇女，其家属欲起诉市政府并索赔5亿美元并同时指控市政府、警局和参与此次行动的警员"""
    zh_example = """Chinese:  在警方一次拙劣的突击行动中意外死亡的妇女，其家属欲起诉市政府并索赔5亿美元并同时指控市政府、警局和参与此次行动的警员
English: The family of a woman who died after a botched police raid intend to sue the city for $ 500 million , naming the city , the police department , and the officers involved in the raid ."""
    example_en_len = len(
        "English: The family of a woman who died after a botched police raid"
        " intend to sue the city for $ 500 million , naming the city , the"
        " police department , and the officers involved in the raid . Chinese:"
    )
    example_zh_len = len(
        "在警方一次拙劣的突击行动中意外死亡的妇女，"
        "其家属欲起诉市政府并索赔5亿美元并同时指控市政府、"
        "警局和参与此次行动的警员"
    )
    a_file = os.path.join(DATA_FOLDER, "opensubtitles/en-zh_cn.xml")
    doc_folder = os.path.join(DATA_FOLDER, "opensubtitles/OpenSubtitles/xml")
    alignments = ET.parse(a_file).getroot()
    xtargetss = []
    en_docs = []
    zh_docs = []
    for element in alignments:
        # for e in ele:
        #     print(e.attrib['xtargets'])

        en_doc = element.attrib["fromDoc"].split(".gz")[0]
        if "63013.xml" in en_doc:  # used as example
            continue
        xtargetss.append([e.attrib["xtargets"] for e in element])
        en_docs.append(en_doc)
        zh_docs.append(element.attrib["toDoc"].split(".gz")[0])

    seeder = random.Random(seed)
    prev_bucket = 0
    pbar = tqdm(
        total=len(buckets) * n_examples, desc="Processing open subtitles"
    )
    remove_prefix_count = 20
    all_en_data = []
    all_zh_data = []
    for bucket in buckets:
        en_data = []
        zh_data = []
        # idxes = seeder.sample(list(range(len(xtargetss))), k=n_examples)
        # xtargetss = [xtargetss[i] for i in idxes]
        # en_docs = [en_docs[i] for i in idxes]
        # zh_docs = [zh_docs[i] for i in idxes]
        temp = list(zip(xtargetss, en_docs, zh_docs))
        seeder.shuffle(temp)
        _xtargetss, _en_docs, _zh_docs = zip(*temp)
        for xtargets, en_doc, zh_doc in tqdm(
            zip(_xtargetss, _en_docs, _zh_docs), total=len(xtargetss)
        ):
            try:
                zh_tree = ET.parse(
                    os.path.join(
                        doc_folder,
                        zh_doc,
                    )
                ).getroot()
                en_tree = ET.parse(
                    os.path.join(
                        doc_folder,
                        en_doc,
                    )
                ).getroot()
            except Exception as e:
                continue

            zh_sentences = {}
            for sentence in zh_tree.findall("s"):
                s_id = sentence.attrib["id"]
                words = [word.text for word in sentence.findall("w")]
                zh_sentences[s_id] = words

            en_sentences = {}
            for sentence in en_tree.findall("s"):
                s_id = sentence.attrib["id"]
                words = [word.text for word in sentence.findall("w")]
                en_sentences[s_id] = words

            en_text = []
            zh_text = []
            en_l = example_en_len
            zh_l = example_zh_len
            next_en_text = ""
            next_zh_text = ""
            next_en_l = 0
            next_zh_l = 0
            for xtarget in xtargets[remove_prefix_count:]:
                en_ids, zh_ids = [tmp.strip() for tmp in xtarget.split(";")]
                if en_ids and zh_ids:
                    _text = " ".join(
                        sum(
                            [en_sentences[en_id] for en_id in en_ids.split()],
                            [],
                        )
                    )
                    _text = re.sub(r"\{.*?\}", "", _text)
                    next_en_text = _text
                    next_en_l = len(next_en_text.split())
                    _text = "".join(
                        sum(
                            [zh_sentences[zh_id] for zh_id in zh_ids.split()],
                            [],
                        )
                    )
                    _text = re.sub(r"\{.*?\}", "", _text)
                    next_zh_text = _text
                    next_zh_l = len(next_zh_text)
                    if en_l + zh_l + next_en_l + next_zh_l > bucket:
                        # insert
                        if (
                            (en_l + zh_l) < 2.6 * en_l
                            and en_l != example_en_len
                            and zh_l != example_zh_len
                        ):
                            input_str = (
                                en_example
                                + "\n\n"
                                + f"English: {' '.join(en_text)}\nChinese:"
                            )
                            answer = "".join(zh_text)
                            en_data.append(
                                {
                                    "instruction": en_instruction,
                                    "input": input_str,
                                    "answers": [answer],
                                    "input_length": en_l,
                                    "total_length": en_l + zh_l,
                                    "length_bucket": bucket,
                                    "en_doc": en_doc,
                                    "zh_doc": zh_doc,
                                }
                            )
                            input_str = (
                                zh_example
                                + "\n\n"
                                + f"Chinese: {''.join(zh_text)}\nEnglish:"
                            )
                            answer = " ".join(en_text)
                            zh_data.append(
                                {
                                    "instruction": zh_instruction,
                                    "input": input_str,
                                    "answers": [answer],
                                    "input_length": zh_l,
                                    "total_length": en_l + zh_l,
                                    "length_bucket": bucket,
                                    "en_doc": en_doc,
                                    "zh_doc": zh_doc,
                                }
                            )
                        # clear buffer
                        en_text = []
                        zh_text = []
                        en_l = example_en_len
                        zh_l = example_zh_len

                    en_text.append(next_en_text)
                    zh_text.append(next_zh_text)
                    en_l += next_en_l
                    zh_l += next_zh_l
            # insert
            if (
                (en_l + zh_l) < 2.6 * en_l
                and en_l != example_en_len
                and zh_l != example_zh_len
                and (en_l + zh_l) > prev_bucket
            ):
                input_str = (
                    en_example
                    + "\n\n"
                    + f"English: {' '.join(en_text)}\nChinese:"
                )
                answer = "".join(zh_text)
                en_data.append(
                    {
                        "instruction": en_instruction,
                        "input": input_str,
                        "answers": [answer],
                        "input_length": en_l,
                        "total_length": en_l + zh_l,
                        "length_bucket": bucket,
                        "en_doc": en_doc,
                        "zh_doc": zh_doc,
                    }
                )
                input_str = (
                    zh_example
                    + "\n\n"
                    + f"Chinese: {''.join(zh_text)}\English:"
                )
                answer = " ".join(en_text)
                zh_data.append(
                    {
                        "instruction": zh_instruction,
                        "input": input_str,
                        "answers": [answer],
                        "input_length": zh_l,
                        "total_length": en_l + zh_l,
                        "length_bucket": bucket,
                        "en_doc": en_doc,
                        "zh_doc": zh_doc,
                    }
                )
            # clear data
            en_text = []
            zh_text = []
            en_l = example_en_len
            zh_l = example_zh_len
            if len(en_data) >= n_examples:
                break
        if len(en_data) > n_examples:
            indexes = seeder.sample(list(range(len(en_data))), k=n_examples)
            en_data = [en_data[i] for i in indexes]
            zh_data = [zh_data[i] for i in indexes]

        all_en_data += en_data
        all_zh_data += zh_data
        prev_bucket = bucket

        # en_data = json.loads(
        #     cc.convert(json.dumps(en_data, ensure_ascii=False))
        # )
        # json.dump(
        #     en_data, open(out_filename, "w"), ensure_ascii=False, indent=4
        # )
        # print(f"data saved to {out_filename}")
        # out_filename = os.path.join(
        #     ROOT_FOLDER,
        #     "output",
        #     "translation",
        #     f"subtitles_zh2en_len={bucket}_shots=1.json",
        # )
        # zh_data = json.loads(
        #     cc.convert(json.dumps(zh_data, ensure_ascii=False))
        # )
        # json.dump(
        #     zh_data, open(out_filename, "w"), ensure_ascii=False, indent=4
        # )
        # print(f"data saved to {out_filename}")
    pbar.close()
    out_filename = os.path.join(
        OUTPUT_FOLDER,
        f"open-subtitles-en2zh.jsonl",
    )
    with open(out_filename, "w") as f:
        f.write(
            cc.convert(
                "\n".join(
                    [
                        json.dumps(row, ensure_ascii=False)
                        for row in all_en_data
                    ]
                )
            )
        )
    print(
        f"Subtitles en2zh saved to {out_filename} with"
        f" {len(all_en_data)} samples."
    )
    out_filename = os.path.join(
        OUTPUT_FOLDER,
        f"open-subtitles-zh2en.jsonl",
    )
    with open(out_filename, "w") as f:
        f.write(
            cc.convert(
                "\n".join(
                    [
                        json.dumps(row, ensure_ascii=False)
                        for row in all_zh_data
                    ]
                )
            )
        )
    print(
        f"Subtitles zh2en saved to {out_filename} with"
        f" {len(all_zh_data)} samples."
    )


def main(
    seed: int,
    buckets: List[int],
    article_len: int = 400,
    one_shot: bool = True,
    n_examples: int = 1000,
):
    assert one_shot, "Only one shot is supoorted."
    buckets = sorted(buckets)
    news_commentary(
        seed,
        buckets,
        n_examples=n_examples,
        article_len=article_len,
    )
    open_subtitles(seed, buckets, n_examples)
    ted_talk(seed, buckets, n_examples)


if __name__ == "__main__":
    cc = OpenCC("t2s")
    # main(111, buckets=[1000, 2000, 4000, 6000, 8000, 12000], n_examples=200)
    # main(111, buckets=[32000, 64000, 128000], n_examples=100)
    StrictFire(main)
