import sys
import os

ROOT_FOLDER = os.path.abspath(__file__)
for _ in range(2):
    ROOT_FOLDER = os.path.dirname(ROOT_FOLDER)
sys.path.append(ROOT_FOLDER)
import json
import random
from typing import List
from strictfire import StrictFire
from tqdm import tqdm
from xopen import xopen
from lost_in_the_middle.src.lost_in_the_middle.prompting import (
    Document,
    get_qa_prompt,
)
from copy import deepcopy
from xopen import xopen
from opencc import OpenCC

OUTPUT_FOLDER = os.path.join(ROOT_FOLDER, "data", "qa")
DATA_FOLDER = os.path.join(ROOT_FOLDER, "raw_data", "qa")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def hotpotqa(
    buckets: List[int],
    seed: int,
    one_shot: bool = True,
    n_examples: int = 1000,
):
    filename = os.path.join(
        DATA_FOLDER, "hotpotqa", "hotpot_dev_distractor_v1.json"
    )
    ori_data = json.load(open(filename, "r"))
    instruction = (
        "Answer the question based on the given paragraphs. Note that some"
        " paragraphs might be irrelevant."
    )
    example = """Paragraph 1: Pratia is a genus of flowering plants in the family Campanulaceae, native to Asia, Australia and New Zealand.
Paragraph 2: Sutherlandia is a genus of flowering plants in the family Fabaceae.
Question: Are Sutherlandia and Pratia in the same family?
Answer: no."""
    data = [[] for _ in buckets]
    for row in tqdm(ori_data, desc="Processing hotpotqa"):
        input_str = []
        for i, para in enumerate(row["context"], start=1):
            input_str.append(f"Paragraph {i}: " + " ".join(para[1]))
        input_str.append(f"Question: {row['question']}")
        input_str.append("Answer:")
        input_str = "\n".join(input_str)
        if one_shot:
            input_str = example + "\n\n" + input_str
        input_l = len(input_str.split())
        answer = row["answer"]
        total_l = input_l + len(answer.split())
        for j, bucket in enumerate(buckets):
            if total_l < bucket:
                data[j].append(
                    {
                        "instruction": instruction,
                        "input": input_str,
                        "answers": [row["answer"]],
                        "input_length": input_l,
                        "total_length": total_l,
                        "length_bucket": bucket,
                    }
                )
                break
    # {
    #     "instruction": "<task description>",
    #     "input": "<task input with one-shot example>",
    #     "answers": ["<answer 1>", "<answer 2>", ..., "<answer n>"],
    #     "input_length": <int, number of words in instruction and input separated by space>,
    #     "total_length": <int, number of words in instruction, input and gold answer separated by space>,
    #     "length_bucket": <int, the length bucket to which this instance belongs>
    # }

    seeder = random.Random(seed)
    out_data = []
    out_filename = os.path.join(OUTPUT_FOLDER, "hotpotqa.jsonl")
    for j, bucket in enumerate(buckets):
        if len(data[j]) == 0:
            print(f"HotpotQA data doesn't support length up to {bucket}.")
            continue

        if len(data[j]) > n_examples:
            data[j] = seeder.sample(data[j], k=n_examples)
        print(
            f"HotpotQA data has {len(data[j])} samples within the"
            f" {bucket} bucket."
        )
        out_data += data[j]

    with open(out_filename, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in out_data]
            )
        )
    print(
        f"HotpotQA data saved to {out_filename} with {len(out_data)} samples."
    )


def triviaqa(
    buckets: List[int],
    seed: int,
    one_shot: bool = True,
    n_examples: int = 1000,
):
    web_filename = os.path.join(
        DATA_FOLDER,
        "triviaqa",
        "qa",
        "web-dev.json",
    )
    wiki_filename = os.path.join(
        DATA_FOLDER, "triviaqa", "qa", "wikipedia-dev.json"
    )
    web_data = json.load(open(web_filename, "r"))
    wiki_data = json.load(open(wiki_filename, "r"))
    input_data = web_data["Data"] + wiki_data["Data"]
    wiki_root_folder = os.path.join(
        DATA_FOLDER, "triviaqa", "evidence", "wikipedia"
    )
    web_root_folder = os.path.join(DATA_FOLDER, "triviaqa", "evidence", "web")
    instruction = "Answer the question based on the given Context."
    example = """Context: The Port of Incheon () is the main port in South Korea, located in Incheon.
Question: In which country is the port of Incheon?
Answer: South Korea."""
    data = [[] for _ in buckets]
    for row in tqdm(input_data, desc="Processing triviaqa"):
        if row.get("SearchResults"):
            evidence_path = row["SearchResults"][0]["Filename"]
            evidence_fullpath = os.path.join(web_root_folder, evidence_path)
        elif row.get("EntityPages"):
            evidence_path = row["EntityPages"][0]["Filename"]
            evidence_fullpath = os.path.join(wiki_root_folder, evidence_path)
        else:
            tqdm.write(f"QuestionID={row['QuestionID']} has no evidence.")
            continue
        evidence: str = open(evidence_fullpath, "r").read().strip()
        question: str = row["Question"]
        answers: List[str] = (
            [row["Answer"]["Value"]]
            + [row["Answer"]["NormalizedValue"]]
            + row["Answer"]["Aliases"]
            + row["Answer"]["NormalizedAliases"]
        )
        input_str = f"Context: {evidence}\nQuestion: {question}\nAnswer:"
        if one_shot:
            input_str = example + "\n\n" + input_str

        if "\x00" in input_str:
            continue
        # for i, para in enumerate(row["context"], start=1):
        #     input_str += f"Paragraph {i}: " + " ".join(para[1])
        # input_str += f" Question: {row['question']}"
        # input_str += " Answer:"
        input_l = len((instruction + " " + input_str).split())
        max_answer = max(answers, key=lambda x: len(x))
        total_l = len(
            (instruction + " " + input_str + " " + max_answer).split()
        )
        for j, bucket in enumerate(buckets):
            if total_l < bucket:
                data[j].append(
                    {
                        "instruction": instruction,
                        "input": input_str,
                        "answers": answers,
                        "input_length": input_l,
                        "total_length": total_l,
                        "length_bucket": bucket,
                        "evidence_path": evidence_path,
                    }
                )
                break
    seeder = random.Random(seed)
    out_data = []
    out_filename = os.path.join(OUTPUT_FOLDER, "triviaqa.jsonl")
    for j, bucket in enumerate(buckets):
        if len(data[j]) == 0:
            print(f"TriviaQA data doesn't support length up to {bucket}.")
            continue
        if len(data[j]) > n_examples:
            data[j] = seeder.sample(data[j], k=n_examples)
        print(
            f"TriviaQA data has {len(data[j])} samples within the"
            f" {bucket} bucket."
        )
        out_data += data[j]
    with open(out_filename, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in out_data]
            )
        )
    print(
        f"TriviaQA data saved to {out_filename} with {len(out_data)} samples."
    )


def _create_multiqa_data_multi_index(
    num_total_documents_list: List[int],
    gold_indexes_list: List[List[int]],
):
    input_path = os.path.join(
        ROOT_FOLDER,
        "lost_in_the_middle",
        "nq-open-contriever-msmarco-retrieved-documents.jsonl.gz",
    )

    if any(
        num_total_documents < 2
        for num_total_documents in num_total_documents_list
    ):
        raise ValueError(
            "`num_total_documents` must be at least 2 for all entries"
        )
    for gold_indexes in gold_indexes_list:
        if any(gold_index < 0 for gold_index in gold_indexes):
            raise ValueError("`gold_index` must be at least 0")
    for gold_indexes, num_total_documents in zip(
        gold_indexes_list, num_total_documents_list
    ):
        if any(
            gold_index >= num_total_documents for gold_index in gold_indexes
        ):
            raise ValueError(
                "`gold_index` must be less than `num_total_documents`"
                f" ({num_total_documents})"
            )

    num_output_examples = 0
    all_outputs = [[] for _ in num_total_documents_list]
    with xopen(input_path) as fin:
        for line in tqdm(fin, desc="Reading nq data"):
            qa_retrieval_result = json.loads(line)
            # Get documents that don't contain the answer
            valid_distractors_with_retrieval_indices = [
                (idx, doc)
                for idx, doc in enumerate(qa_retrieval_result["ctxs"])
                if doc["hasanswer"] is False
            ]
            for i, (num_total_documents, gold_indexes) in enumerate(
                zip(num_total_documents_list, gold_indexes_list)
            ):
                if num_total_documents > len(
                    valid_distractors_with_retrieval_indices
                ):
                    continue

                # Take the top `num_total_documents - 1` distractors
                distractor_docs_with_retrieval_indices = deepcopy(
                    valid_distractors_with_retrieval_indices[
                        : num_total_documents - 1
                    ]
                )
                for (
                    original_retrieval_index,
                    distractor_doc,
                ) in distractor_docs_with_retrieval_indices:
                    distractor_doc["original_retrieval_index"] = (
                        original_retrieval_index
                    )
                    distractor_doc["isgold"] = False
                distractor_docs = [
                    x[1] for x in distractor_docs_with_retrieval_indices
                ]

                content_selection_example = deepcopy(qa_retrieval_result)
                gold_chunk = {
                    "title": qa_retrieval_result["nq_annotated_gold"]["title"],
                    "text": qa_retrieval_result["nq_annotated_gold"][
                        "chunked_long_answer"
                    ],
                    "hasanswer": True,
                    "isgold": True,
                }
                ctxs = distractor_docs.copy()
                # Insert the gold chunk at thet specific index
                outputs = all_outputs[i]
                gold_index = gold_indexes[len(outputs) % len(gold_indexes)]
                ctxs.insert(gold_index, gold_chunk)

                content_selection_example["ctxs"] = ctxs
                outputs.append(content_selection_example)
            all_outputs.append(outputs)
    return all_outputs


def multi_document_qa(
    seed: int, buckets: List[int], one_shot: bool, n_examples: int
):
    length2td = {
        1000: 5,
        2000: 10,
        4000: 30,
        6000: 40,
        8000: 60,
        16000: 120,
        32000: 200,
        64000: 400,
        128000: 800,
    }
    tds = []
    for b in buckets:
        if b not in length2td:
            raise ValueError(
                f"Specific the number of documents used for length={b}."
            )
        tds.append(length2td[b])
    # for _td in [5, 10, 30, 40, 60, 80, 90]:
    buckets = {bucket: [] for bucket in buckets}
    indexes_list = []
    for _td in tds:
        if _td >= 200:
            indexes_list.append([0] + list(range(9, _td, 10)))
        else:
            indexes_list.append([0] + list(range(4, _td, 5)))
    rows_list = _create_multiqa_data_multi_index(tds, indexes_list)

    for i, _td in enumerate(tds):
        td_bucket_count = {bucket: 0 for bucket in buckets}
        example = (
            f"Document [{_td+1}](Title: LLaMA) LLaMA (Large Language Model"
            " Meta AI) is a family of large language models (LLMs), released"
            " by Meta AI starting in February 2023.\n\nQuestion: when is"
            " LLaMA released\nAnswer: February 2023."
        )
        # if _td >= 200:
        #     indexes = [0] + list(range(9, _td, 10))
        # else:
        #     indexes = [0] + list(range(4, _td, 5))
        # rows = _create_multiqa_data_multi_index(_td, indexes)
        rows = rows_list[i]
        for input_example in tqdm(
            rows, desc=f"Processing total document={_td}"
        ):
            question = input_example["question"]
            documents = []
            for ctx in deepcopy(input_example["ctxs"]):
                documents.append(Document.from_dict(ctx))
            if not documents:
                raise ValueError(
                    f"Did not find any documents for example: {input_example}"
                )
            prompt = get_qa_prompt(
                question,
                documents,
                mention_random_ordering=False,
                query_aware_contextualization=False,
            )
            instruction = (
                "Write a high-quality answer for the given question"
                " using only the provided search results (some of"
                " which might be irrelevant)."
            )
            input = prompt.split(instruction)[1].strip()
            if one_shot:
                input, question = input.rsplit("\n\n", maxsplit=1)
                input += f"\n{example}\n{question}"
            input_example["instruction"] = instruction
            input_example["input"] = input

            l = len((instruction + " " + input).split())
            input_example["input_length"] = l
            max_answer = max(input_example["answers"], key=lambda x: len(x))
            l = len((instruction + " " + input + " " + max_answer).split())
            input_example["total_length"] = l
            # data.append(input_example)
            for bucket in buckets:
                if l < bucket:
                    input_example["length_bucket"] = bucket
                    buckets[bucket].append(input_example)
                    td_bucket_count[bucket] += 1
                    break
    seeder = random.Random(seed)
    out_data = []
    out_filename = os.path.join(OUTPUT_FOLDER, "nq-open.jsonl")
    for bucket, data in buckets.items():
        if len(data) == 0:
            print(
                "Multi-Document QA data doesn't have data with length"
                f" {bucket}."
            )
            continue
        if len(data) > n_examples:
            data = seeder.sample(data, k=n_examples)
        print(
            f"Multi-Document QA has {len(data)} samples within the"
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
        f"Multi-Document QA data saved to {out_filename} with"
        f" {len(out_data)} samples."
    )


def drcd(
    seed: int, buckets: List[int], one_shot: bool, n_examples: int = 1000
):
    length2td = {
        1000: 1,
        2000: 3,
        4000: 5,
        6000: 10,
        8000: 15,
        12000: 25,
        16000: 30,
        24000: 40,
        32000: 60,
        64000: 100,
        128000: 200,
    }

    input_path = os.path.join(
        DATA_FOLDER,
        f"DRCD",
        f"DRCD_test.json",
    )
    seeder = random.Random(seed)
    in_data = json.load(open(input_path, "r"))
    paragraphs = [
        {"id": para["id"], "context": para["context"], "title": item["title"]}
        for item in in_data["data"]
        for para in item["paragraphs"]
    ]
    tds = []
    for b in buckets:
        if b not in length2td:
            raise ValueError(
                f"Specific the number of documents used for length={b}."
            )
        tds.append(length2td[b])
    instruction = (
        "請使用下列提供的段落為以下的問題寫一個高質量的答案(部分段落可能與問題無關)。"
    )
    # for _td in tqdm([1, 3, 5, 10, 15, 20], desc="Processing drcd"):
    buckets = {bucket: [] for bucket in buckets}
    for _td in tqdm(tds, desc="Processing drcd"):
        # for _td in tqdm([1], desc="Processing drcd"):
        td_bucket_count = {bucket: [] for bucket in buckets}
        example = (
            f"文章 [{_td+1}](標題: 文心一言)"
            " 文心一言是由百度公司開發的聊天機械人，能夠與人互動、"
            "回答問題及協作創作。"
            "該產品被傳媒稱為國際著名聊天機械人ChatGPT的中國版及其競爭對手。"
            "目前已開放用戶申請加入體驗，"
            "但現僅支援百度帳號繫結中國大陸電話號碼的企業級用戶和個人用戶。"
            "\n\n問題: 文心一言是由哪間公司開發的？\n答案: 百度公司。"
        )
        if _td == 1:
            n_list = [0]
        elif _td in [2, 3]:
            n_list = [0, 1]
        else:
            n_list = [0] + list(range(4, _td, 5))

        for _n in n_list:
            for item in in_data["data"]:
                for para in item["paragraphs"]:
                    for qa in para["qas"]:
                        question = qa["question"]
                        answers = [ans["text"] for ans in qa["answers"]]
                        contexts = seeder.sample(
                            [
                                para2
                                for para2 in paragraphs
                                if para2["id"] != para["id"]
                            ],
                            k=_td - 1,
                        )
                        contexts.insert(
                            _n,
                            {
                                "context": para["context"],
                                "id": para["id"],
                                "title": item["title"],
                            },
                        )
                        input_str = []
                        for i, doc in enumerate(contexts, start=1):
                            input_str.append(
                                f"文章 [{i}](標題: {doc['title']})"
                                f" {doc['context']}"
                            )
                        input_str = "\n".join(input_str)
                        if one_shot:
                            input_str += (
                                f"\n{example}\n問題: {question}\n答案:"
                            )
                        else:
                            input_str += f"\n\n問題: {question}\n答案:"

                        input_l = len(instruction) + len(input_str)
                        max_answer = max(answers, key=lambda x: len(x))
                        total_l = (
                            len(instruction) + len(input_str) + len(max_answer)
                        )
                        for bucket in buckets:
                            if total_l < bucket:
                                data_dict = {
                                    "ctx": contexts,
                                    "question": question,
                                    "answers": answers,
                                    "instruction": instruction,
                                    "input": input_str,
                                    "input_length": input_l,
                                    "total_length": total_l,
                                    "length_bucket": bucket,
                                    "gold_index": _n,
                                }
                                buckets[bucket].append(data_dict)
                                td_bucket_count[bucket].append(total_l)
                                break
        # td_bucket_str = {
        #     b: f"{np.mean(rows):.0f}" for b, rows in td_bucket_count.items()
        # }
        # print(
        #     f"n_documents={_td} has the following length distribution: {td_bucket_str}"
        # )
    out_data = []
    out_filename = os.path.join(OUTPUT_FOLDER, "drcd_semantic-single.jsonl")
    for bucket, data in buckets.items():
        if len(data) > n_examples:
            data = seeder.sample(data, k=n_examples)
        if len(data) > 0:
            out_data += data
            print(f"drcd has {len(data)} samples within the {bucket} bucket.")
        else:
            print(f"drcd doesn't support length of {bucket}.")
    with open(out_filename, "w") as f:
        f.write(
            "\n".join(
                [
                    cc.convert(json.dumps(row, ensure_ascii=False))
                    for row in out_data
                ]
            )
        )
    print(f"drcd data saved to {out_filename} with {len(out_data)} samples.")


def dureader(
    buckets: List[int],
    seed: int,
    one_shot: bool = True,
    n_examples: int = 1000,
):
    example = """文章 [1](标题：民生银行信用卡申请快两个月了还没建档民生银行信用卡申请快两个月了还没建档) 你的信用记录在观察期。全中国就民生会这样做。耐心等待 有就有 无则无
文章 [2](标题：民生银行信用卡为什么半个月了都没有记录) 正常，办卡的人多，就慢！打电话加急没开通
问题：民生银行信用卡为什么还没有建档
答案：你的信用记录在观察期。"""
    inst = (
        "请使用以下提供的段落为以下问题写出一个高质量的答案（部分段落可能与问题相关）。"
    )
    in_filename = os.path.join(DATA_FOLDER, "dureader/devset/search.dev.json")
    ori_data = [json.loads(l) for l in open(in_filename)]
    in_filename = os.path.join(DATA_FOLDER, "dureader/devset/zhidao.dev.json")
    ori_data += [json.loads(l) for l in open(in_filename)]
    data = [[] for _ in buckets]
    for row in tqdm(ori_data, desc="Processing dureader"):
        if row["question_type"] in ["DESCRIPTION", "ENTITY"]:
            question = row["question"]
            paras = [
                f"文章 [{i}](标题：{doc['title']})"
                f" {''.join(doc['paragraphs'])}"
                for i, doc in enumerate(row["documents"], 1)
            ]
            # List of str
            answers = row["answers"]
            if (
                not answers
                or not paras
                or max([len(ans) for ans in answers]) > 300
            ):
                continue
            input_str = "\n".join(paras) + f"\n问题：{question}\n答案："
            if one_shot:
                input_str = example + "\n\n" + input_str
            input_length = len(inst + input_str)
            total_length = input_length + max([len(ans) for ans in answers])
            for j, bucket in enumerate(buckets):
                if total_length < bucket:
                    data[j].append(
                        {
                            "instruction": inst,
                            "input": input_str,
                            "answers": answers,
                            "input_length": input_length,
                            "total_length": total_length,
                            "length_bucket": bucket,
                            "question_id": row["question_id"],
                        }
                    )
                    break
    seeder = random.Random(seed)
    out_data = []
    out_filename = os.path.join(OUTPUT_FOLDER, "dureader.jsonl")
    for j, bucket in enumerate(buckets):
        if len(data[j]) == 0:
            print(f"Dureader data doesn't have data with length {bucket}.")
            continue
        if len(data[j]) > n_examples:
            data[j] = seeder.sample(data[j], k=n_examples)
        print(
            f"Dureader data has {len(data[j])} samples within the"
            f" {bucket} bucket."
        )
        out_data += data[j]
    with open(out_filename, "w") as f:
        f.write(
            cc.convert(
                "\n".join(
                    [json.dumps(row, ensure_ascii=False) for row in out_data]
                )
            )
        )
    print(
        f"Dureader data saved to {out_filename} with {len(out_data)} samples."
    )


def c3(
    buckets: List[int],
    seed: int,
    one_shot: bool,
    n_examples: int = 1000,
):
    example = """段落1: 我跟你们这些大学生不一样，我必须一边工作一边学习汉语，白天上班，晚上上课，虽然很累，但是觉得很有意思。
段落2: 今天我在电梯里遇到了校长，他问我同学们的学习怎么样、生活怎么样。校长非常关心大家，我们很喜欢这位校长。
问题：根據段落2，大家喜欢校长是因为： A. 能在电梯里遇到校长 B. 校长经常想我们 C. 校长关心同学们
答案：C. 校长关心同学们"""
    inst = (
        "下面我会提供许多文章的段落。我会在最后提出的问题，"
        "让你根据某个指定的段落在数个中选择出正确的答案。"
    )

    in_filename = os.path.join(DATA_FOLDER, "c3/c3-m-test.json")
    data = json.load(open(in_filename))

    processed_data = []
    for row in tqdm(data, desc="Processing c3"):
        document = row[0][0]
        if len(document) < 100:
            continue
        processed_data.append({"document": document, "questions": row[1]})

    seeder = random.Random(seed)
    prev_bucket = 0
    n_docs = {32000: 100, 64000: 160, 128000: 300}
    out_data = []
    bucket_count = {bucket: 0 for bucket in buckets}
    for bucket in buckets:
        bucket_processed_data = deepcopy(processed_data)
        data = []
        if bucket in n_docs:
            n_doc = n_docs[bucket]
        else:
            n_doc = (bucket - 200) // 300

        pbar = tqdm(desc=f"Processing C3 bucket={bucket}", total=n_examples)
        while len(data) < n_examples:
            gold_indexes = [0] + list(range(4, n_doc, 5))
            for gold_index in gold_indexes:
                trial = []
                while True:
                    docs = seeder.sample(bucket_processed_data, k=n_doc)
                    questions = docs[gold_index]["questions"]
                    candidate_i = seeder.randrange(len(questions))
                    input_str = [
                        f"段落{i}: " + doc["document"].replace("\n", "")
                        for i, doc in enumerate(docs, 1)
                    ]
                    input_str = "\n".join(input_str)
                    question, choices, answer = (
                        questions[candidate_i]["question"],
                        questions[candidate_i]["choice"],
                        questions[candidate_i]["answer"],
                    )
                    answer = f"{chr(ord('A')+choices.index(answer))}. {answer}"
                    choice = " ".join(
                        [
                            f"{chr(ord('A')+i)}. {c}"
                            for i, c in enumerate(choices)
                        ]
                    )
                    input_str += (
                        f"\n问题：根據段落{gold_index+1}，{question} {choice}\n答案："
                    )
                    if one_shot:
                        input_str = example + "\n\n" + input_str
                    input_l = len(input_str)
                    total_l = input_l + len(answer)
                    if total_l > prev_bucket and total_l <= bucket:
                        data.append(
                            {
                                "instruction": inst,
                                "input": input_str,
                                "answers": [answer],
                                "input_length": input_l,
                                "total_length": total_l,
                                "gold_index": gold_index,
                                "length_bucket": bucket,
                            }
                        )
                        bucket_count[bucket] += 1
                        # pop question and document
                        # docs[gold_index]["questions"].pop(candidate_i)
                        # if len(docs[gold_index]["questions"]) == 0:
                        #     bucket_processed_data.pop(
                        #         bucket_processed_data.index(docs[gold_index])
                        #     )
                        break
                    else:
                        if len(trial) > 200:
                            raise ValueError(
                                f"Can't create data within the {prev_bucket} <"
                                f" {bucket}. The lengths are {trial}"
                            )
                        trial.append(total_l)
                pbar.update(1)
        out_data += data
        pbar.close()
        prev_bucket = bucket
    out_filename = os.path.join(OUTPUT_FOLDER, "c3.jsonl")
    with open(out_filename, "w") as f:
        f.write(
            cc.convert(
                "\n".join(
                    [json.dumps(row, ensure_ascii=False) for row in out_data]
                )
            )
        )
    print(f"C3 data saved to {out_filename} with {len(out_data)} samples.")


def newsqa(
    buckets: List[int],
    seed: int,
    one_shot: bool,
    n_examples: int,
):
    example = """Article 1: Congressman Jared Polis
(D) Colorado: District 02
Congressman Jason Chaffetz
(R) Utah: District 03
Article 2: (CNN) -- CNN explores Beijing's underground music scene and the bands making the rest of the world sit up and listen.
Question: Based on article 1, what did the new legislators document for CNN?
Answer: Utah."""
    inst = (
        "I will provide you multiple news article below. Answer the question"
        " based on a specfic article."
    )

    in_filename = os.path.join(
        DATA_FOLDER, "newsqa/combined-newsqa-data-v1.json"
    )
    data = json.load(open(in_filename))["data"]

    processed_data = []
    for row in tqdm(data, desc="Processing newsqa"):
        if "86bd905861391cbd3a98de15c83768b6d1400304" in row["storyId"]:
            continue
        document = row["text"]
        questions = []
        for question in row["questions"]:
            if "s" in question["consensus"]:
                questions.append(
                    {
                        "question": question["q"],
                        "answer": document[
                            question["consensus"]["s"] : question["consensus"][
                                "e"
                            ]
                        ],
                    }
                )
        if questions:
            processed_data.append(
                {
                    "document": document,
                    "questions": questions,
                    "id": row["storyId"],
                }
            )

    seeder = random.Random(seed)
    prev_bucket = 0
    n_docs = {32000: 50, 64000: 100, 128000: 200}
    out_data = []
    bucket_count = {bucket: 0 for bucket in buckets}
    for bucket in buckets:
        bucket_processed_data = deepcopy(processed_data)
        if bucket in n_docs:
            n_doc = n_docs[bucket]
        elif one_shot:
            n_doc = (bucket - len(example) - len(inst)) // 500
        else:
            n_doc = (bucket - len(inst)) // 500

        pbar = tqdm(
            desc=f"Processing newsqa bucket={bucket}", total=n_examples
        )
        data = []
        while len(data) < n_examples:
            gold_indexes = [0] + list(range(4, n_doc, 5))
            for gold_index in gold_indexes:
                trial = []
                while True:
                    docs = seeder.sample(bucket_processed_data, k=n_doc)
                    questions = docs[gold_index]["questions"]
                    candidate_i = seeder.randrange(len(questions))
                    input_str = [
                        f"Article {i}: " + doc["document"].replace("\n", "")
                        for i, doc in enumerate(docs, 1)
                    ]
                    input_str = "\n".join(input_str)
                    question, answer = (
                        questions[candidate_i]["question"],
                        questions[candidate_i]["answer"],
                    )
                    question = question[0].lower() + question[1:]
                    input_str += (
                        f"\nQuestion: Based on article {gold_index+1},"
                        f" {question}\nAnswer:"
                    )
                    if one_shot:
                        input_str = example + "\n\n" + input_str
                    input_l = len(input_str.split())
                    total_l = input_l + len(answer.split())
                    if total_l > prev_bucket and total_l <= bucket:
                        data.append(
                            {
                                "instruction": inst,
                                "input": input_str,
                                "answers": [answer],
                                "input_length": input_l,
                                "total_length": total_l,
                                "length_bucket": bucket,
                                "gold_index": gold_index,
                            }
                        )
                        bucket_count[bucket] += 1
                        break
                    else:
                        if len(trial) > 200:
                            raise ValueError(
                                f"Can't create data within the {prev_bucket} <"
                                f" {bucket}. The lengths are {trial}"
                            )
                        trial.append(total_l)
                pbar.update(1)
        out_data += data
        pbar.close()
        prev_bucket = bucket

    out_filename = os.path.join(OUTPUT_FOLDER, "newsqa.jsonl")
    with open(out_filename, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in out_data]
            )
        )
    print(f"Newsqa data saved to {out_filename} with {len(out_data)} samples.")
    print(f"Newsqa data distribution: {bucket_count}")


def duorc(
    buckets: List[int],
    seed: int,
    one_shot: bool = True,
    n_examples: int = 1000,
):
    example = """Plot (Old Joy): Old Joy tells the story of two friends, Kurt (Will Oldham) and Mark (Daniel London), as they reunite for a weekend camping trip in the Cascade mountain range and Bagby Hot Springs, east of Portland, Oregon. Kurt lives a hand-to-mouth hippie lifestyle, while Mark has moved on from that scene and gotten a proper job and a house. The film is a story of friendship, loss and alienation. For Mark, the weekend outing offers a respite from the pressure of his imminent fatherhood. Tagging along for the ride is Lucy, Mark's dog (played by Reichardt's dog of the same name).
Plot ( Donkey Xote): This is a true adventure comedy. The donkey, Rucio, tells the true story of Don Quixote and defends the idea that he wasnt mad, but in reality, Quixote was a very intelligent, passionate and enthusiastic fellow. We then follow Don Quixote, his squire, Sancho Panza (Quixotes best friend and the wealthiest man in town), Sanchos donkey, Rucio (who wants to be a horse) and a real horse, Quixotes faithful steed, Rocinante (who hates leaving his stable) on their adventure to duel the Knight of the Moon where, if Quixote wins the duel the true identity of Dulcinea will be revealed.
Question: What was the name of the hot springs they visited?
Answer: Bagby Hot Springs."""
    instruction = (
        "Write a high-quality answer for the given question using only the"
        " provided plots (some of which might be irrelevant)."
    )
    in_filename = os.path.join(
        DATA_FOLDER, "duorc/dataset/ParaphraseRC_test.json"
    )
    data = json.load(open(in_filename))
    processed_data = []
    for row in tqdm(data, desc="Processing duorc"):
        plot = row["plot"]
        title = row["title"]
        questions = []
        for q in row["qa"]:
            if q["no_answer"]:
                continue
            questions.append(
                {
                    "q": q["question"],
                    "a": q["answers"],
                }
            )
        if questions:
            processed_data.append(
                {
                    "plot": plot,
                    "title": title,
                    "questions": questions,
                }
            )

    seeder = random.Random(seed)
    prev_bucket = 0
    n_docs = {32000: 35, 64000: 70, 128000: 120}
    out_data = []
    bucket_count = {bucket: 0 for bucket in buckets}
    for bucket in buckets:
        data = []
        if bucket in n_docs:
            n_doc = n_docs[bucket]
        elif one_shot:
            n_doc = max(
                (bucket - len(example.split()) - len(instruction.split()))
                // 800,
                2,
            )
        else:
            n_doc = max((bucket - len(instruction.split())) // 800, 2)

        pbar = tqdm(desc=f"Processing duorc bucket={bucket}", total=n_examples)
        while len(data) < n_examples:
            gold_indexes = [0] + list(range(4, n_doc, 5))
            for gold_index in gold_indexes:
                trial = []
                while True:
                    docs = seeder.sample(processed_data, k=n_doc)
                    questions = docs[gold_index]["questions"]
                    candidate_i = seeder.randrange(len(questions))
                    input_str = [
                        f"Plot ({doc['title']}): "
                        + doc["plot"].replace("\n", "")
                        for i, doc in enumerate(docs, 1)
                    ]
                    input_str = "\n".join(input_str)
                    question, answers = (
                        questions[candidate_i]["q"],
                        questions[candidate_i]["a"],
                    )
                    input_str += f"\nQuestion: {question}\nAnswer:"
                    if one_shot:
                        input_str = example + "\n\n" + input_str
                    input_l = len(input_str.split())
                    total_l = input_l + max([len(a.split()) for a in answers])
                    if total_l > prev_bucket and total_l <= bucket:
                        data.append(
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
                        bucket_count[bucket] += 1
                        break
                    else:
                        if len(trial) > 200:
                            raise ValueError(
                                f"Can't create data within the {prev_bucket} <"
                                f" {bucket}. The lengths are {trial}"
                            )
                        trial.append(total_l)
                pbar.update(1)
        out_data += data
        pbar.close()
        prev_bucket = bucket

    out_filename = os.path.join(OUTPUT_FOLDER, "duorc.jsonl")
    with open(out_filename, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(row, ensure_ascii=False) for row in out_data]
            )
        )
    print(f"Duorc data saved to {out_filename} with {len(out_data)} samples.")
    print(f"Duorc data distribution: {bucket_count}")


def main(
    seed: int,
    buckets: List[int],
    one_shot: bool = False,
    n_examples: int = 1000,
):
    buckets = sorted(buckets)
    hotpotqa(buckets, seed, one_shot, n_examples)
    dureader(buckets, seed, one_shot=one_shot, n_examples=n_examples)
    triviaqa(buckets, seed, one_shot=one_shot, n_examples=n_examples)
    multi_document_qa(seed, buckets, one_shot, n_examples)
    drcd(seed, buckets, one_shot, n_examples=n_examples)
    c3(buckets, seed, one_shot, n_examples=n_examples)
    newsqa(buckets, seed, one_shot, n_examples=n_examples)
    duorc(buckets, seed, one_shot, n_examples=n_examples)


if __name__ == "__main__":
    cc = OpenCC("t2s")
    StrictFire(main)
