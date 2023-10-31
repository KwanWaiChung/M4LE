import os
import sys

sys.setrecursionlimit(12000 * 12000)

ROOT_FOLDER = os.path.abspath(__file__)
for _ in range(2):
    ROOT_FOLDER = os.path.dirname(ROOT_FOLDER)
sys.path.append(ROOT_FOLDER)
from strictfire import StrictFire
from tqdm import tqdm
from typing import List, Dict, Callable
from rouge import Rouge
from rouge_chinese import Rouge as ZHRouge
from utils.bleu import compute
from typing import Literal

import json
import collections
import string
import re
import jieba
import numpy as np
import pandas as pd

jieba.initialize()
en_rouge = Rouge()
zh_rouge = ZHRouge()

"""Official evaluation script for SQuAD version 2.0.

In addition to basic functionality, we also compute additional statistics and
plot precision-recall curves if an additional na_prob.json file is provided.
This file is expected to map question ID's to the model's predicted probability
that a question is unanswerable.
"""


def _normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _get_tokens(s, lang):
    if not s:
        return []
    if lang == "en":
        return _normalize_answer(s).split()
    elif lang == "zh":
        return list(jieba.cut(_normalize_answer(s)))


def marc_preprocess(data: List[Dict]):
    for row in data:
        if "gen_resp" not in row:
            continue
        gen_resp = row["gen_resp"].split(".")[0].strip()
        gen_resp = [tmp.strip() for tmp in sorted(gen_resp.split(","))]
        answer = [
            ans.strip()
            for ans in sorted(row["answers"][0].split(".")[0].split(","))
        ]
        row["gen_resp"] = gen_resp
        row["answers"] = answer
    return data


def online_shopping_preprocess(data: List[Dict]):
    for row in data:
        if "gen_resp" not in row:
            continue
        gen_resp = row["gen_resp"].split("。")[0].strip()
        gen_resp = [tmp.strip() for tmp in sorted(gen_resp.split("，"))]
        answer = [ans.strip() for ans in sorted(row["answers"][0].split("，"))]
        row["gen_resp"] = gen_resp
        row["answers"] = answer
    return data


def news2016_preprocess(data: List[Dict]):
    for row in data:
        if "gen_resp" not in row:
            continue
        gen_resp = row["gen_resp"]
        if '"' in gen_resp:
            gen_resp = row["gen_resp"].split('"')[1].strip()
        if '"' in gen_resp:
            gen_resp = row["gen_resp"].split('"')[0].strip()
        row["gen_resp"] = gen_resp
    return data


def thucnews_explicit_preprocess(data: List[Dict]):
    for row in data:
        if "gen_resp" not in row:
            continue
        gen_resp = row["gen_resp"].split("。")[0].strip()
        gen_resp = [tmp.strip() for tmp in sorted(gen_resp.split("，"))]
        answer = [ans.strip() for ans in sorted(row["answers"][0].split("，"))]
        row["gen_resp"] = gen_resp
        row["answers"] = answer
    return data


def mnds_news_explicit_preprocess(data: List[Dict]):
    for row in data:
        if "gen_resp" not in row:
            continue
        gen_resp = row["gen_resp"].split(".")[0].strip()
        gen_resp = [tmp.strip() for tmp in sorted(gen_resp.split(","))]
        answer = [ans.strip() for ans in sorted(row["answers"][0].split(","))]
        row["gen_resp"] = gen_resp
        row["answers"] = answer
    return data


def mnds_news_semantic_preprocess(data: List[Dict]):
    for row in data:
        answer = row["answers"][0].split(".")[0]
        row["answers"] = [answer]
    return data


def thucnews_semantic_preprocess(data: List[Dict]):
    for row in data:
        answer = row["answers"][0].split("。")[0]
        row["answers"] = [answer]
    return data


def compute_f1(a_gold: str, a_pred: str, lang: str = "en"):
    gold_toks = _get_tokens(a_gold, lang)
    pred_toks = _get_tokens(a_pred, lang)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate_by_dict(
    data,
    end_tokens: List[str] = [],
    metric: Literal[
        "bleu", "single_acc", "exact_match", "acc", "rouge", "qa_f1", "cls_f1"
    ] = "acc",
    lang: str = "en",
    preprocess: Callable[[List[Dict]], List[Dict]] = None,
):
    scores = {}
    complete_count = 0
    total = {}
    predictions = {}
    references = {}
    # for i in tqdm(data, desc="Evaluating"):
    if preprocess:
        data = preprocess(data)
    for i in data:
        if "gen_resp" in i:
            gen_resp = i["gen_resp"]
            for t in end_tokens:
                gen_resp = gen_resp.split(t)[0].strip()
            if isinstance(gen_resp, list):
                gen_resp = [r.strip() for r in gen_resp]
            else:
                gen_resp = gen_resp.strip()

            if metric == "acc":
                scores.setdefault(i["length_bucket"], []).append(
                    any(
                        [
                            str(ans).lower() in gen_resp.lower()
                            for ans in i["answers"]
                        ]
                    )
                )
            elif metric == "exact_match":
                scores.setdefault(i["length_bucket"], []).append(
                    any(
                        [
                            str(ans).lower().strip()
                            == gen_resp.lower().strip()
                            for ans in i["answers"]
                        ]
                    )
                )
            elif metric == "single_acc":
                total.setdefault(i["length_bucket"], []).append(
                    len(i["answers"])
                )
                scores.setdefault(i["length_bucket"], []).append(
                    len(set(i["answers"]) & set(gen_resp))
                )
            elif metric == "rouge":
                try:
                    if lang == "en":
                        score = max(
                            [
                                en_rouge.get_scores(hyps=gen_resp, refs=ref)[
                                    0
                                ]["rouge-l"]["r"]
                                for ref in i["answers"]
                                if ref
                            ],
                        )
                    elif lang == "zh":
                        score = max(
                            [
                                zh_rouge.get_scores(
                                    hyps=" ".join(jieba.cut(gen_resp)),
                                    refs=" ".join(jieba.cut(ref)),
                                )[0]["rouge-l"]["r"]
                                for ref in i["answers"]
                                if ref
                            ],
                        )
                    scores.setdefault(i["length_bucket"], []).append(score)
                except ValueError:
                    scores.setdefault(i["length_bucket"], []).append(0)
            elif metric == "qa_f1":
                f1s = [compute_f1(ans, gen_resp, lang) for ans in i["answers"]]
                f1 = max(f1s)
                scores.setdefault(i["length_bucket"], []).append(f1)
            elif metric == "cls_f1":
                assert type(gen_resp) == list
                assert type(i["answers"]) == list
                n_corr = len(set(i["answers"]) & set(gen_resp))
                prec = n_corr / len(set(gen_resp))
                rec = n_corr / len(set(i["answers"]))
                if prec + rec:
                    scores.setdefault(i["length_bucket"], []).append(
                        2 * prec * rec / (prec + rec)
                    )
                else:
                    scores.setdefault(i["length_bucket"], []).append(0)
            elif metric == "bleu":
                predictions.setdefault(i["length_bucket"], []).append(gen_resp)
                references.setdefault(i["length_bucket"], []).append(
                    i["answers"]
                )
            else:
                raise NotImplementedError()
            complete_count += 1
    multiplier = 1
    if metric == "bleu":
        length_buckets = sorted(predictions)
        return {
            bucket: compute(
                predictions=predictions[bucket],
                references=references[bucket],
                lang=lang,
            )["bleu"]
            * multiplier
            if predictions[bucket]
            else np.nan
            for bucket in length_buckets
        }
    elif metric == "single_acc":
        length_buckets = sorted(scores)
        return {
            bucket: sum(scores[bucket]) * multiplier / total[bucket]
            if scores[bucket]
            else np.nan
            for bucket in length_buckets
        }
    else:
        length_buckets = sorted(scores)
        return {
            bucket: sum(scores[bucket]) * multiplier / len(scores[bucket])
            if scores[bucket]
            else np.nan
            for bucket in length_buckets
        }


def evaluate():
    tasks_param = {
        "arxiv": {
            "end_tokens": [
                "\n",
            ],
            "metric": "rouge",
            "lang": "en",
            "preprocess": None,
        },
        "pubmed": {
            "end_tokens": [
                "\n",
            ],
            "metric": "rouge",
            "lang": "en",
            "preprocess": None,
        },
        "booksum": {
            "end_tokens": [
                "\n",
            ],
            "metric": "rouge",
            "lang": "en",
            "preprocess": None,
        },
        "cnnnews": {
            "end_tokens": [
                "\n",
            ],
            "metric": "rouge",
            "lang": "en",
            "preprocess": None,
        },
        "wikihow": {
            "end_tokens": [
                "\n",
            ],
            "metric": "rouge",
            "lang": "en",
            "preprocess": None,
        },
        "cnewsum": {
            "end_tokens": ["\n", "新闻"],
            "metric": "rouge",
            "lang": "zh",
            "preprocess": None,
        },
        "clts+": {
            "end_tokens": ["\n", "新闻"],
            "metric": "rouge",
            "lang": "zh",
            "preprocess": None,
        },
        "cepsum": {
            "end_tokens": [
                "\n",
            ],
            "metric": "rouge",
            "lang": "zh",
            "preprocess": None,
        },
        "lcsts": {
            "end_tokens": [
                "\n",
            ],
            "metric": "rouge",
            "lang": "zh",
            "preprocess": None,
        },
        "ncls": {
            "end_tokens": [
                "\n",
            ],
            "metric": "rouge",
            "lang": "zh",
            "preprocess": None,
        },
        "triviaqa": {
            "end_tokens": ["\n", "."],
            "metric": "acc",
            "lang": "en",
            "preprocess": None,
        },
        "newsqa": {
            "end_tokens": ["\n", "."],
            "metric": "acc",
            "lang": "en",
            "preprocess": None,
        },
        "duorc": {
            "end_tokens": ["\n", "."],
            "metric": "acc",
            "lang": "en",
            "preprocess": None,
        },
        "nq-open": {
            "end_tokens": ["\n", "."],
            "metric": "acc",
            "lang": "en",
            "preprocess": None,
        },
        "c3": {
            "end_tokens": ["\n", "。"],
            "metric": "acc",
            "lang": "zh",
            "preprocess": None,
        },
        "dureader": {
            "end_tokens": ["\n", "。"],
            "metric": "acc",
            "lang": "zh",
            "preprocess": None,
        },
        "open-subtitles-en2zh": {
            "end_tokens": [],
            "metric": "bleu",
            "lang": "zh",
            "preprocess": None,
        },
        "open-subtitles-zh2en": {
            "end_tokens": [],
            "metric": "bleu",
            "lang": "en",
            "preprocess": None,
        },
        "news-commentary-en2zh": {
            "end_tokens": [],
            "metric": "bleu",
            "lang": "zh",
            "preprocess": None,
        },
        "news-commentary-zh2en": {
            "end_tokens": [],
            "metric": "bleu",
            "lang": "en",
            "preprocess": None,
        },
        "tedtalks-en2zh": {
            "end_tokens": [],
            "metric": "bleu",
            "lang": "zh",
            "preprocess": None,
        },
        "tedtalks-zh2en": {
            "end_tokens": [],
            "metric": "bleu",
            "lang": "en",
            "preprocess": None,
        },
        "wow": {
            "end_tokens": [",", "\n"],
            "metric": "acc",
            "lang": "en",
            "preprocess": None,
        },
        "marc": {
            "end_tokens": [],
            "metric": "cls_f1",
            "lang": "en",
            "preprocess": marc_preprocess,
        },
        "online-shopping": {
            "end_tokens": [],
            "metric": "cls_f1",
            "lang": "en",
            "preprocess": online_shopping_preprocess,
        },
        "wikitext-103": {
            "end_tokens": ["."],
            "metric": "exact_match",
            "lang": "en",
            "preprocess": None,
        },
        "wiki2019zh": {
            "end_tokens": ["。"],
            "metric": "exact_match",
            "lang": "zh",
            "preprocess": None,
        },
        "news2016": {
            "end_tokens": ["\n"],
            "metric": "rouge",
            "lang": "zh",
            "preprocess": news2016_preprocess,
        },
        "bigpatent_global_cls": {
            "end_tokens": [":"],
            "metric": "acc",
            "lang": "en",
            "preprocess": None,
        },
        "bigpatent_global_sum": {
            "end_tokens": ["\n"],
            "metric": "rouge",
            "lang": "en",
            "preprocess": None,
        },
        "mnds-news_explicit-single_cls+ret": {
            "end_tokens": [],
            "metric": "exact_match",
            "lang": "en",
        },
        "thucnews_explicit-single_cls+ret": {
            "end_tokens": [],
            "metric": "exact_match",
            "lang": "zh",
        },
        "drcd_explicit-single_ret": {
            "end_tokens": ["，", "\n"],
            "metric": "acc",
            "lang": "en",
        },
        "mnds-news_explicit-multiple_cls+ret": {
            "end_tokens": [],
            "metric": "cls_f1",
            "lang": "en",
            "preprocess": mnds_news_explicit_preprocess,
        },
        "thucnews_explicit-multiple_cls+ret": {
            "end_tokens": [],
            "metric": "cls_f1",
            "lang": "zh",
            "preprocess": thucnews_explicit_preprocess,
        },
        "drcd_semantic-single_qa": {
            "end_tokens": ["\n", "。"],
            "metric": "acc",
            "lang": "zh",
        },
        "mnds-news_semantic-multiple_cls+cnt": {
            "end_tokens": ["."],
            "metric": "exact_match",
            "lang": "en",
            "preprocess": mnds_news_semantic_preprocess,
        },
        "thucnews_semantic-multiple_cls+cnt": {
            "end_tokens": ["。"],
            "metric": "exact_match",
            "lang": "zh",
            "preprocess": thucnews_semantic_preprocess,
        },
        "hotpotqa": {
            "end_tokens": ["\n", "."],
            "metric": "acc",
            "lang": "en",
            "preprocess": None,
        },
    }
    results = []
    for model in os.listdir("outputs"):
        if not os.path.isdir(os.path.join("outputs", model)):
            continue
        for task, params in tasks_param.items():
            p = f"outputs/{model}/{task}.jsonl"
            if os.path.exists(p):
                data = [json.loads(row) for row in open(p)]
                scores: Dict[int, float] = evaluate_by_dict(data, **params)
                for bucket, score in scores.items():
                    results.append(
                        {
                            "Model": model,
                            "Task": task,
                            "Bucket": bucket,
                            "Score": score,
                        }
                    )

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(results).to_csv(
        "results/all_result.csv", header=True, index=False
    )


def main():
    evaluate()


if __name__ == "__main__":
    StrictFire(main)
