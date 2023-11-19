import os
import json
import torch
import openai
import logging
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from openai.error import (
    RateLimitError,
    APIError,
    APIConnectionError,
    Timeout,
    AuthenticationError,
)
from strictfire import StrictFire
from tqdm import tqdm
from utils.misc import get_logger
from typing import Dict, Any, List
from time import sleep
from datasets import load_dataset


SEED = 111
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def openai_generate(
    model_name: str,
    prompt: str,
    api_key: str,
    print_prompt: bool = False,
    max_tokens: int = 128,
    temperature: float = 1,
    top_p: float = 1,
    **kwargs,
):
    prompt = prompt.replace("\\n", "\n")
    if print_prompt:
        print("Prompt:", prompt)
        print("=" * 50)
    if model_name == "gpt-3.5-turbo":
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            api_key=api_key,
            **kwargs,
        )
    else:
        completion = openai.Completion.create(
            model=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            api_key=api_key,
            **kwargs,
        )
    if model_name == "gpt-3.5-turbo":
        resp = completion["choices"][0]["message"]["content"]
    else:
        resp = completion["choices"][0]["text"]
    return resp


def llama_generate(
    prompt,
    model,
    tokenizer,
    debug: bool = False,
    **kwargs,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    if not debug:
        input_ids = input_ids.to(DEVICE)
    if debug:
        output = "some dummy text."
        return output, input_ids.shape[1]
    else:
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs,
            )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s[input_ids.shape[1] :])
    output = output.replace(tokenizer.eos_token, "")
    return output, input_ids.shape[1]


def main(
    model_path: str,
    task: str,
    output_key: str = "gen_resp",
    api_key: str = None,
    load_8bit: bool = False,
    temperature: float = 1.0,
    top_p: float = 1,
    top_k: int = 50,
    do_sample: bool = False,
    load_model_args: Dict[str, Any] = {},
    resume: bool = False,
):
    model_name = model_path.split("/")[-1]
    logger = get_logger(
        name=__name__,
        console_level="info",
        file_level="debug",
        log_path=os.path.join("log", model_name + ".log"),
        maxBytes=10000000,
    )
    data = load_dataset("wckwan/M4LE", task, split="test")
    data = [row for row in data]
    out_filename = os.path.join("outputs", model_name, task + ".jsonl")
    out_data = []
    if os.path.exists(out_filename):
        out_data = [json.loads(line) for line in open(out_filename)]
    if (
        resume
        and out_data
        and sum([output_key in row for row in out_data]) == len(data)
    ):
        print(f"{out_filename} has finished.")
        return
    use_openai = False
    if model_path in ["gpt-3.5-turbo-16k", "gpt-3.5-turbo"]:
        use_openai = True
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **load_model_args,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        try:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        except AttributeError as e:
            logger.info(e)
            logger.info(
                "Can't set the tokenizer.pad_token_id but it's probably"
                " ok if the model is chatglm."
            )
    logger.info(f"loaded model `{model_path}`")

    print_first_prompt = False
    if resume and out_data:
        matched = 0
        for row in out_data:
            for ori_row in data:
                if output_key in row:
                    ori_row[output_key] = row[output_key]
                    matched += 1
                    break
        logger.info(f"Resumed {matched} instances from {out_filename}.")

    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    for i, row in enumerate(tqdm(data)):
        if resume and output_key in row:
            continue
        prompt = row["instruction"] + "\n" + row["input"]
        error_occured = False
        if use_openai:
            prompt_len = len(tokenizer(prompt)["input_ids"])
            for i in range(20):
                try:
                    resp = openai_generate(
                        model_name=model_path,
                        prompt=prompt,
                        api_key=api_key,
                        temperature=temperature,
                        top_p=top_p,
                    )
                except (
                    APIConnectionError,
                    RateLimitError,
                    APIError,
                    Timeout,
                    AuthenticationError,
                ):
                    logger.exception("Retry after 30 seconds.")
                    sleep(30)
                except Exception:
                    error_occured = True
                    logger.exception(f"Error occured at {i}.")
                    break
                else:
                    break
            else:
                error_occured = True
                logger.exception(f"Error occured at {i}.")
        else:
            try:
                generation_config = GenerationConfig(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_new_tokens=row["length_bucket"],
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=do_sample,
                )
                resp, prompt_len = llama_generate(
                    prompt=prompt,
                    model=model,
                    tokenizer=tokenizer,
                    generation_config=generation_config,
                )
            except Exception:
                error_occured = True
                logger.exception(f"Error occured at {i}.")
        if not print_first_prompt and not error_occured:
            tqdm.write(prompt)
            tqdm.write(resp)
            print_first_prompt = True
        row["error"] = error_occured
        if not error_occured:
            row[output_key] = resp
            row["prompt_len"] = prompt_len
        if i % 10 == 0:
            with open(out_filename, "w", encoding="utf-8") as f:
                f.write(
                    "\n".join([json.dumps(row, ensure_ascii=False) for row in data])
                )
            logger.debug(
                f"Ran {i+1}/{len(data)}."
                f" prompt_len={prompt_len if not error_occured else 'ERROR'}."
                f" Saved to {out_filename}"
            )
    with open(out_filename, "w", encoding="utf-8") as f:
        f.write("\n".join([json.dumps(row, ensure_ascii=False) for row in data]))
    logger.info(f"Finished running. Output saved in {out_filename}.")


if __name__ == "__main__":
    StrictFire(main)
