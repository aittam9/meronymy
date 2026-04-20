"""Score pairs of sentences with a Hugging Face causal language model.

The script reads a TSV file with two text columns, computes the logprob of each
sentence under a causal LM, and reports whether the first sentence is more
likely than the second.
"""

import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm 

from utils.models import MODELS
from utils.data_load import load_data, unpack_data


CACHE_DIR = "/extra/mattia.proietti/hf_models"
DATA_PATH = "../data"
BASE_OUTDIR = "../results/probabilities"
TEST_OUTDIR = "../test/probabilities"
DEVICE = "cuda:0"



@dataclass
class SentenceScore:
	text: str
	logprob: float
	avg_logprob: float


def load_model(model_name: str, device:str =  DEVICE, cache_dir: str | None = None):
	tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
	model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"

	model.to(device)
	model.eval()
	return model, tokenizer


def sentence_logprob(model, tokenizer, sentence: str, device:str | None = None) -> SentenceScore:
	input_ids = tokenizer.encode(sentence, add_special_tokens=False)
	if tokenizer.bos_token_id is not None:
		input_ids = [tokenizer.bos_token_id] + input_ids

	inputs = {
		"input_ids": torch.tensor([input_ids], device=device),
		"attention_mask": torch.ones((1, len(input_ids)), device=device, dtype=torch.long),
	}

	with torch.no_grad():
		logits = model(**inputs).logits

	shift_logits = logits[:, :-1, :]
	shift_labels = inputs["input_ids"][:, 1:]

	token_logprobs = torch.log_softmax(shift_logits, dim=-1).gather(
		2, shift_labels.unsqueeze(-1)
	).squeeze(-1)

	total_logprob = token_logprobs.sum().item()
	avg_logprob = token_logprobs.mean().item() if token_logprobs.numel() else float("nan")
	return SentenceScore(sentence, total_logprob, avg_logprob)


def score_pair(model, tokenizer, first: str, second: str, device: str | None = None) -> Tuple[SentenceScore, SentenceScore, bool]:
	first_score = sentence_logprob(model, tokenizer, first, device)
	second_score = sentence_logprob(model, tokenizer, second, device)
	return first_score, second_score, first_score.logprob > second_score.logprob


def main() -> None:
	parser = argparse.ArgumentParser(description="Score sentence pairs with a causal LM.")
	parser.add_argument("--data", required=True, help="TSV file with paired sentences")
	parser.add_argument("--task", required = True, choices = ["questions", "statements", "fake"], help="The type of input to feed the model with. Choose between 'questions', 'statements', or 'fake'.")
	parser.add_argument("--model", required = True, choices=MODELS.keys(), help="Hugging Face causal LM name or path")
	parser.add_argument("--test", required=False, action = "store_true", help="Optional path to save the scored TSV")
	args = parser.parse_args()

	# load and unpack data
	df, data = load_data(args.data)
	print(df.head())
	nodes, sents, swapped,  _ = unpack_data(df, args.task)

	# output path
	if not args.test:
		out_path = f"{BASE_OUTDIR}/{args.task}/{args.model}"
	else:
		out_path = f"{TEST_OUTDIR}/{args.task}/{args.model}"
	
	os.makedirs(out_path, exist_ok=True)
	print(f"Results will be saved at {out_path}")

	#load model
	model_id = MODELS[args.model]
	print(f"Loading {model_id}...")
	model, tokenizer = load_model(model_id, device=DEVICE, cache_dir=CACHE_DIR)

	print(f"Example sentence: {sents[0]}\nExample swapped sentence: {swapped[0]}\n")
	rows = []
	for n, s, s_swap in tqdm(zip(nodes, sents, swapped), total=len(nodes)):
		first_text = s
		second_text = s_swap
		first_score, second_score, first_is_greater = score_pair(
			model, tokenizer, first_text, second_text, device = DEVICE
		)
		rows.append(
			{
				"nodes" : n,
				"sents_logprob": first_score.logprob,
				"swapped_logprob": second_score.logprob,
				"sents_avg_logprob": first_score.avg_logprob,
				"swapped_avg_logprob": second_score.avg_logprob,
				"sents_logprob_greater": first_is_greater,
			}
		)

	# build and save results dataframe
	result = pd.DataFrame(rows)
	
	# print(result)
	print()
	print(f"First sentence has higher logprob in {result['sents_logprob_greater'].mean() * 100:.2f}% of pairs")

	# result.to_csv(out_path, sep="\t", index=False)
	print(f"Saved results to {out_path}")

	# save the results in a dataframe
	result.to_csv(f"{out_path}/{data}_{args.task}_results.tsv", sep = "\t")


if __name__ == "__main__":
	main()
