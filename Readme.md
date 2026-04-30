## The *quasi*-semantic space of LLMs: a case study on the *part-whole* relation

## Script Command Instructions

Run all commands from the `scripts/` directory.

### 1) `get_answers.py`

Generates model answers for original and swapped prompts, then saves results in:

- `../results/<task>/<model>/<dataset>_<task>_results.tsv`

#### Command

```bash
python get_answers.py --data <dataset> --model <model> --task <task>
```

#### Arguments

- `--data` (`-d`): `mcrae` | `mcrae_lemma` | `conceptnet`
- `--model` (`-m`):
	- `llama-3.2-1b-it`
	- `llama-3.2-3b-it`
	- `llama-3.1-8b-it`
	- `gemma-3-1b-it`
	- `gemma-3-4b-it`
	- `gemma-7b-it`
- `--task` (`-t`): `questions` | `statements`

#### Examples

```bash
python get_answers.py -d mcrae -m llama-3.2-1b-it -t questions
python get_answers.py -d conceptnet -m gemma-3-4b-it -t statements
```

### 2) `get_probs.py`

Computes and compares sentence log-probabilities for original vs swapped pairs and saves results in:

- `../results/probabilities/<task>/<model>/<dataset>_<task>_results.tsv`
- if `--test` is used: `../test/probabilities/<task>/<model>/<dataset>_<task>_results.tsv`

#### Command

```bash
python get_probs.py --data <dataset> --model <model> --task <task> [--test]
```

#### Arguments

    - `--data`: `mcrae` | `mcrae_lemma` | `conceptnet` |  `fake_meronyms`
- `--model`: same model IDs listed above
- `--task`: `questions` | `statements`
- `--test`: optional flag to write outputs under `../test/probabilities/...`

#### Examples

```bash
python get_probs.py --data mcrae_lemma --model llama-3.2-3b-it --task questions
python get_probs.py --data conceptnet --model gemma-7b-it --task statements --test
```




**Note** Part of the code has been written with the aid of artificial intelligence. In particular, AI has been employed in boilerplate code, bash scripting and regression analysis.


