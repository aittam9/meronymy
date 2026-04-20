#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
	echo "Usage: $0 <data> <task>"
	echo "  data: mcrae | mcrae_lemma | conceptnet | fake | all"
	echo "  task: questions | statements"
	exit 1
fi

DATA="$1"
TASK="$2"


case "$DATA" in
	mcrae|mcrae_lemma|conceptnet|fake|all) ;;
	*)
		echo "Invalid data: $DATA"
		echo "Expected one of: mcrae, mcrae_lemma, conceptnet, fake, all"
		exit 1
		;;
esac

case "$TASK" in
	questions|statements) ;;
	*)
		echo "Invalid task: $TASK"
		echo "Expected one of: questions, statements"
		exit 1
		;;
esac



SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
	echo "Python executable not found: $PYTHON_BIN"
	exit 1
fi

mapfile -t MODELS < <(
	"$PYTHON_BIN" - <<'PY'
from utils.models import MODELS
for model_name in MODELS:
		print(model_name)
PY
)

if [[ ${#MODELS[@]} -eq 0 ]]; then
	echo "No models found in utils.models.MODELS"
	exit 1
fi

if [[ "$DATA" == "all" ]]; then
	DATASETS=("mcrae" "mcrae_lemma" "conceptnet" "fake")
else
	DATASETS=("$DATA")
fi

echo "Running task=$TASK on ${#MODELS[@]} models and ${#DATASETS[@]} dataset(s)"

for MODEL in "${MODELS[@]}"; do
	for DATA_ITEM in "${DATASETS[@]}"; do
		echo ""
		echo "=== Running model: $MODEL ==="
		echo "Data: $DATA_ITEM, Task: $TASK"
		"$PYTHON_BIN" get_probs.py --data "$DATA_ITEM" --task "$TASK" --model "$MODEL"
	done
done

echo ""
echo "All runs completed."
