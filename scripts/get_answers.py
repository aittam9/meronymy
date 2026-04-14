from typing import Tuple, List, Literal 
import argparse 
from collections import Counter
import os


from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm 
import torch
import pandas as pd
import outlines

from transformers.utils import logging
logging.set_verbosity_error() 

from utils.models import  MODELS 


torch.manual_seed(42)
torch.cuda.manual_seed(42)


CACHE_DIR = "/extra/mattia.proietti/hf_models"
DATA_PATH = "../data"
BASE_OUTDIR = "../results"
DEVICE = "cuda:0"


def unpack_data(df, task:Literal["questions", "statements"]) -> Tuple[List[str], List[str], List[str], Literal["yes", "no"] | Literal["true", "false"]]:
    """
    Parse the input data and return a dataframe with the following columns:
    - nodes: the nodes of the graph
    - questions: the questions to feed the model with
    - statements: the statements to feed the model with
    - swapped_questions: the swapped version of the questions
    - swapped_statements: the swapped version of the statements
    - output_type: the type of output expected from the model (yes/no for questions, true/false for statements)
    
    """
    
    #store the nodes
    nodes = df["nodes"].tolist()
      # format the prompt appropriately and extract te right input following the chosen task
    if task == "questions":
        sents = df["questions"].tolist()
        swapped = df["swapped_questions"].tolist()
        output_type  = Literal["yes", "no"]

    
    # if true or false task is chosen
    elif task == "statements":
        sents = df["statements"].tolist()
        swapped = df["swapped_statements"].tolist()
        output_type  = Literal["true", "false"]


    else:
        print("Invalid task. Choose between 'questions' and 'statements'.")
        return
    
    return nodes, sents, swapped, output_type

def load_data(data_arg) -> Tuple[pd.DataFrame, str]:
    # load the data
    if "mcrae" in data_arg:
        if "lemma" in data_arg:
            data_path = f"{DATA_PATH}/mcrae/mcrae_lemma.tsv"
            data = "mcrae_lemma"
            df = pd.read_csv(data_path, sep="\t")
        else:
            data_path = f"{DATA_PATH}/mcrae/mcrae.tsv"
            data = "mcrae"
            df = pd.read_csv(data_path, sep="\t")
    elif "conceptnet" in data_arg:
        data_path = f"{DATA_PATH}/conceptnet/conceptnet.tsv"
        data = "conceptnet"
        df = pd.read_csv(data_path, sep="\t")
    
    print(f"\n{data} data loaded from {data_path}\n")
    return df, data 
    
def prompt_format(sent, task) -> str:
    """
    Format the prompt appropriately for the model. This is a placeholder function and can be modified to fit the specific requirements of the model being used.
    """
    if task == "questions":
        return f"Answer yes or no to the following question: {sent} \nAnswer:"
    else:
        return f"Judge if the following statement is true or false: {sent} \nAnswer:"
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type = str, choices = ["mcrae", "mcrae_lemma", "conceptnet"], help = "The dataset to shuffle. Refer to the last folder it is in.")
    parser.add_argument("--model", "-m", choices=list(MODELS.keys()), help = "The model to test")
    parser.add_argument("--task", "-t", type = str,  choices = ["questions", "statements"], help = "The type of input to feed the model with. Choose between 'questions' and 'statements'.")
    
    args = parser.parse_args()

    
    # load data
    df, data = load_data(args.data)
    
    print(df.head(), data, args.model, args.task)
    #instantiate the output path for the results
    out_path = f"{BASE_OUTDIR}/{args.task}/{args.model}"
    os.makedirs(out_path, exist_ok=True)

    print(f"\nStarting feeding the model with {args.task} inputs\n")
    #unpack data
    nodes, sents, swapped, output_type = unpack_data(df, args.task)
    
   
    # load model 
    model_id = MODELS[args.model]
    print(f"Loading {model_id}...")
    model_kwargs = {"cache_dir": CACHE_DIR}
    if DEVICE != "cpu":
        model_kwargs["device_map"] = DEVICE

    model = outlines.from_transformers(
                AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs),
                AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
                )
    
    print(f"\nExample prompt: {prompt_format(sents[0], args.task)}\n")
    #feeding the model with the chosen inputs
    answers = [model(prompt_format(s, args.task), output_type, max_new_tokens=10, temperature=0., do_sample=False)   for s in tqdm(sents) if s] 
    # testing swapped 
    print(f"\nStarting feeding the model with {args.task} swapped\n")
    print(f"Example prompt swapped: {prompt_format(swapped[0], args.task)}\n")
    answer_swapped = [model(prompt_format(s, args.task), output_type, max_new_tokens=10, temperature=0., do_sample=False) for s in tqdm(swapped) if s]

    # save the results in a dataframe
    results = pd.DataFrame(zip(nodes,  answers, answer_swapped),
                            columns = ["nodes", "answers", "answers_swapped"]).set_index("nodes")

    
    # save the results in a dataframe
    results.to_csv(f"{out_path}/{data}_{args.task}_results.tsv", sep = "\t")
        

    print(f"Results DataFrames saved at {out_path}")
    print(f"\nBrief summary for {args.task}: {Counter(answers)}")
    print(f"\nBrief summary for {args.task}_swapped: {Counter(answer_swapped)}")
    
    # accuracy meronymy knowledge criterion: the model should answer "yes" to the original question and "no" to the swapped version (or "true"/"false" for statements)
    targets  = "yes/no" if args.task == "questions" else "true/false"
    
    accuracy = len([(i,e) for i, e in zip(answers, answer_swapped) if i == targets.split("/")[0] and e == targets.split("/")[1]]) / len(answers) * 100
    print(f"\nAccuracy of the model on the Meronymy Knowledge Criterion: {accuracy}")
            
            
    