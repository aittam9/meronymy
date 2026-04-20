import pandas as pd 
from typing import Tuple, List, Dict, Literal

CACHE_DIR = "/extra/mattia.proietti/hf_models"
DATA_PATH = "../data"
BASE_OUTDIR = "../results"
# BASE_OUTDIR = "../test"
DEVICE = "cuda:0"


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
    
    elif "fake" in data_arg:
        data_path = f"{DATA_PATH}/fake/fake.tsv"
        data = "fake"
        df = pd.read_csv(data_path, sep="\t")
    
    print(f"\n{data} data loaded from {data_path}\n")
    return df, data 



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