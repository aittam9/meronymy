import random as r
import pandas as pd
import os
import re
from mcrae_utils import statements_from_nodes, questions_from_nodes
from typing import List, Tuple

CN_PATH = "../../data/conceptnet/conceptnet.tsv"
MCRAE_PATH = "../../data/mcrae/mcrae.tsv"


def make_random_tuples(seed_list):
    new_tuples = []
    while True:
        t = tuple(r.sample(seed_list,2))
        if not t in new_tuples:
            new_tuples.append(t)
            if len(new_tuples) == len(seed_list):
                break 
    return new_tuples

def get_holonyms(path):
    data = pd.read_csv(path, sep = "\t")
    holonyms = [i.split(",")[-1].strip() for i in data["nodes"].tolist() if i]
    #new_tuples = make_random_tuples(holonyms)
    return holonyms


if __name__ == "__main__":
    
    cn_hol = get_holonyms(CN_PATH)
    mcr_hol = get_holonyms(MCRAE_PATH)
    hol = sorted(set(cn_hol + mcr_hol))
    print(len(hol))
    new_tuples = make_random_tuples(hol)
    statements, swapped_statements = statements_from_nodes(new_tuples)
    questions, swapped_questions = questions_from_nodes(new_tuples)

    df = pd.DataFrame({
        "nodes": list(map(lambda x: ",".join(x), new_tuples)),
        "statements": statements,
        "swapped_statements": swapped_statements,
        "questions": questions,
        "swapped_questions": swapped_questions
    })

    df.to_csv("../../data/fake/fake.tsv", sep = "\t", index = False)