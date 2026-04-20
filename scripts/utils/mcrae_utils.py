import re 
import spacy 
import pandas as pd
from typing import List, Tuple
from tqdm.auto import tqdm

nlp = spacy.load("en_core_web_lg")

#read the dataset and take the concept and the feature on conditions
def read_mcrae_partofs() -> List[Tuple[str]]:
    
    """Read the mcrae features and extract the relevant concept and associated feature
       with a condition on the third column to filter only external and internal components.
       Return a list of tuples (concept, has_feature) """

    mr_df = pd.read_excel("../source_data/mcrae_feats.xlsx")
    parts = []
    #extract feats on conditions of WB_Label col
    for r in mr_df.itertuples():
        if r.WB_Label in ["internal_component", "external_component"]:
            parts.append((r.Concept, r.Feature))
    
    return parts


#clean the extracted oconcept and features
def clean_parts(parts_list: List[Tuple[str]]) -> Tuple[List[Tuple], List[str]]:
    """Clean the extracted McRae features. Removes underscores, digit and equalize homonyms (i.e. bat_(animal)--> bat)
    
    Returns:
    a list of tuples with cleaned relations (concept, has feature);
    a list of strings representing the full 'sentence'."""

    #clean the meronyms removing underscores and digits
    parts_list = [(p[0], re.sub(r"\s{2,}", " " ,re.sub( r"[_\d]", " " ,p[1]))) for p in parts_list]
    mr_parts_clean = []
    splitons = re.compile(r"(has a |has)")
    
    #remove all the multiword meronyms (i.e. adj + noun)
    for i in range(len(parts_list)):
        split = re.split(splitons, parts_list[i][1])
        target = split[-1].strip()
        if not len(target.split()) > 1:
            mr_parts_clean.append(parts_list[i])
    
    parts_clean = sorted(set(mr_parts_clean))
    
    #get sents 
    mr_concepts_sents = [re.sub(r"(_\(.*\))", "", " ".join(s)) for s in mr_parts_clean]
    
    return parts_clean, mr_concepts_sents


#get holnyms and meronyms nodes both original and lemmatized
def get_nodes(parts_sents: List[Tuple[str]]) -> Tuple[List[Tuple[str]], List[Tuple[str]]]:

    """Extract the nodes of the part of relations (holonym, meronym) in both original and lemmatized form.
    
    Returns:
    a list of tuples with original nodes
    a list of tuples with lemmatized nodes """

    nodes = []
    nodes_lemma = []
    
    for s in tqdm(parts_sents):
        #separate holonyms and meronyms
        holonym, meronym = s.split()[0].strip(), s.split()[-1].strip()
        nodes.append((meronym, holonym))
        
        #lemmatize plural meronyms/holonyms
        meronym = nlp(meronym)[0].lemma_
        holonym = nlp(holonym)[0].lemma_
        nodes_lemma.append((meronym, holonym))

    #assure the nodes are unique and sort them alfabetically
    nodes_lemma= sorted(set(nodes_lemma))
    nodes = sorted(set(nodes)) 
    
    return nodes, nodes_lemma 

def final_clean(nodes: List[str]):
    to_remove =  ["protein", "water", "fat", "television", "bar", "number"]
    nodes = [n for n in nodes if not n[0] in to_remove]
    return nodes

def statements_from_nodes(nodes: List[Tuple]) -> Tuple[List[str], List[str]]:
    """ 
    make statements from a list of nodes meronym, holonym
    """
    statements = []
    swapped_statements = []

    for n in nodes:
        meronym = n[0]
        holonym = n[1]
        phrase = f"The {meronym} is part of the {holonym}"
        phrase_swapped = f"The {holonym} is part of the {meronym}"

        statements.append(phrase)
        swapped_statements.append(phrase_swapped)
    return statements, swapped_statements

def questions_from_nodes(nodes: List[Tuple]) -> Tuple[List[str], List[str]]:
    questions = []
    swapped_questions = []

    for n in nodes:
        meronym = n[0]
        holonym = n[1]
        q = f"Is the {meronym} part of the {holonym}?"
        q_swapped = f"Is the {holonym} part of the {meronym}?"

        questions.append(q)
        swapped_questions.append(q_swapped)
    return questions, swapped_questions
