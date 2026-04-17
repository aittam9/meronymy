import os
import re
import spacy
import concepcy 

import pandas as pd 
from tqdm.auto import tqdm 

from typing import List, Dict, Tuple

"""Utils to extract partofs relations from a list of words and pre-process the obtained sentences to yeld a DataFrame containing:
  
  nodes: tuples of two nodes of a partof relation, a meronym and a holonym
  statements: normalized statements about the meronym relation i.e. the x is a part of the y
  swapped_statements: a swapped version of the statement (i.e. the y is a part of the x)
  questions: yes or no questions about the relation (i.e. is the x a part of the y?)
  swapped_questions: a swapped version of the questions (i.e. is the y a part of the x?)"""


#load spacy and add concepcy to its pipe (conceptnet wrapper)
def nlp_config(concepcy_edge_weight : float = 3.0)-> spacy.pipeline:
  
  """Configure the spacy pipeline adding concepcy to it with the choosen edge weight parameter"""
  
  nlp = spacy.load("en_core_web_sm")
  nlp.add_pipe(
      "concepcy",
      config={
          "relations_of_interest": ["PartOf"],
          "filter_edge_weight": concepcy_edge_weight,
          "filter_missing_text": True,
          "as_dict": True
      }
  )
  return nlp


#extract partofs relation from a list of words leveraging conceptnet
def extract_partofs(concepts: List[str], nlp) -> List[str]:
    """Extract partofs relations when found from conceptnet starting from the given concept seeds.
    Returns:
    a list of strings extracted from the surfaceText in ConceptNet"""

    partofs = []
    regex = re.compile(r"\[+|\]+")
    #iterate over concepts and extract all the found partof relations
    print("Iterating over concepts to extract partofs...")
    for o in tqdm(concepts):
        try:
          doc = nlp(o)
          if doc[0]._.partof:
              for rel in doc[0]._.partof:
                  sent = re.sub(r"[\.]", "",rel["text"]) #get rid of periods if any
                  #handle the errors in conceptnet (i.e. "*book has pages")
                  if not sent.startswith("*"):
                    partofs.append(re.sub(regex, "", sent).lower())
        except:
           continue
      
    #filter out reflexive lines (x is part of x)
    partofs = [l for l in partofs if l.strip().split()[0] != l.strip().split()[-1]]
    return partofs


#function to normalize the sentences in the fixed form : [Det NP1 rel Det NP2]
def normalize_sents(partofs_list: List[str]) -> Tuple[List[Tuple], List[str]]:
  
  """Normalizes the extracted sentences forcing all of them in the same template
  [Det] [meronym] is a part of [Det] [holonym]
  Returns:
  a list of tuples containing the nodes of each relation (meronym, holonym)
  a list of strings with the normalized statements
  a list of the normalized statements in swapped order"""
  
  normalized = []
  swapped = []
  nodes = []
  split_ons = re.compile(r"is part of|is a part of")
  
  for s in partofs_list:
    #divide the nodes removing the relation
    s = re.split(split_ons, s)

    #split the node if made of multiple words
    np1, np2 = s[0].strip().split(" "), s[1].strip().split(" ")
    if len(np1) > 1 and len(np2) > 1:
      #check if the first word is an indeterminate article
      if (np1[0] in ["a", "an", "the"] and np2[0] in ["a", "an", "the"]):
        np1 = np1[1:]
        np2 = np2[1:]

    #join nps to pass them from lists to strings
    np1, np2 = " ".join(np1), " ".join(np2)
    #concatenate and normalize the np1 and np2 with the normalized relations
    s = f"the {np1} is a part of the {np2}".strip()
    normalized.append(s)
    
    #swap the statements
    s_swap = f"the {np2} is a part of the {np1}".strip()
    swapped.append(s_swap)
    
    #store the nodes to use them later as a reference
    nodes.append((np1,np2))
  return nodes, normalized, swapped


#transforms the sentences into questions
def make_questions(partofs_list: List[str], swapped: List[str]) -> Tuple[List[str], List[str]]:
    """Makes questions and swapped questions out of the normalized statements
    Returns:
    a list of meronymy questions
    a list of swapped meronymy questions"""

    questions = [re.sub(r"\s{2,}", " ", "Is "+" ".join(s) +"?") for s in [re.split(r"\sis\s", s) for s in partofs_list]]
    #make a swapped version
    swapped_questions = [re.sub(r"\s{2,}", " ", "Is "+" ".join(s) +"?") for s in [re.split(r"\sis\s", s) for s in swapped]]
    return questions, swapped_questions


# def swap_nodes(sent):
#   split_ons = re.compile(r"is part of|is a part of")
#   t_split = re.split(split_ons, sent)
#   n1,n2 = t_split[0], t_split[1]
#   return f"{n2} is a part of {n1}".strip()


def make_df(nodes,
            statements,
            swapped_statements,
            questions, 
            swapped_questions,
            output_path= "../processed_data"
            )-> pd.DataFrame:  #add output path
   
    """Take nodes, statements and questions and make a dataframe """

    #convert the nodes tuples in comma separated strings
    nodes = [",".join(n) for n in nodes] 
    
    #make the dataframe
    df = pd.DataFrame(zip(nodes,
                          statements,
                          swapped_statements,
                          questions,
                          swapped_questions), 
                          columns = ["nodes", "statements", "swapped_statements", "questions", "swapped_questions"]).set_index("nodes")
    
    df.to_csv(output_path, sep = "\t")
    df = df.drop_duplicates()
    print(df.head(5))
    return df

# #read mcrae concepts and clean them
# def read_concepts_clean(path: str):
    
#     """Read the concepts and make a first clean"""
    
#     df = pd.read_excel(path)
#     #normalize marked concepts
#     rem = re.compile(r"(_\(.*\))")
#     mr_concepts  = sorted(set([re.sub(rem, "", w) for w in df["Concept"].tolist()]))
#     return mr_concepts

#read mcrae concepts and clean them
def read_concepts_clean(path: str, data:str = "mcrae") -> List[str]:
    
    """Read the concepts, make a first clean and remove multi word concepts.
    Return:
    list of single word cleaned mcrae concepts if data == 'mcrae'
    list of single word cleaned things concepts if data == 'things'
    """
    
    df = pd.read_excel(path)
    #normalize marked concepts
    rem = re.compile(r"(_\(.*\))")
    
    if data == "mcrae":
      concepts  = sorted(set([re.sub(rem, "", w) for w in df["Concept"].tolist()]))
    elif data == "things":
       concepts  = sorted(set([re.sub(rem, "", w) for w in df["Word"].tolist()]))
    else:
       print("invalid data! Data should be 'mcrae' or 'things'")

    concepts = sorted(set([c for c in concepts if not len(c.split()) > 1]))
    return concepts

def make_questions_from_statements(statements: List[str])-> List[str]:
    questions = []
    splitons = r" is | are "
    for s in statements:
        verb = re.search(splitons, s).group().strip()
        
        split = " ".join(re.split(splitons, s))
        question = f"{verb} {split}?"

        questions.append(question)
    return questions


