import json
from typing import List, Dict, Tuple
import spacy
from tqdm.auto import tqdm

nlp = spacy.load("en_core_web_lg")

def clean_dict(dict2clean: Dict, words2remove: List) -> Tuple[Dict[str,str], List[Tuple], Dict[str,str]]:
    """
        Function to further clean the dictionary on the base of a list of words to be removed.
        Args:
            dict2clean: the dictionary to be cleaned
            words2remove: the words to be removed

        Return:
            a new cleaned dictionary (without the target words)
            a list of tuples of cleaned nodes (holonym, meronym)
            a dictionary with the key value pairs removed from the original
    """
    #extract the "strange cases" from the source dictionary and create a new one with them
    strange_dict = {}
    for s in words2remove:
        if s:
            if dict2clean[s]:
                strange_dict[s] = dict2clean[s]
            else:
                strange_dict[s] = s
    
    #clean the dict and make a list of tuples out of it
    list_of_tuple = []
    for k in dict2clean:
        if not k in words2remove:
            split = dict2clean[k].split(",")
            if split:
                for v in split:
                    if v and not k == v.lower():
                        tupla = k,v.lower()
                        if not tupla in list_of_tuple:
                            list_of_tuple.append(tupla)
    
    # make antother final dict with the cleaned tuple         
    ultra_clean = {}
    for t in list_of_tuple:
        k = t[0]
        if not k in ultra_clean:
            ultra_clean[k] = [] 
        if not t[1] in ultra_clean[k]:
            ultra_clean[k].append(t[1]) 
    
    #refine the dict joining the values lists
    ultra_clean = {s: ",".join(ultra_clean[s]) for s in ultra_clean}

    return ultra_clean, list_of_tuple, strange_dict


# write the dict into a file
def write_dict(dict2write: Dict, file_name : str = "") -> None:
    with open(file_name, "w") as outfile:
        json.dump(dict2write, outfile, indent = 1)

# write the nodes into a file
def write_nodes(nodes2write: List, file_name: str = "", invert = False) -> None:
    with open(file_name, "w") as outfile:
        for n in nodes2write:
            hol = n[0]
            mer = n[1]
            if invert:
                node = f"{mer},{hol}"
            else:
                node = f"{hol},{mer}"   
            outfile.write(node+ "\n")


def get_data_from_gens(list_of_tuple, get_info = False):
    """
        Build a grammatical number-aware list of statements and questions based on the generated
        meronyms.
        tuples must be (meronym, holonym)
    """
    infos = []
    statements = []
    questions = []
    
    for t in tqdm(list_of_tuple):
        mer = t[0]
        hol = t[1]
        tag = nlp(mer)
        #check if the node is made of more than one word
        if len(mer.split()) > 1:
            for noun in tag.noun_chunks:
                #if so get the grammatical number of the root of the phrase
                try:
                    number = noun.root.morph.get("Number")[0]
                    info = mer, noun.root.pos_, number
                except:
                    pass

        #if single word
        else:
            try: 
                number = tag[0].morph.get("Number")[0]
            except:
                #assign singular number to all the failed cases
                number = "Sing"
            #store the numbers and the pos tags
            info = mer, tag[0].pos_, number
        
        #force the verb tags ending in s to be plural noun
        if info[1] == "VERB" and info[0].endswith("s") and not info[0] in ["harness", "mattress", "cross", "pancreas"]:
            info = info[0], "NOUN", "Plur"
        elif info[0] == "pivot":
            info = info[0], "NOUN", "Sing"

        infos.append(info)

        #return gatehered infos if required
        
        
        if info[-1] == "Sing":
            sent = f"The {mer} is a part of the {hol}"
            quest = f"Is the {mer} a part of the {hol}?"
        else:
            sent = f"The {mer} are a part of the {hol}"
            quest = f"Are the {mer} a part of the {hol}?"
            
        statements.append(sent)
        questions.append(quest)
    if get_info:
        return infos
    else:
    
        return statements, questions