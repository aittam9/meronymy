import os 
import json 
from collections import Counter
from typing import Dict, List, Tuple, Literal 

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


data_map = {'mcrae_lemma_questions_results': 'McRae-lemma', 'mcrae_questions_results': 'McRae', 'conceptnet_questions_results': 'ConceptNet',
            "mcrae_lemma_statements_results": "McRae-lemma", "mcrae_statements_results": "McRae", "conceptnet_statements_results": "ConceptNet",
            "fake_questions_results": "Fake", "fake_statements_results": "Fake"}





def load_results(path, experiment = ["probabilities", "probabilities_templates", "prompting"]):
    all_results_questions = []
    all_results_statements = []
    if experiment in ["probabilities", "probabilities_templates"]:
        path = os.path.join(path, experiment)
    for subtask in ["questions", "statements"]:
        subtask_path = os.path.join(path, subtask)
        for model in sorted(os.listdir(subtask_path)):
            model_path = os.path.join(subtask_path, model)
            for file in sorted(os.listdir(model_path)):
                if file.endswith(".tsv"):
                    data = pd.read_csv(os.path.join(model_path, file), sep="\t")
                    data["model"] = model
                    data["task"] = subtask
                    data["data"] = data_map[file.split(".")[0]]
    
                    if experiment == "prompting":
                        if subtask == "questions":
                            data["correct"] = (data["answers"] == "yes") & (data["answers_swapped"] == "no")
                            all_results_questions.append(data)
                        else:
                            data["correct"] = (data["answers"] == True) & (data["answers_swapped"] == False)
                            all_results_statements.append(data)
                    
                    elif experiment in ["probabilities", "probabilities_templates"]:
                        # data["correct"] = data["sents_logprob_greater"]
                        if subtask == "questions":
                            all_results_questions.append(data)
                        else:
                            all_results_statements.append(data)
                    else:
                        raise ValueError("Experiment not recognized. Choose between 'prompting', 'probabilities' and 'probabilities_templates'.")
    return pd.concat(all_results_questions, ignore_index=True), pd.concat(all_results_statements, ignore_index=True)



def plot_prob_experiment(all_results_q_prob, all_results_s_prob, template = False):
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sns.barplot(x="model", y="sents_logprob_greater", hue="data", data=all_results_q_prob, ax=axes[0], legend=False).set(ylabel = "Probability Accuracy")
    axes[0].set_title("Questions")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_ylim(0, 1)
    axes[0].axhline(y=0.5, color="r", linestyle="--")

    sns.barplot(x="model", y="sents_logprob_greater", hue="data", data=all_results_s_prob, ax=axes[1])
    axes[1].set_title("Statements")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_ylim(0, 1)
    axes[1].axhline(y=0.5, color="r", linestyle="--")
    

    axes[1].legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        ncol=2,
        frameon=True,
        title="Data",
        fontsize=11,
        title_fontsize=11,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=0.8,
        borderpad=0.3,
        labelspacing=0.3
    )
  
    title = "Accuracy of models on questions and statements probabilities"
    if template:
        title = title + " (with chat template)"
    plt.suptitle(title)
    plt.tight_layout()



def plot_model_prob_diff(df, model, data, ax=None):
    prob1 = df[(df["model"] == model) & (df["data"] == data)]["sents_logprob"]
    prob2 = df[(df["model"] == model) & (df["data"] == data)]["swapped_logprob"]
    differences_distribution = prob1 - prob2
    mean_diff = differences_distribution.mean()
    
    sns.histplot(differences_distribution, bins=30, kde=True, ax=ax)
    target_ax = ax if ax is not None else plt.gca()
    target_ax.axvline(mean_diff, color="black", linestyle="--", linewidth=1.5, label=f"Mean: {mean_diff:.3f}")
    target_ax.legend()
    
    if ax:
        ax.set_title(f"{model}")
    else:
        plt.title(f"Distribution of the differences between log probabilities of original and swapped sentences for {model} on {data}")



def plot_distribution_differences(df, data):
    fig, axes = plt.subplots(2, 3, figsize=(20, 6), sharey=True)
    axes_flat = axes.flatten()
    
    for idx, model in enumerate(sorted(df["model"].unique().tolist())):
        if idx < len(axes_flat):
            plot_model_prob_diff(df, model, data, ax=axes_flat[idx])
    
    plt.suptitle(f"Distribution of probability differences for {data}")
    plt.tight_layout()
