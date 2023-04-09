from sys import argv
from os import listdir
import pandas as pd
import json

bleu_results = []
chrf2_results = []
devtest_dir = argv[1]
model_name = argv[2]
base_path = "results"

for lang_pair in sorted(listdir(devtest_dir)):
    json_fname = f"{lang_pair}_itv2_scores.json"
    try:
        with open(f"{devtest_dir}/{lang_pair}/{json_fname}", 'r') as f:
            bleu_dict, chrf2_dict = json.load(f)
        bleu_results.append([lang_pair, bleu_dict["score"]])
        chrf2_results.append([lang_pair, chrf2_dict["score"]])
    except FileNotFoundError:
        pass

df_bleu = pd.DataFrame(bleu_results, columns=["lang_pair", "score"])
averages = df_bleu.iloc[:, 1:].mean(axis=0)
df_bleu = df_bleu.append(averages, ignore_index=True)
df_bleu.to_csv(f"{base_path}/{model_name}_bleu_itv2.csv", index=False)

df_chrf2 = pd.DataFrame(chrf2_results, columns=["lang_pair", "score"])
averages = df_chrf2.iloc[:, 1:].mean(axis=0)
df_chrf2 = df_chrf2.append(averages, ignore_index=True)
df_chrf2.to_csv(f"{base_path}/{model_name}_chrf2_itv2.csv", index=False)