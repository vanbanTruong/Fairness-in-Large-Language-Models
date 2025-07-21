import numpy as np
import datetime
import os
import pandas as pd
from encoder_only.intrinsic_bias.similarity_based.ceat.ceat import *
from encoder_only.intrinsic_bias.similarity_based.ceat.generate_ebd_bert import *

def run_experiment():
    print("------------Encoder-only LMs: Intrinsic bias - Similarity_based bias - CEAT------------")
    if not os.path.exists("data/ceat/bert_weat.pickle"):
        print("------------Generating embeddings------------")

    f = open('data/ceat/data.json')
    data = json.load(f)

    ceat_groups = [
        [data["flowers"],data["insects"],data["pleasant"],data["unpleasant"]], 
        [data["instruments"], data["weapons"], data["pleasant"], data["unpleasant"]], 
        [data["european_3"],data["african_3"],data["pleasant_3"],data["unpleasant_3"]], 
        [data["european_4"],data["african_4"],data["pleasant_3"],data["unpleasant_3"]], 
        [data["european_4"],data["african_4"],data["pleasant_5"],data["unpleasant_5"]],
        [data["male"],data["female"],data["career"],data["family"]], 
        [data["math"],data["arts"],data["male_term"],data["female_term"]],
        [data["science"],data["arts_8"],data["male_term_8"],data["female_term_8"]],
        [data["mental_disease"],data["physical_disease"],data["temporary"],data["permanent"]],
        [data["young_name"],data["old_name"],data["pleasant_5"],data["unpleasant_5"]],
        [data["african_female"],data["european_male"],data["af_bias"],data["em_bias_foraf"]], 
        [data["african_female"],data["european_male"],data["af_unique_bias"],data["em_unique_bias_foraf"]],
        [data["mexican_female"],data["european_male"],data["lf_bias"],data["em_bias_forlf"]],
        [data["mexican_female"],data["european_male"],data["lf_unique_bias"],data["em_unique_bias_foraf"]]
    ]

    groups =[
        ["flowers", "insects", "pleasant", "unpleasant"],
        ["instruments", "weapons", "pleasant", "unpleasant"],
        ["european_3", "african_3", "pleasant_3", "unpleasant_3"],
        ["european_4", "african_4", "pleasant_3", "unpleasant_3"],
        ["european_4", "african_4", "pleasant_5", "unpleasant_5"],
        ["male", "female", "career", "family"],
        ["math", "arts", "male_term", "female_term"],
        ["science", "arts_8", "male_term_8", "female_term_8"],
        ["mental_disease", "physical_disease", "temporary", "permanent"],
        ["young_name", "old_name", "pleasant_5", "unpleasant_5"],
        ["african_female", "european_male", "af_bias", "em_bias_foraf"],
        ["african_female", "european_male", "af_unique_bias", "em_unique_bias_foraf"],
        ["mexican_female", "european_male", "lf_bias", "em_bias_forlf"],
        ["mexican_female", "european_male", "lf_unique_bias", "em_unique_bias_foraf"]
    ]

    e_lst = []
    p_lst = []
    for e in range(1,15):
        e_lst.append([])
        p_lst.append([])

        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

        pes,  p_value = ceat_meta(ceat_groups = ceat_groups,test=e,N=1000)
        print(f"PES is {pes}.")
        print(f"P-value is {p_value}")
        e_lst[e-1].append(pes)
        e_lst[e-1].append(p_value)
        print(" ")
        
    e_ary = np.array(e_lst)
    res = []
    for i, group in enumerate(groups):
        res.append({
            "Target 1": group[0],
            "Target 2": group[1],
            "Attribute 1": group[2],
            "Attribute 2": group[3],
            "Effect size": e_lst[i][0],
            "P-value": e_lst[i][1]
        })

    df = pd.DataFrame(res)
    df.to_csv("encoder_only/intrinsic_bias/similarity_based/ceat/result.csv", index=False)
