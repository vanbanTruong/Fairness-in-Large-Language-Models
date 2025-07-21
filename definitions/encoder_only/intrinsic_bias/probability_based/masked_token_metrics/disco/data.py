import re
import pandas as pd


def load_names():
    "Load gendered names from Lauscher et al. (2021) used by DisCo."

    df = pd.read_csv("data/disco/name_pairs.txt", sep="\t", header=None)
    return df

def load_nouns():
    "Load gendered nouns from Zhao et al. (2018) used by DisCo."

    df = pd.read_csv("data/disco/generalized_swaps.txt", sep="\t", header=None)
    return df
