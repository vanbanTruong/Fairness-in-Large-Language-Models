import glob
from models.bert import *
import pandas as pd
from time import sleep
from encoder_only.intrinsic_bias.similarity_based.seat import seat
from encoder_only.intrinsic_bias.similarity_based.seat.data import  load_json

def run_experiment():
  print("------------Encoder-only LMs: Intrinsic bias - Similarity_based bias - SEAT------------")
  BERT = BERT_MODEL()
  sleep(5)
  list_test_json = glob.glob("data/seat/*")
  res = []
  for test_json in list_test_json:
    encs = load_json(test_json)

    encs_targ1 = BERT.encode(encs["targ1"]["examples"])
    encs_targ2 = BERT.encode(encs["targ2"]["examples"])
    encs_attr1 = BERT.encode(encs["attr1"]["examples"])
    encs_attr2 = BERT.encode(encs["attr2"]["examples"])

    encs["targ1"]["encs"] = encs_targ1
    encs["targ2"]["encs"] = encs_targ2
    encs["attr1"]["encs"] = encs_attr1
    encs["attr2"]["encs"] = encs_attr2
    esize, pval = seat.run_test(encs, n_samples=10000)
    res.append({
      "Target 1": encs["targ1"]["category"],
      "Target 2": encs["targ2"]["category"],
      "Attribute 1": encs["attr1"]["category"],
      "Attribute 2": encs["attr2"]["category"],
      "Effect size": esize,
      "P-value": pval
    })

    print("-------------------------------------------------------------")

  df = pd.DataFrame(res)
  df.to_csv("encoder-only/intrinsic_bias/similarity_based/seat/result.csv", index=False)