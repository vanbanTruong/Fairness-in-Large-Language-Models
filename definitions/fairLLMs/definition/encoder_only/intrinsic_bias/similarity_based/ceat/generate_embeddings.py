import os, pickle
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

_DIR = os.path.dirname(os.path.abspath(__file__))
SEN_DICT = os.path.join(_DIR, "sen_dic_1.pickle")
MAX_SENTENCES = 1000
MODEL = "bert-base-uncased"

european_3 = ['Adam','Harry','Roger','Alan','Ryan','Andrew','Jack','Matthew','Stephen',
              'Brad','Greg','Paul','Jonathan','Peter','Amanda','Courtney','Melanie','Katie',
              'Kristin','Nancy','Stephanie','Ellen','Lauren','Colleen','Emily','Megan',
              'Rachel','Betsy','Justin','Frank','Josh','Heather']
african_3  = ['Alonzo','Theo','Alphonse','Jerome','Leroy','Torrance','Darnell','Lamar',
              'Lionel','Tyree','Deion','Lamont','Malik','Terrence','Tyrone','Lavon',
              'Marcellus','Wardell','Nichelle','Ebony','Shaniqua','Jasmine','Tanisha','Tia',
              'Latoya','Yolanda','Yvette','Jamel','Shereen','Latisha','Lakisha','Malika']
pleasant_3 = ['caress','freedom','health','love','peace','cheer','friend','heaven','loyal',
              'pleasure','diamond','gentle','honest','lucky','rainbow','diploma','gift',
              'honor','miracle','sunrise','family','happy','laughter','paradise','vacation']
unpleasant_3 = ['abuse','crash','filth','murder','sickness','accident','death','grief',
                'poison','stink','assault','disaster','hatred','pollute','tragedy','bomb',
                'divorce','jail','poverty','ugly','cancer','evil','kill','rotten','vomit']

male   = ['John','Paul','Mike','Kevin','Steve','Greg','Jeff','Bill']
female = ['Amy','Joan','Lisa','Sarah','Diana','Kate','Ann','Donna']
career = ['executive','management','professional','corporation','salary','office','business','career']
family = ['home','parents','children','family','cousins','marriage','wedding','relatives']

mental_disease   = ['sad','hopeless','gloomy','tearful','miserable','depressed']
physical_disease = ['sick','illness','influenza','disease','virus','cancer']
temporary = ['impermanent','unstable','variable','fleeting','short','brief','occasional']
permanent = ['stable','always','constant','persistent','chronic','prolonged','forever']

young_name = ['Tiffany','Michelle','Cindy','Kristy','Brad','Eric','Joey','Billy']
old_name   = ['Ethel','Bernice','Gertrude','Agnes','Cecil','Wilbert','Mortimer','Edgar']
pleasant_5   = ['joy','love','peace','wonderful','pleasure','friend','laughter','happy']
unpleasant_5 = ['agony','terrible','horrible','nasty','evil','war','awful','failure']


GUO_CANDIDATES = {
    3:  (("european", european_3), ("african", african_3),
         ("pleasant", pleasant_3), ("unpleasant", unpleasant_3)),
    6:  (("male", male), ("female", female),
         ("career", career), ("family", family)),
    9:  (("mental", mental_disease), ("physical", physical_disease),
         ("temporary", temporary), ("permanent", permanent)),
    10: (("young", young_name), ("old", old_name),
         ("pleasant", pleasant_5), ("unpleasant", unpleasant_5)),
}


def load_bert():
    tok = BertTokenizer.from_pretrained(MODEL)
    model = BertModel.from_pretrained(MODEL)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()
    print(f"[INFO] {MODEL} on {dev}")
    return tok, model, dev


@torch.no_grad()
def embed_word(word, sentence, tok, model, dev):
    """Mean of the word's subword-token hidden states in `sentence`.
    Uncased: match lowercased word ids. None if not locatable."""
    enc = tok(sentence, return_tensors="pt", truncation=True, max_length=128).to(dev)
    wids = tok(word.lower(), add_special_tokens=False)["input_ids"]
    if not wids:
        return None
    ids = enc["input_ids"][0].tolist()
    for i in range(len(ids) - len(wids) + 1):
        if ids[i:i+len(wids)] == wids:
            h = model(**enc).last_hidden_state[0]
            return h[i:i+len(wids)].mean(dim=0).cpu().numpy()
    return None


def build_pool(word, sents, tok, model, dev):
    vecs = []
    for s in sents[:MAX_SENTENCES]:
        v = embed_word(word, s, tok, model, dev)
        if v is not None:
            vecs.append(v)
    return np.array(vecs) if vecs else None


def filter_and_equalize(groups, keys):
    """Drop words absent from the pickle; truncate T1/T2 to equal length."""
    resolved = []
    for name, words in groups:
        kept = [w for w in words if w in keys]
        dropped = [w for w in words if w not in keys]
        if dropped:
            print(f"    [drop] {name}: {dropped}")
        resolved.append((name, kept))
    (n1, w1), (n2, w2) = resolved[0], resolved[1]
    if len(w1) != len(w2):
        m = min(len(w1), len(w2))
        print(f"    [WARN] targets unequal ({n1}={len(w1)}, {n2}={len(w2)}) "
              f"-> truncating both to {m}")
        resolved[0] = (n1, w1[:m])
        resolved[1] = (n2, w2[:m])
    return resolved


def main():
    sen_d = pickle.load(open(SEN_DICT, "rb"))
    keys = set(sen_d)
    tok, model, dev = load_bert()

    for num, groups in GUO_CANDIDATES.items():
        print(f"\n=== WEAT {num} ===")
        resolved = filter_and_equalize(groups, keys)
        merged = {}
        for name, words in resolved:
            for w in words:
                pool = build_pool(w, sen_d.get(w, []), tok, model, dev)
                if pool is None:
                    print(f"    [WARN] '{w}': 0 usable embeddings")
                else:
                    merged[w] = pool
                    print(f"    {w}: {len(pool)}")
        out = os.path.join(_DIR, f"bert_weat{num}.pickle")
        pickle.dump(merged, open(out, "wb"))
        print(f"  saved {out} ({len(merged)} words)")


if __name__ == "__main__":
    main()