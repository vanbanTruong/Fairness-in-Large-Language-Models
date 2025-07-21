import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('transformers.tokenization_utils').disabled = True
from transformers import BertModel, BertTokenizer
import json
import pickle
import datetime
import json

from models.bert import *

f = open('data/ceat/data.json')
data = json.load(f)
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-cased')
model_bert = BertModel.from_pretrained('bert-base-cased')

def short_sen(sen,wd):
    """
    shorten the raw comment, take only 9 words including the target word
    """
    wds = sen.split()
    wd_idx = wds.index(wd)
    if len(wds) >=9:
        if wd_idx < 4:
            wds_used = wds[:9]
        elif (len(wds) - wd_idx - 1 < 4):
            wds_used = wds[-9:]
        else:
            wds_used = wds[(wd_idx-4):(wd_idx+4)]
        new_sen = ' '.join(wds_used)
    else:
        new_sen = sen
    return new_sen

def bert(wd_lst):
    sen_dict = pickle.load(open('data/ceat/sen_dic_1.pickle','rb'))
    wd_idx_dict = {wd:[] for wd in wd_lst}
    out_dict = {wd:[] for wd in wd_lst}
    for wd in wd_lst:
        current_idx = torch.tensor(tokenizer_bert.encode(wd,add_special_tokens=False)).unsqueeze(0).tolist()[0]
        wd_idx_dict[wd] = current_idx
    
    i = 0
    for wd in wd_lst:
        target = wd_idx_dict[wd][-1]
        tem = []
        for idx,sen in enumerate(sen_dict[wd]):
            i += 1
            if i%5000 == 0:
                now = datetime.datetime.now()
                print(now.strftime("%Y-%m-%d %H:%M:%S"))
                print(str(i)+' finished.')
            if idx == 1000:
                break

            sen = short_sen(sen,wd)
            input_ids = torch.tensor(tokenizer_bert.encode(sen, add_special_tokens=False)).unsqueeze(0) 
            exact_idx = input_ids.tolist()[0].index(target)
            outputs = model_bert(input_ids)
            exact_state_vector = outputs[0][0,int(exact_idx),:].cpu().detach().numpy() 
            out_dict[wd].append(exact_state_vector)
    n = 'data/ceat/bert_weat.pickle'
    pickle.dump(out_dict,open(n,'wb'))

def generate():
    lst = []
    distinct_values = set()

    for key, values in data.items():
        for value in values:
            distinct_values.add(value)

    lst = distinct_values

    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    bert(lst)
    print("bert finish")
