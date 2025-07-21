from transformers import BertForMaskedLM, BertTokenizer, AutoConfig
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from pattern3.text.en import pluralize
from encoder_only.intrinsic_bias.probability_based.masked_token_metrics.cbs.bias_utils import collate, how_many_tokens, find_mask_token
from encoder_only.intrinsic_bias.probability_based.masked_token_metrics.cbs.configuration import configuration

device = torch.device("cuda:"+str(0))

def log_probability_for_single_sentence(model, tokenizer,
                                        template, attr, nation_dict, en_nationality, nationality, MSK, last=False, use_pretrained=False):

    col_dict = collate(en_nationality, nationality)
    vocab = tokenizer.get_vocab()
    softmax = torch.nn.Softmax(dim=-1)  # Specify dimension for softmax

    results = []

    attribute_num = len(tokenizer.tokenize(attr))
    for number in nation_dict.keys():

        nations = nation_dict[number]
        how_many = int(number)
        target_mask = ' '.join(['[MASK]' for _ in range(how_many)])
        attribute_mask = ' '.join(['[MASK]' for _ in range(attribute_num)])

        if '[AAA]' in template:
            sentence = template.replace('[TTT]', target_mask).replace('[AAA]', attr)
            prior_sentence = template.replace('[TTT]', target_mask).replace('[AAA]', attribute_mask)
        else:
            sentence = template.replace('[TTT]', target_mask).replace('[AAAs]', pluralize(attr))
            prior_sentence = template.replace('[TTT]', target_mask).replace('[AAAs]', attribute_mask)

        input_ids = tokenizer(sentence, return_tensors='pt').to(device)

        if not use_pretrained:
            target_prob = model(**input_ids)
        else:
            target_prob = model(**input_ids)[0]

        prior_input_ids = tokenizer(prior_sentence, return_tensors='pt').to(device)

        if not use_pretrained:
            prior_prob = model(**prior_input_ids)
        else:
            prior_prob = model(**prior_input_ids)[0]
        
        masked_tokens = find_mask_token(tokenizer, sentence, how_many, MSK)
        masked_tokens_prior = find_mask_token(tokenizer, prior_sentence, how_many, MSK, last)

        logits = []
        prior_logits = []
        for mask in masked_tokens:
            logits.append(softmax(target_prob[0][mask]).detach().cpu().numpy())

        for mask in masked_tokens_prior:
            prior_logits.append(softmax(prior_prob[0][mask]).detach().cpu().numpy())

        for nat in nations:
            ddf = [col_dict[nat]]
            nat_logit = 1.0
            nat_prior_logit = 1.0

            for token in tokenizer.tokenize(nat):
                for logit in logits:
                    nat_logit *= float(logit[vocab[token]])
                for prior_logit in prior_logits:
                    nat_prior_logit *= float(prior_logit[vocab[token]])

            normalized_prob = np.log(float(nat_logit / nat_prior_logit))
            ddf.append(normalized_prob)
            results.append(ddf)

    return pd.DataFrame(results, columns=['nationality', 'normalized_prob']).sort_values(
        "normalized_prob", ascending=False)


def log_probability_for_single_sentence_multiple_attr(model, tokenizer,
                                                      template, occ, nation_dict, en_nationality, nationality, MSK, use_pretrained=False):
    last = False
    if template.find('[TTT]') > template.find('[AAA]') and template.find('[TTT]') > template.find('[AAAs]'):
        last = True

    mean_scores = []
    var_scores = []
    std_scores = []

    for attr in occ:
        ret_df = log_probability_for_single_sentence(model, tokenizer,
                                                      template, attr, nation_dict, en_nationality, nationality, MSK, last, use_pretrained)

        mean_scores.append(ret_df['normalized_prob'].mean())
        var_scores.append(ret_df['normalized_prob'].var())
        std_scores.append(ret_df['normalized_prob'].std())

    mean_scores = np.array(mean_scores)
    var_scores = np.array(var_scores)
    std_scores = np.array(std_scores)

    return mean_scores, var_scores, std_scores


def log_probability_for_multiple_sentence(model, tokenizer, templates, occ, en_nationality, nationality, MSK, use_pretrained=False):

    nation_dict = how_many_tokens(nationality, tokenizer)

    total_mean = []
    total_var = []
    total_std = []

    for template in tqdm(templates):
        m, v, s = log_probability_for_single_sentence_multiple_attr(model, tokenizer,
                                                                    template, occ, nation_dict, en_nationality, nationality, MSK, use_pretrained)

        total_mean.append(m.mean())
        total_var.append(v.mean())
        total_std.append(s.mean())

    return total_mean, total_var, total_std

def run_experiment():
    print("------------Encoder-only LMs: Intrinsic bias - Probability-based bias - CBS------------")
    CBS = []
    for language in ["en", "zh"]:
        print(f"------------CBS in {language}------------")
        nationality = configuration[language]['nationality']
        bert_model = configuration[language]['bert_model']
        template_path = configuration[language]['template_path']
        occ_path = configuration[language]['occ_path']
        MSK = configuration[language]['MSK']

        en_nationality = configuration['en']['nationality']

        tokenizer = BertTokenizer.from_pretrained(bert_model)
        MSK = tokenizer.mask_token_id

        print("Using pretrained model!")
        model = BertForMaskedLM.from_pretrained(bert_model)

        model.eval()
        model.to(device)

        with open(occ_path, 'r', encoding="utf8") as f:
            tt = f.readlines()

        occ = []

        for i in range(len(tt)):
            occ.append(tt[i].rstrip())

        print("Occupations loading complete!")

        with open(template_path, 'r', encoding="utf8") as f:
            tt = f.readlines()

        templates = []

        for i in range(len(tt)):
            templates.append(tt[i].rstrip())
        print("Templates loading complete!")


        total_mean, total_var, total_std = log_probability_for_multiple_sentence(model, tokenizer, templates, occ, en_nationality, nationality, MSK, use_pretrained=True)

        print("CB score of {} in {} : {}".format(bert_model, language, np.array(total_var).mean()))
        CBS.append(np.array(total_var).mean())

    df = pd.DataFrame({
        'Language': ["English", "Chinese"],
        'CB Score': CBS,
    })

    df.to_csv("encoder_only/intrinsic_bias/probability_based/masked_token_metrics/cbs/result.csv", index=False)