from collections import defaultdict
import torch
from tqdm import tqdm
import pandas as pd
from encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.data import *
from encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.evaluate import *

def calculate_aula(model, token_ids, log_softmax, attention, device):
    '''
    Given token ids of a sequence, return the averaged log probability of
    unmasked sequence (AULA or AUL).
    '''
    token_ids = token_ids.to(device)
    output = model(token_ids)
    logits = output.logits.squeeze(0)
    log_probs = log_softmax(logits)
    token_ids = token_ids.view(-1, 1).detach()
    token_log_probs = log_probs.gather(1, token_ids)[1:-1]
    if attention:
        attentions = torch.mean(torch.cat(output.attentions, 0), 0)
        averaged_attentions = torch.mean(attentions, 0)
        averaged_token_attentions = torch.mean(averaged_attentions, 0)
        token_log_probs = token_log_probs.squeeze(1) * averaged_token_attentions[1:-1]
    sentence_log_prob = torch.mean(token_log_probs)
    score = sentence_log_prob.item()

    ranks = get_rank_for_gold_token(log_probs, token_ids)

    return score, ranks

def run_experiment():
    print("------------Encoder-only LMs: Intrinsic bias - Probability-based bias - AUL------------")
    tokenizer, model = load_tokenizer_and_model()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    bias_score_list = []
    bias_type_list = []
    dataset_list = []
    for i in range(0, 2):
        total_score = 0
        stereo_score = 0
        log_softmax = torch.nn.LogSoftmax(dim=1)
        count = defaultdict(int)
        scores = defaultdict(int)
        all_ranks = []
        if i == 0:
            print("------------ AUL - Crows-Pairs Dataset------------")
            inputs = load_crows_pairs_dataset()
            dataset_name = "Crows-Pairs"
        else:
            print("------------ AUL - Stereoset Dataset------------")
            inputs = load_stereoset_dataset()
            dataset_name = "Stereoset"

        for input in tqdm(inputs):
            bias_type = input['bias_type']
            count[bias_type] += 1

            pro_sentence = input['stereotype']
            pro_token_ids = tokenizer.encode(pro_sentence, return_tensors='pt').to(device)
            anti_sentence = input['anti-stereotype']
            anti_token_ids = tokenizer.encode(anti_sentence, return_tensors='pt').to(device)

            with torch.no_grad():
                attention = False
                pro_score, pro_ranks = calculate_aula(model, pro_token_ids, log_softmax, attention, device)
                anti_score, anti_ranks = calculate_aula(model, anti_token_ids, log_softmax, attention, device)
                
            all_ranks += anti_ranks
            all_ranks += pro_ranks
            total_score += 1
            if pro_score > anti_score:
                stereo_score += 1
                scores[bias_type] += 1

        bias_score = (stereo_score / total_score) * 100

        bias_score_list.append(bias_score)
        bias_type_list.append("Bias score")
        dataset_list.append(dataset_name)

        for bias_type, score in sorted(scores.items()):
            bias_score = (score / count[bias_type]) * 100
            bias_score_list.append(bias_score)
            bias_type_list.append(bias_type)
            dataset_list.append(dataset_name)
  
        all_ranks = [rank for rank in all_ranks if rank != -1]
        accuracy = sum([1 for rank in all_ranks if rank == 1]) / len(all_ranks)
        accuracy *= 100
        bias_score_list.append(accuracy)
        bias_type_list.append('Accuracy')
        dataset_list.append(dataset_name)

    df = pd.DataFrame({
        'Bias_type': bias_type_list,
        'Dataset': dataset_list,
        'Score': bias_score_list,
    })

    df.to_csv("encoder_only/intrinsic_bias/probability_based/pseudo_log_likelihood_metrics/aul/result.csv", index=False)