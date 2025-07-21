import torch
import difflib

from transformers import AutoModelForMaskedLM, AutoTokenizer

def load_tokenizer_and_model():
    '''
    Load tokenizer and model to evaluate.
    '''
    pretrained_weights = 'bert-base-cased'
    model = AutoModelForMaskedLM.from_pretrained(pretrained_weights,
                                                 output_hidden_states=True,
                                                 output_attentions=True,
                                                 cache_dir = "bert-base-cache")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)

    model = model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    return tokenizer, model


def get_span(seq1, seq2, operation):
    '''
    Extract spans that are shared or diffirent between two sequences.

    Parameters
    ----------
    operation: str
        You can select "equal" which extract spans that are shared between
        two sequences or "diff" which extract spans that are diffirent between
        two sequences.
    '''
    seq1 = [str(x) for x in seq1.tolist()]
    seq2 = [str(x) for x in seq2.tolist()]

    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    template1, template2 = [], []
    for op in matcher.get_opcodes():
        if (operation == 'equal' and op[0] == 'equal') \
                or (operation == 'diff' and op[0] != 'equal'):
            template1 += [x for x in range(op[1], op[2], 1)]
            template2 += [x for x in range(op[3], op[4], 1)]

    return template1, template2


def get_rank_for_gold_token(log_probs, token_ids):
    '''
    Get rank for gold token from log probability.
    '''
    sorted_indexes = torch.sort(log_probs, dim=1, descending=True)[1]
    ranks = torch.where(sorted_indexes == token_ids)[1] + 1
    ranks = ranks.tolist()

    return ranks