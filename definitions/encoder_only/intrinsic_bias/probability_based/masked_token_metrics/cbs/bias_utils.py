import torch

def collate(eng, tgt):
    collate_dict = dict()

    for i, n in enumerate(tgt):
        collate_dict[n.lower()] = eng[i]

    return collate_dict


def how_many_tokens(word_list, tokenizer):
    word_token_dict = dict()
    for word in word_list:
        word = word.lower()
        tokens = tokenizer.tokenize(word)

        if str(len(tokens)) not in word_token_dict:
            word_token_dict[str(len(tokens))] = [word]
        else:
            word_token_dict[str(len(tokens))].append(word)

    return word_token_dict


def find_mask_token(tokenizer, sentence, how_many, MSK, last=False):
    tokens = tokenizer.encode(sentence)
    for i, tk in enumerate(tokens):
        if tk == MSK:
            if last == False:
                return list(range(i, i + how_many))
            else:
                last = False