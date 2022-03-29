import en_vectors_web_lg, random, re, json
import numpy as np

def tokenize(ques_list, use_glove):
    token_to_ix = {
        'PAD': 0,
        'UNK': 1,
    }

    spacy_tool = None
    pretrained_emb = []
    if use_glove:
        spacy_tool = en_vectors_web_lg.load()
        pretrained_emb.append(spacy_tool('PAD').vector)
        pretrained_emb.append(spacy_tool('UNK').vector)

    for ques in ques_list:
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques.lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for word in words:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)
                if use_glove:
                    pretrained_emb.append(spacy_tool(word).vector)

    pretrained_emb = np.array(pretrained_emb)

    return token_to_ix, pretrained_emb


def proc_ques(ques, token_to_ix, max_token):
    ques_ix = np.zeros(max_token, np.int64)

    words = re.sub(
        r"([.,'!?\"()*#:;])",
        '',
        ques.lower()
    ).replace('-', ' ').replace('/', ' ').split()
    q_len = 0
    for ix, word in enumerate(words):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
            q_len += 1
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix + 1 == max_token:
            break

    return ques_ix, q_len, len(words)

def ans_stat(ans_list):
    ans_to_ix, ix_to_ans = {}, {}
    for i, ans in enumerate(ans_list):
        ans_to_ix[ans] = i
        ix_to_ans[i] = ans

    return ans_to_ix, ix_to_ans

def shuffle_list(ans_list):
    random.shuffle(ans_list)

def qlen_to_key(q_len):
    if 1<= q_len <=3:
        return '1-3'
    if 4<= q_len <=8:
        return '4-8'
    if 9<= q_len:
        return '9-15'

def ans_to_key(ans_idx):
    if 0 <= ans_idx <= 99 :
        return '0-99'
    if 100 <= ans_idx <= 299 :
        return '100-299'
    if 300 <= ans_idx <= 999 :
        return '300-999'
