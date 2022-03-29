"""
Author: Mateusz Malinowski
Email: mmalinow@mpi-inf.mpg.de

The script assumes there are two files
- first file with ground truth answers
- second file with predicted answers
both answers are line-aligned

The script also assumes that answer items are comma separated.
For instance, chair,table,window

It is also a set measure, so not exactly the same as accuracy 
even if dirac measure is used since {book,book}=={book}, also {book,chair}={chair,book}

Logs:
    05.09.2015 - white spaces surrounding words are stripped away so that {book, chair}={book,chair}
"""

import sys

#import enchant

from numpy import prod
from nltk.corpus import wordnet as wn
from tqdm import tqdm

def file2list(filepath):
    with open(filepath,'r') as f:
        lines =[k for k in 
            [k.strip() for k in f.readlines()] 
        if len(k) > 0]

    return lines


def list2file(filepath,mylist):
    mylist='\n'.join(mylist)
    with open(filepath,'w') as f:
        f.writelines(mylist)


def items2list(x):
    """
    x - string of comma-separated answer items
    """
    return [l.strip() for l in x.split(',')]


def fuzzy_set_membership_measure(x,A,m):
    """
    Set membership measure.
    x: element
    A: set of elements
    m: point-wise element-to-element measure m(a,b) ~ similarity(a,b)

    This function implments a fuzzy set membership measure:
        m(x \in A) = max_{a \in A} m(x,a)}
    """
    return 0 if A==[] else max(map(lambda a: m(x,a), A))


def score_it(A,T,m):
    """
    A: list of A items 
    T: list of T items
    m: set membership measure
        m(a \in A) gives a membership quality of a into A 

    This function implements a fuzzy accuracy score:
        score(A,T) = min{prod_{a \in A} m(a \in T), prod_{t \in T} m(a \in A)}
        where A and T are set representations of the answers
        and m is a measure
    """
    if A==[] and T==[]:
        return 1

    # print A,T

    score_left=0 if A==[] else prod(list(map(lambda a: m(a,T), A)))
    score_right=0 if T==[] else prod(list(map(lambda t: m(t,A),T)))
    return min(score_left,score_right) 


# implementations of different measure functions
def dirac_measure(a,b):
    """
    Returns 1 iff a=b and 0 otherwise.
    """
    if a==[] or b==[]:
        return 0.0
    return float(a==b)


def wup_measure(a,b,similarity_threshold=0.925):
    """
    Returns Wu-Palmer similarity score.
    More specifically, it computes:
        max_{x \in interp(a)} max_{y \in interp(b)} wup(x,y)
        where interp is a 'interpretation field'
    """
    def get_semantic_field(a):
        weight = 1.0
        semantic_field = wn.synsets(a,pos=wn.NOUN)
        return (semantic_field,weight)


    def get_stem_word(a):
        """
        Sometimes answer has form word\d+:wordid.
        If so we return word and downweight
        """
        weight = 1.0
        return (a,weight)


    global_weight=1.0

    (a,global_weight_a)=get_stem_word(a)
    (b,global_weight_b)=get_stem_word(b)
    global_weight = min(global_weight_a,global_weight_b)

    if a==b:
        # they are the same
        return 1.0*global_weight

    if a==[] or b==[]:
        return 0


    interp_a,weight_a = get_semantic_field(a) 
    interp_b,weight_b = get_semantic_field(b)

    if interp_a == [] or interp_b == []:
        return 0

    # we take the most optimistic interpretation
    global_max=0.0
    for x in interp_a:
        for y in interp_b:
            local_score=x.wup_similarity(y)
            if local_score > global_max:
                global_max=local_score

    # we need to use the semantic fields and therefore we downweight
    # unless the score is high which indicates both are synonyms
    if global_max < similarity_threshold:
        interp_weight = 0.1
    else:
        interp_weight = 1.0

    final_score=global_max*weight_a*weight_b*interp_weight*global_weight
    return final_score 
###


def get_scores(input_gt, input_pred, threshold_0=0.0, threshold_1=0.9):
    element_membership_acc=dirac_measure
    element_membership_wups_0=lambda x,y: wup_measure(x,y,threshold_0)
    element_membership_wups_1=lambda x,y: wup_measure(x,y,threshold_1)

    set_membership_acc=\
            lambda x,A: fuzzy_set_membership_measure(x,A,element_membership_acc)
    set_membership_wups_0=\
            lambda x,A: fuzzy_set_membership_measure(x,A,element_membership_wups_0)
    set_membership_wups_1=\
            lambda x,A: fuzzy_set_membership_measure(x,A,element_membership_wups_1)

    score_list_acc    = []
    score_list_wups_0 = []
    score_list_wups_1 = []
    pbar = tqdm(zip(input_gt,input_pred))
    pbar.set_description('Computing Acc')

    for (ta,pa) in pbar:
        score_list_acc.append(score_it(items2list(ta),items2list(pa),set_membership_acc)) 
            
    #final_score=sum(map(lambda x:float(x)/float(len(score_list)),score_list))
    final_score_acc=float(sum(score_list_acc))/float(len(score_list_acc))
    final_score_acc *= 100.0

    pbar = tqdm(zip(input_gt,input_pred))
    pbar.set_description('Computing Wups_0.0')
    for (ta,pa) in pbar:
        score_list_wups_0.append(score_it(items2list(ta),items2list(pa),set_membership_wups_0))
    #final_score=sum(map(lambda x:float(x)/float(len(score_list)),score_list))
    final_score_wups_0=float(sum(score_list_wups_0))/float(len(score_list_wups_0))
    final_score_wups_0 *= 100.0

    pbar = tqdm(zip(input_gt,input_pred))
    pbar.set_description('Computing Wups_0.9')
    for (ta,pa) in pbar:
        score_list_wups_1.append(score_it(items2list(ta),items2list(pa),set_membership_wups_1)) 
    #final_score=sum(map(lambda x:float(x)/float(len(score_list)),score_list))
    final_score_wups_1=float(sum(score_list_wups_1))/float(len(score_list_wups_1))
    final_score_wups_1 *= 100.0 

    # filtering to obtain the results
    #print 'full score:', score_list
    # print('accuracy = {0:.2f} | WUPS@{1} = {2:.2f} | WUPS@{3} = {4:.2f}'.format(
    #     final_score_acc, threshold_0, final_score_wups_0, threshold_1, final_score_wups_1))
    return final_score_acc, final_score_wups_0, final_score_wups_1

def get_acc(gts, preds):
    sum_correct = 0
    assert len(gts) == len(preds)
    for gt, pred in zip(gts, preds):
        if gt == pred:
            sum_correct += 1
    acc = 100.0 * float(sum_correct/ len(gts))
    return acc
