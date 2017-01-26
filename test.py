__author__ = 'ssm'

'''
test and evaluate transE
return median( median of rank of the correct triplets among all corrupt ones), hits@top10
'''

import Util
import numpy as np
import datetime

eid, ide, rid, idr = Util.load_data()
test_triplets = Util.load_file(eid, rid, "test.txt")
train_triplets = Util.load_file(eid, rid, "train.txt")
valid_triplets = Util.load_file(eid, rid, "valid.txt")
true_triplets = test_triplets + train_triplets + valid_triplets
head_table, tail_table = Util.get_lookup_table(true_triplets)
relation_vecs = np.load('.\\data\\transE_re.npy')
entity_vecs = np.load('.\\data\\transE_en.npy')


def dist(vec_1, vec_2):
    return np.linalg.norm(vec_1-vec_2)

def get_rank(sorted_dissim, triplet):
    '''
    sorted_dissim : [((triplet_1), dist), ((triplet_2), dist), ...]
    '''
    for i in range(len(sorted_dissim)):
        if sorted_dissim[i][0] == triplet:
            return i
    print('cannot find correct triplet')
    exit()

def test():
    '''
    for each triplet_t
        1. generate corrupt sets
        2. delete true triplets
        sort ascending, store t's rank
        record hits@top 10
    '''
    hits10 = 0
    ranks = []
    i = 0
    print('total {}'.format(len(test_triplets)))
    for triplet in test_triplets[:500]:

        i += 1
        #replace head
        #generate corrupt sets (replace only head)
        c_set = []
        for id in ide:
            if id not in tail_table[(triplet[1], triplet[2])]:
                c_set.append((id, triplet[1], triplet[2]))
        c_set.append(triplet) # for getting rank
        #calculate dissimilarity
        dissimilarity = {}
        for t in c_set:
            dissimilarity[t] = dist(entity_vecs[t[0]] + relation_vecs[t[2]], entity_vecs[t[1]])

        sorted_dissim = sorted(dissimilarity.items(), key=lambda a: a[1])
        rank = get_rank(sorted_dissim, triplet)
        ranks.append(rank)
        print('process {} {} rank {} total {}'.format(i, datetime.datetime.now(), rank, len(ide)))
        if rank < 10:
            hits10 += 1

        #replace tail
        #generate corrupt sets (replace only tail)
        c_set = []
        for id in ide:
            if id not in head_table[(triplet[0], triplet[2])]:
                c_set.append((triplet[0], id, triplet[2]))
        c_set.append(triplet) # for getting rank
        #calculate dissimilarity
        dissimilarity = {}
        for t in c_set:
            dissimilarity[t] = dist(entity_vecs[t[0]] + relation_vecs[t[2]], entity_vecs[t[1]])
        sorted_dissim = sorted(dissimilarity.items(), key=lambda a: a[1])
        rank = get_rank(sorted_dissim, triplet)
        ranks.append(rank)
        print('process {} {} rank {} total {}'.format(i, datetime.datetime.now(), rank, len(ide)))
        if rank < 10:
            hits10 += 1
    print("number of test triplet * 2 : {}".format(len(test_triplets)*2))
    print("hits @ top 10 : {}".format(hits10))
    print("median of all ranks : {}".format(sum(ranks)/len(ranks)))

test()