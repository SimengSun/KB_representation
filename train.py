__author__ = 'ssm'


'''
train Trans E model
https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf

1. initialize
2. for each epoch:
        for each batch:
            generate corrupted sample
            gradient descent
3. store entity_vecs, relation_vecs
'''


import Util
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt

eid, ide, rid, idr = Util.load_data()
triplets = Util.load_file(eid, rid, "train.txt")
test_triplets = Util.load_file(eid, rid, "test.txt")
valid_triplets = Util.load_file(eid, rid, "valid.txt")
true_triplets = test_triplets + triplets + valid_triplets
head_table, tail_table = Util.get_lookup_table(true_triplets)

relation_vecs = [0 for i in range(len(rid))]
entity_vecs = [0 for j in range(len(eid))]
epochs = 1000
batches = 100
batch_size = int(len(eid)/batches)
dim = 100.
margin = 1
rate = 0.01
flat_rate = 1e-12

plot_x = []
plot_y = []

def init():
    '''
    initialize relation_vecs and entity_vecs, of normal distribution
    :return:
    '''
    for id in ide:
        entity_vecs[id] = np.random.uniform(-6/math.sqrt(dim), 6/math.sqrt(dim), int(dim))
    for id in idr:
        relation_vecs[id] = np.random.uniform(-6/math.sqrt(dim), 6/math.sqrt(dim), int(dim))/dim

def dist(vec_1, vec_2):
    return np.linalg.norm(vec_1-vec_2)

def norm(vec):
    return vec / np.linalg.norm(vec)
    #return vec / len(vec)

def loop():
    '''
    :return:
    '''
    update_id = 0 # for updating learning rate
    for epoch in range(epochs):
        plot_x.append(epoch)
        total_loss = 0
        # normalize entity vecs
        for batch in range(batches):
            # sample batch
            batch_samples = []
            for i in range(batch_size):
                randn = np.random.randint(0, len(triplets))
                batch_samples.append(triplets[randn])

            # generate corrupted samples
            combined = [] # [((h, t, l),(h', t', l)),...]
            for triplet in batch_samples: # (head, tail, relation)
                if np.random.random() < 0.5:
                    randn_h = 0
                    while 1:
                        randn_h = np.random.randint(0, len(eid))
                        if randn_h not in tail_table[(triplet[1], triplet[2])]:
                            break
                    combined.append(((randn_h, triplet[1], triplet[2]), triplet))
                else:
                    randn_t = 0
                    while 1:
                        randn_t = np.random.randint(0, len(eid))
                        if randn_t not in head_table[(triplet[0], triplet[2])]:
                            break
                    combined.append(((triplet[0], randn_t, triplet[2]), triplet))

            #SGD EBM
            for c in combined:
                h_ = c[0][0]
                t_ = c[0][1]
                l = c[0][2]
                h = c[1][0]
                t = c[1][1]
                loss = margin + dist(entity_vecs[h] + relation_vecs[l], entity_vecs[t]) - \
                    dist(entity_vecs[h_] + relation_vecs[l], entity_vecs[t_])
                if  loss > 0:
                    update_id += 1.
                    #r = rate
                    #r = rate * math.pow(flat_rate / rate, update_id/(epochs*batches*batch_size))
                    r = rate * math.exp(-update_id * epochs/(epochs*batches*batch_size))
                    derv = 2 * (entity_vecs[h] + relation_vecs[l] - entity_vecs[t])
                    # dig holes for desired
                    entity_vecs[h] -= norm(derv * r)
                    entity_vecs[t] -= -1 * norm(derv * r)
                    relation_vecs[l] -= derv * r

                    derv = 2 * (entity_vecs[h_] + relation_vecs[l] - entity_vecs[t_])
                    # build hills for corrupt
                    entity_vecs[h_] += norm(derv * r)
                    entity_vecs[t_] += -1 * norm(derv * r)
                    total_loss += loss
        print("epoch {} {} {}".format(epoch, datetime.datetime.now(), total_loss))
        plot_y.append(total_loss)

#initialize vecs
init()
#loop
loop()
#store result
np.save( ".\\data\\transE_en", entity_vecs)
np.save(".\\data\\transE_re", relation_vecs)

plt.plot(plot_x, plot_y, '-+r')
plt.show()
