__author__ = 'ssm'

def load_data():
    '''
    :return: entity-id, id-entity, relation-id, id-relation
            {entity:id, entity:id,...}
    '''
    entity_id = {}
    id_entity = {}
    relation_id = {}
    id_relation = {}
    with open(".\\data\\entity2id.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            sp = line.split('\t')
            if len(sp) < 2:
                continue
            entity_id[sp[0]] = int(sp[1].strip())
            id_entity[int(sp[1].strip())] = sp[0]
    with open(".\\data\\relation2id.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            sp = line.split('\t')
            if len(sp) < 2:
                continue
            relation_id[sp[0]] = int(sp[1].strip())
            id_relation[int(sp[1].strip())] = sp[0]
    return entity_id, id_entity, relation_id, id_relation

def load_file(enityt_id, relation_id, file):
    '''
    :return: (h, r, t) list
            [(h1, r1, t1), (h2, r2, t2),...]
    '''
    ret = []
    with open(".\\data\\" + file, "r") as f:
        lines = f.readlines()
        for line in lines[:-1]:
            sp = line.split('\t')
            if len(sp) < 3:
                continue
            if sp[0].strip() not in enityt_id:
                print('missing entity')
            ret.append((enityt_id[sp[0].strip()], enityt_id[sp[1].strip()], relation_id[sp[2].strip()]))
    return ret

def get_lookup_table(triplet_list):
    '''
    triplet_list = [(h,t,r), ..]
    return :
        head_table {(h1,r1):[t1, t2, ...], (h2,r2):[t1, t2, ...]}
        same with tail_table
    '''
    head_table = {}
    tail_table = {}
    for triplet in triplet_list:
        if (triplet[0], triplet[2]) in head_table:
            head_table[(triplet[0], triplet[2])].append(triplet[1])
        else:
            head_table[(triplet[0], triplet[2])] = [triplet[1]]
        if (triplet[1], triplet[2]) in tail_table:
            tail_table[(triplet[1], triplet[2])].append(triplet[0])
        else:
            tail_table[(triplet[1], triplet[2])] = [triplet[0]]
    return head_table, tail_table

'''
eid, ide, rid, idr = load_data()
train = load_file(eid, rid, "train.txt")
print(train[:3])
'''