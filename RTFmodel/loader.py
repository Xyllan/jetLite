import numpy as np

vocab = set()


def get_word_embeddings():
    embeddings_index = {}
    all_words = []
    with open('../data/glove.6B.300d.txt') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            all_words.append(word)
    return all_words, embeddings_index
all_words, embeddings_index = get_word_embeddings()

basic_labels = ["OTHER", "GEN-AFF", "ORG-AFF", "PART-WHOLE", "PER-SOC", "PHYS", "ART"]

subtype_labels = ["OTHER", "GEN-AFF:Citizen-Resident-Religion-Ethnicity", "ORG-AFF:Employment", "PART-WHOLE:Subsidiary", "ORG-AFF:Membership", "ORG-AFF:Ownership", "PER-SOC:Business", "GEN-AFF:Org-Location", "PHYS:Located", "PART-WHOLE:Geographical", "ORG-AFF:Founder", "ART:User-Owner-Inventor-Manufacturer", "PHYS:Near", "PER-SOC:Family", "PART-WHOLE:Artifact", "PER-SOC:Lasting-Personal", "ORG-AFF:Student-Alum", "ORG-AFF:Investor-Shareholder", "ORG-AFF:Sports-Affiliation"]

subtype_order_labels = ["OTHER", "GEN-AFF:Citizen-Resident-Religion-Ethnicity", "ORG-AFF:Employment-1", "ORG-AFF:Employment", "GEN-AFF:Citizen-Resident-Religion-Ethnicity-1", "PART-WHOLE:Subsidiary-1", "ORG-AFF:Membership", "ORG-AFF:Ownership", "PER-SOC:Business", "GEN-AFF:Org-Location", "PHYS:Located-1", "PHYS:Located", "PART-WHOLE:Geographical", "ORG-AFF:Founder-1", "ORG-AFF:Membership-1", "PART-WHOLE:Geographical-1", "ART:User-Owner-Inventor-Manufacturer", "PHYS:Near", "ART:User-Owner-Inventor-Manufacturer-1", "PART-WHOLE:Subsidiary", "PHYS:Near-1", "PER-SOC:Family", "GEN-AFF:Org-Location-1", "PER-SOC:Family-1", "PART-WHOLE:Artifact-1", "PER-SOC:Business-1", "PART-WHOLE:Artifact", "PER-SOC:Lasting-Personal", "ORG-AFF:Student-Alum-1", "ORG-AFF:Founder", "ORG-AFF:Student-Alum", "ORG-AFF:Ownership-1", "ORG-AFF:Investor-Shareholder", "ORG-AFF:Investor-Shareholder-1", "ORG-AFF:Sports-Affiliation", "ORG-AFF:Sports-Affiliation-1", "PER-SOC:Lasting-Personal-1"]

labels = None
label_to_id = None
id_to_label = None
relation_detail = None


# basic -> only the 6 base types + OTHER
# subtype -> include subtypes
# subtype_with_order -> include ordering for subtypes
def set_relation_detail(rel_detail = 'basic'):
    global labels, label_to_id, id_to_label, relation_detail
    if rel_detail == 'basic':
        labels = basic_labels
    elif rel_detail == 'subtype':
        labels = subtype_labels
    elif rel_detail == 'subtype_with_order':
        labels = subtype_order_labels
    else:
        labels = None
    relation_detail = rel_detail
    label_to_id = {l: i for i, l in enumerate(labels)}
    id_to_label = {v: k for k, v in label_to_id.items()}


def get_embedding(word):
    try:
        return embeddings_index[word]
    except KeyError:
        return np.zeros((300), dtype = np.float32)

word_delim = '|||'
info_delim = '}}}'


def get_info(line):
    words = line.split(word_delim)
    label = label_to_id[words[0]]
    info_splits = [info.split(info_delim) for info in words[1:]]
    features = [np.hstack((get_embedding(info[0]), float(info[1]), float(info[2]))) for info in info_splits]
    return np.vstack(features), label


def load_examples(filename, force_load = False):
    import os
    x = None
    y = None
    base_name = os.path.splitext(filename)[0]
    try:
        x = np.load(base_name + '_x.npy')
        y = np.load(base_name + '_y.npy')
    except FileNotFoundError:
        force_load = True
    if force_load:
        with open(filename, "r") as fp:
            labels = []
            features = []
            for line in fp:
                feat, lbl = get_info(line)
                features.append(feat)
                labels.append(lbl)
            x = np.stack(features, axis=0)
            y = np.vstack(labels)
            np.save(base_name + '_x.npy', x)
            np.save(base_name + '_y.npy', y)
    return x, y


def load_training():
    suf = '7' if relation_detail == 'basic' else ('19' if relation_detail == 'subtype' else '37')
    return load_examples('training%s.txt' % suf)


def load_test():
    suf = '7' if relation_detail == 'basic' else ('19' if relation_detail == 'subtype' else '37')
    return load_examples('test%s.txt' % suf)
