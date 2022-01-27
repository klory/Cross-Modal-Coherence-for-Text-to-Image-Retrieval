import os
import json
import numpy as np
import re
import json
import pandas as pd
import pdb
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, multilabel_confusion_matrix
import torch

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def clean_state_dict(state_dict):
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k[:min(6,len(k))] == 'module' else k # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def infinite_loader(loader):
    """
    arguments:
        loader: torch.utils.data.DataLoader
    return:
        one batch of data
    usage:
        data = next(sample_data(loader))
    """
    while True:
        for batch in loader:
            yield batch


def compute_confidence_score(txt_feats, img_feats, classifier, valid_questions):
    N = len(txt_feats)
    confidences = []
    # print(f'\n*********** beta={beta:.4f} ************')
    for i in range(N):
        repeated_txt_feats = txt_feats[i:i+1].repeat(N,1) # [N, feat_dim]
        prob = torch.sigmoid(classifier(repeated_txt_feats, img_feats))[:, valid_questions]
        prob = prob.detach().cpu().numpy() # [N, valid_questions]

        # confidence = np.exp(beta*abs(prob-0.5)).sum(axis=1)

        entropy = - (prob * np.log(prob) + (1-prob) * np.log(1-prob)) # [N, valid_questions] # entropy
        confidence = (1.0 / entropy).sum(axis=1) # [N]

        confidences.append(confidence)
    confidences = np.stack(confidences, axis=0) # [N, N]
    return confidences

def compute_sim_mat(rcps, imgs, confidences, metric='sim', retrieved_type='image'):
    if metric == 'confidence':
        return confidences
    else:
        imgs_normed = imgs / np.linalg.norm(imgs, axis=1)[:, None]
        rcps_normed = rcps / np.linalg.norm(rcps, axis=1)[:, None]
        if retrieved_type == 'recipe':
            sims = np.dot(imgs_normed, rcps_normed.T)  # [N, N]
        else:
            sims = np.dot(rcps_normed, imgs_normed.T)
        
        # print(metric, sims)
        
        if 'confidence' in metric:
            T = 0.05
            # print(sims)
            # print(T)
            cnt = 0
            for row in range(len(sims)):
                sorted_sim = np.sort(sims[row])[::-1]
                diff = sorted_sim[0]-sorted_sim[1]
                if diff < T:
                    # print(row+1, diff)
                    cnt += 1
                    # plt.figure()
                    # plt.plot(sims[row], 'r.', label='sim')
                    # plt.plot(confidences[row], 'g.', label='conf')
                    sims[row] *= confidences[row]
                    # plt.plot(sims[row], 'b.', label='sim*conf')
                    # plt.legend()
                    # plt.savefig(f'outputs/{row+1}_diff={diff:.2f}.jpg')
            # print(cnt)

        return sims

def get_positions(sim_mat):
    # sort each row in DESCENDING order
    sorting = np.argsort(-1*sim_mat, axis=1)
    N = len(sorting)
    positions = []
    for i in range(N):
        row = sorting[i,:].tolist()
        pos = row.index(i)
        positions.append(pos+1)
    return np.array(positions)

def rank(rcps, imgs, confidences, metric='sim', retrieved_type='image', retrieved_range=500, verbose=True):
    N = retrieved_range
    data_size = imgs.shape[0]
    glob_medR = []
    glob_recall = {1: [], 5: [], 10: []}
    # average over 10 sets
    for i in range(1):
        ids_sub = np.random.choice(data_size, N, replace=False)
        ids_sub = np.arange(N)
        imgs_sub = imgs[ids_sub, :]
        rcps_sub = rcps[ids_sub, :]
        CIs_sub = confidences[ids_sub, :] # [N, all]
        CIs_sub = CIs_sub[:, ids_sub] # [N, N]
        sims = compute_sim_mat(rcps_sub, imgs_sub, CIs_sub, metric=metric, retrieved_type=retrieved_type) # [N, N]
        positions = get_positions(sims) # [N]

        sims.sort(axis=1)
        sims_sorted = np.flip(sims, axis=1)
        diffs = sims_sorted[:, 0] - sims_sorted[:, 1]
        # plt.figure()
        # plt.plot(positions, diffs, '.')
        # plt.savefig(f'outputs/{metric}_{i}.jpg')

        # with open(f'outputs/{metric}_0.05.txt', 'w') as f:
        #     f.write('\n'.join([f'{pos:>3d}, {diff:>6.2f}' for pos, diff in zip(positions, diffs)]))

        medR = np.median(positions)
        glob_medR.append(medR)
        # print(medR, ids_sub[:10])
        
        for k in glob_recall.keys():
            glob_recall[k].append( (positions<=k).sum()/N )

    glob_medR = np.array(glob_medR)

    for i in glob_recall.keys():
        glob_recall[i] = np.array(glob_recall[i])

    if verbose:
        print(f'Range = {retrieved_range}, MedR = {glob_medR.mean():.4f}({glob_medR.std():.4f})')
        for k,v in glob_recall.items():
            print(f'Recall@{k} = {v.mean():.4f}')
    
    return glob_medR, glob_recall


def make_saveDir(title, args=None):
    save_dir = title
    # save_dir += '_{}'.format(args.data_source)
    save_dir += '_retrieval={:.2f}'.format(args.weight_retrieval)
    save_dir += '_classification={:.2f}'.format(args.weight_classification)
    save_dir += '_reweight={:.2f}'.format(args.reweight_limit)
    save_dir += '_weightDecay={:.1f}'.format(args.weight_decay)
    save_dir += '_withAttention={}'.format(args.with_attention)
    save_dir += '_question={}'.format(args.question)
    save_dir += '_maxLen={}'.format(args.max_len)
#    save_dir += '_datasetQ={}'.format(args.dataset_q)
#    save_dir += '_debug={}'.format(args.debug)
    print('=> save_dir:', save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if args:
        with open(os.path.join(save_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    return save_dir


param_counter = lambda params: sum(p.numel() for p in params if p.requires_grad)


def load_recipes(file_path, relations, dataset_q=0):
    data = pd.read_csv(file_path, sep=',')
    n2p = []
    for rel in relations:
        class_dict = data[rel].value_counts().to_dict()
        n2p.append(class_dict[False] / class_dict[True])

    # experimental, ignore for now
    if dataset_q:
        q = abs(dataset_q)
        q_name = relations[q-2]
        if dataset_q > 0:
            data = data[data[q_name]==True]
        else:
            data = data[data[q_name]==False]
        n2p = [n2p[q-2]]
    return data, n2p


def clean_caption(cap):
    # remove non-breaking space: https://stackoverflow.com/a/11566398
    cap = cap.replace('\\xa0', '')
    # replace multiple spaces and tab with one space
    cap = re.sub(r' +|\\t', ' ', tok(cap))
    return cap


def tok(text, ts=False):
    if not ts:
        ts = [',', '.', ';', '(', ')', '?', '!', '&', '%', ':', '*', '"']
    for t in ts:
        text = text.replace(t, ' ' + t + ' ')
    return text


def get_caption_wordvec(cap, w2i, max_len=200):
    '''
    get the caption wordvec for the recipe, the
    number of items might be different for different
    recipe
    '''
#    cap = recipe#.caption
    cap = clean_caption(cap)
    words = re.split(r'\\n| ', cap)
    vec = np.zeros([max_len], dtype=np.int)
    num_words = min(max_len, len(words))
    for i in range(num_words):
        word = words[i]
        if word not in w2i:
            word = '<other>'
        vec[i] = w2i[word]
    return vec, num_words


_true_set = {'yes', 'true', 't', 'y', '1'}
_false_set = {'no', 'false', 'f', 'n', '0'}
def str2bool(value):
    if isinstance(value, str):
        value = value.lower()
        if value in _true_set:
            return True
        if value in _false_set:
            return False

def get_discourse_vec(recipe, relations):

    labels = []
    for rel in relations:
        val = recipe[rel]
        if isinstance(val, str):
            val = str2bool(val)
        if val:
            labels.append(1)
        else:
            labels.append(0)

    return labels


def multilabel_acc(ytrue, ypred):

    f1_for_each_class = f1_score(ytrue, ypred, average=None)
    confusion_for_each_class = multilabel_confusion_matrix(ytrue, ypred)

    return f1_for_each_class, confusion_for_each_class
