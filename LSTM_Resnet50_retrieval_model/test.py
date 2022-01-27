# CITE++
# Base: CUDA_VISIBLE_DEVICES=7 python test.py --data_source=cite --resume=runs/samples3439_retrieval=1.00_classification=0.00_reweight=1000.00_weightDecay=0.0_withAttention=0_question=2,3,4,5,6,7,8_maxLen=200/e19.ckpt --question=2,3,4,5,6,7,8
# CMCA: CUDA_VISIBLE_DEVICES=7 python test.py --data_source=cite --resume=runs/samples3439_retrieval=1.00_classification=0.00_reweight=1000.00_weightDecay=0.0_withAttention=2_question=2,3,4,5,6,7,8_maxLen=200/e19.ckpt --question=2,3,4,5,6,7,8
# CMCM_NoAttn: CUDA_VISIBLE_DEVICES=7 python test.py --data_source=cite --resume=runs/samples3439_retrieval=1.00_classification=0.10_reweight=1000.00_weightDecay=0.0_withAttention=0_question=2,3,4,5,6,7,8_maxLen=200/e19.ckpt --question=2,3,4,5,6,7,8
# CMCM: CUDA_VISIBLE_DEVICES=7 python test.py --data_source=cite --resume=runs/samples3439_retrieval=1.00_classification=0.10_reweight=1000.00_weightDecay=0.0_withAttention=2_question=2,3,4,5,6,7,8_maxLen=200/e19.ckpt --question=2,3,4,5,6,7,8
# CMCM_Q2: CUDA_VISIBLE_DEVICES=7 python test.py --data_source=cite --resume=runs/samples3439_retrieval=1.00_classification=0.10_reweight=1000.00_weightDecay=0.0_withAttention=2_question=2_maxLen=200/e19.ckpt --question=2
# CMCM_Q3: CUDA_VISIBLE_DEVICES=7 python test.py --data_source=cite --resume=runs/samples3439_retrieval=1.00_classification=0.10_reweight=1000.00_weightDecay=0.0_withAttention=2_question=3_maxLen=200/e19.ckpt --question=3
# CMCM_Q4: CUDA_VISIBLE_DEVICES=7 python test.py --data_source=cite --resume=runs/samples3439_retrieval=1.00_classification=0.10_reweight=1000.00_weightDecay=0.0_withAttention=2_question=4_maxLen=200/e19.ckpt --question=4
# CMCM_Q5: CUDA_VISIBLE_DEVICES=7 python test.py --data_source=cite --resume=runs/samples3439_retrieval=1.00_classification=0.10_reweight=1000.00_weightDecay=0.0_withAttention=2_question=5_maxLen=200/e19.ckpt --question=5
# CMCM_Q6: CUDA_VISIBLE_DEVICES=7 python test.py --data_source=cite --resume=runs/samples3439_retrieval=1.00_classification=0.10_reweight=1000.00_weightDecay=0.0_withAttention=2_question=6_maxLen=200/e19.ckpt --question=6
# CMCM_Q7: CUDA_VISIBLE_DEVICES=7 python test.py --data_source=cite --resume=runs/samples3439_retrieval=1.00_classification=0.10_reweight=1000.00_weightDecay=0.0_withAttention=2_question=7_maxLen=200/e19.ckpt --question=7
# CMCM_Q8: CUDA_VISIBLE_DEVICES=7 python test.py --data_source=cite --resume=runs/samples3439_retrieval=1.00_classification=0.10_reweight=1000.00_weightDecay=0.0_withAttention=2_question=8_maxLen=200/e19.ckpt --question=8

# Clue
# Base: CUDA_VISIBLE_DEVICES=7 python test.py --data_source=clue --resume=runs/samples6047_retrieval=1.00_classification=0.00_reweight=1000.00_weightDecay=0.0_withAttention=0_question=0,1,2,3,4,5_maxLen=40/e19.ckpt --question=0,1,2,3,4,5
# CMCA: CUDA_VISIBLE_DEVICES=7 python test.py --data_source=clue --resume=runs/samples6047_retrieval=1.00_classification=0.00_reweight=1000.00_weightDecay=0.0_withAttention=2_question=0,1,2,3,4,5_maxLen=40/e19.ckpt --question=0,1,2,3,4,5
# CMCM_NoAttn: CUDA_VISIBLE_DEVICES=7 python test.py --data_source=clue --resume=runs/samples6047_retrieval=1.00_classification=0.10_reweight=1000.00_weightDecay=0.0_withAttention=0_question=0,1,2,3,4,5_maxLen=40/e19.ckpt --question=0,1,2,3,4,5
# CMCM: CUDA_VISIBLE_DEVICES=7 python test.py --data_source=clue --resume=runs/samples6047_retrieval=1.00_classification=0.10_reweight=1000.00_weightDecay=0.0_withAttention=2_question=0,1,2,3,4,5_maxLen=40/e19.ckpt --question=0,1,2,3,4,5
# CMCM_Visible: CUDA_VISIBLE_DEVICES=7 python test.py --data_source=clue --resume=runs/samples6047_retrieval=1.00_classification=0.10_reweight=1000.00_weightDecay=0.0_withAttention=2_question=0_maxLen=40/e19.ckpt --question=0
# CMCM_Subjective: CUDA_VISIBLE_DEVICES=7 python test.py --data_source=clue --resume=runs/samples6047_retrieval=1.00_classification=0.10_reweight=1000.00_weightDecay=0.0_withAttention=2_question=1_maxLen=40/e19.ckpt --question=1
# CMCM_Action: CUDA_VISIBLE_DEVICES=7 python test.py --data_source=clue --resume=runs/samples6047_retrieval=1.00_classification=0.10_reweight=1000.00_weightDecay=0.0_withAttention=2_question=2_maxLen=40/e19.ckpt --question=2
# CMCM_Story: CUDA_VISIBLE_DEVICES=7 python test.py --data_source=clue --resume=runs/samples6047_retrieval=1.00_classification=0.10_reweight=1000.00_weightDecay=0.0_withAttention=2_question=3_maxLen=40/e19.ckpt --question=3
# CMCM_Meta: CUDA_VISIBLE_DEVICES=7 python test.py --data_source=clue --resume=runs/samples6047_retrieval=1.00_classification=0.10_reweight=1000.00_weightDecay=0.0_withAttention=2_question=4_maxLen=40/e19.ckpt --question=4
# CMCM_Irrelevant: CUDA_VISIBLE_DEVICES=7 python test.py --data_source=clue --resume=runs/samples6047_retrieval=1.00_classification=0.10_reweight=1000.00_weightDecay=0.0_withAttention=2_question=5_maxLen=40/e19.ckpt --question=5

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import pdb
import pprint
import json
from types import SimpleNamespace
from matplotlib import pyplot as plt
from torchnet import meter
import pdb

import sys
sys.path.append('../')
from LSTM_Resnet50_retrieval_model.datasets import CoherenceDataset, val_transform
from LSTM_Resnet50_retrieval_model.networks import TextEncoder, ImageEncoder, DiscourseClassifier
from LSTM_Resnet50_retrieval_model import utils

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

# draw attention if possible:
def save_attention_result(index, dataset, i2w, ranks, capt_attn, save_dir):
    fig = plt.figure(figsize=(12,8))
    txt, txtlen, img, dis_vec = dataset[index]
    capt_alpha = capt_attn[index]

    # caption
    one_vector = txt
    one_alpha = capt_alpha
    one_word_list = [i2w[idx] for idx in one_vector[:txtlen]]
    capt_disp = ' '.join(one_word_list[:10])
    ind = np.arange(txtlen)
    plt.subplot(121)
    plt.barh(ind, one_alpha[:txtlen])
    plt.yticks(ind, one_word_list)

    # images
    plt.subplot(122)
    one_img = img.permute(1,2,0).detach().cpu().numpy()
    scale = one_img.max() - one_img.min()
    one_img = (one_img - one_img.min()) / scale
    plt.imshow(one_img)
    plt.axis('off')
    try:
        plt.savefig(os.path.join(save_dir, 'id={}_rank={}_{}.jpg'.format(index, ranks[index], capt_disp)))
    except:
        print('Exception happens')
        pass
    plt.close(fig)
    

def find_args(resume):
    prefix = resume.rsplit('/', 1)[0]
    args_file = os.path.join(prefix, 'args.json')
    assert os.path.exists(args_file), 'args.json is not found'
    print('encoder args file:', args_file)
    with open(args_file, 'r') as f:
        ckpt_args = json.load(f)
    ckpt_args = SimpleNamespace(**ckpt_args)
    return ckpt_args

##############################
# setup
##############################
import argparse
parser = argparse.ArgumentParser(description='Validation parameters')
parser.add_argument('--data_source', default='cite', type=str, choices=['cite', 'clue'])
parser.add_argument('--question', default='2', type=str)
parser.add_argument('--seed', default=2, type=int)
parser.add_argument("--device", default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--resume', default='')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device('cuda')

# #############################
# model
# #############################
ckpt_args = find_args(args.resume)
ckpt_args.data_source = args.data_source
ckpt_args.max_len = 200 if ckpt_args.data_source=='cite' else 40
ckpt_args.word2vec_file=f'models/word2vec_{ckpt_args.data_source}.bin'

if args.data_source == 'clue':
    relations = ['Visible', 'Subjective', 'Action', 'Story', 'Meta', 'Irrelevant']
    betas = [0.12] # clue
else:
    relations = ['q2_resp', 'q3_resp', 'q4_resp', 'q5_resp', 'q6_resp', 'q7_resp', 'q8_resp']
    betas = [0.13] # cite

text_encoder = TextEncoder(
    emb_dim=ckpt_args.word2vec_dim, 
    hid_dim=ckpt_args.rnn_hid_dim, 
    z_dim=ckpt_args.feature_dim, 
    word2vec_file=ckpt_args.word2vec_file,
    with_attention=ckpt_args.with_attention).to(device)
image_encoder = ImageEncoder(
    z_dim=ckpt_args.feature_dim).to(device)
discourse_classifier = DiscourseClassifier(
    len(relations), ckpt_args.feature_dim).to(device)
ckpt = torch.load(args.resume)
print(ckpt.keys())
text_encoder.load_state_dict(ckpt['text_encoder'])
image_encoder.load_state_dict(ckpt['image_encoder'])
discourse_classifier.load_state_dict(ckpt['discourse_class'])


print(f'Use question(s): {args.question}')
if args.data_source == 'cite':
    valid_questions = torch.tensor([
            int(x)-2 for x in args.question.split(',')],
            dtype=torch.long).to(device)
else:
    valid_questions = torch.tensor([
            int(x) for x in args.question.split(',')],
            dtype=torch.long).to(device)

pp = pprint.PrettyPrinter(indent=2)

##############################
# dataset
##############################
test_set = CoherenceDataset(
        part='test', 
        datasource=ckpt_args.data_source,
        max_len=ckpt_args.max_len,
        word2vec_file=ckpt_args.word2vec_file, 
        transform=val_transform)

test_loader = DataLoader(
        test_set, batch_size=ckpt_args.batch_size, shuffle=False, 
        num_workers=ckpt_args.workers, pin_memory=True, 
        drop_last=False)

print('test data:', len(test_set), len(test_loader))

##############################
# test
##############################
print('==> test')
text_encoder.eval()
image_encoder.eval()
discourse_classifier.eval()
requires_grad(text_encoder, False)
requires_grad(text_encoder, False)
requires_grad(discourse_classifier, False)

feat_dir = f'feats/'
os.makedirs(feat_dir, exist_ok=True)
feat_filename = args.resume.split('/')[-2]
feats_path = os.path.join(feat_dir, f'{feat_filename}.pt')
if not os.path.exists(feats_path):
    txt_feats = []
    txt_attns = []
    img_feats = []
    probs = []
    labels = []
    for batch in tqdm(test_loader):
        for i in range(len(batch)):
            batch[i] = batch[i].to(device)
        txt, txt_len, img, target = batch
        txt_feat, txt_attn = text_encoder(txt.long(), txt_len)
        img_feat = image_encoder(img)
        prob = torch.sigmoid(discourse_classifier(txt_feat, img_feat))[:, valid_questions]
        txt_feats.append(txt_feat.detach().cpu())
        if ckpt_args.with_attention:
            txt_attns.append(txt_attn.detach().cpu())
        img_feats.append(img_feat.detach().cpu())
        probs.append(prob.detach().cpu())
        labels.append(target)

    txt_feats = torch.cat(txt_feats, dim=0)
    if ckpt_args.with_attention:
        txt_attns = torch.cat(txt_attns, dim=0).numpy()
    img_feats = torch.cat(img_feats, dim=0)
    # probs = torch.cat(probs, dim=0).cpu().numpy() # [N, num_classes]

    feats = {}
    feats['txt_feats'] = txt_feats
    feats['img_feats'] = img_feats
    feats['probs'] = probs
    feats['labels'] = labels
    torch.save(feats, feats_path)
else:
    print(f'load features from {feats_path}')
    feats = torch.load(feats_path)
    txt_feats = feats['txt_feats']
    img_feats = feats['img_feats']
    probs = feats['probs']
    labels = feats['labels']

N = len(txt_feats)
img_feats = img_feats.to(device)


##############################
# Test Confidence Score
##############################
# betas = np.linspace(0.01, 0.2, 40)
diffs = []
for beta in betas:
    confidences = []
    print(f'\n*********** beta={beta:.4f} ************')
    for i in range(N):
        repeated_txt_feats = txt_feats[i:i+1].repeat(N,1).to(device) # [N, feat_dim]
        prob = torch.sigmoid(discourse_classifier(repeated_txt_feats, img_feats))[:, valid_questions]
        prob = prob.detach().cpu().numpy() # [N, valid_questions]

        # 1. self-defined confidence score
        confidence = np.exp(beta * abs(prob-0.5)).sum(axis=1)

        # # 2. use inverse entropy as confidence score
        # entropy = - (prob * np.log(prob) + (1-prob) * np.log(1-prob)) # [N, valid_questions]
        # confidence = (1.0 / entropy).sum(axis=1) # [N]

        confidences.append(confidence)
    
    confidences = np.stack(confidences, axis=0) # [N, N]
    print('confidence scores shape:', confidences.shape)

    retrieved_range = min(txt_feats.shape[0], 500)
    print('retrieved_range =', retrieved_range)
    medR_results = []
    for metric in ['confidence', 'sim', 'sim*confidence']:
        medRs, recalls = utils.rank(
            txt_feats.cpu().numpy(), 
            img_feats.cpu().numpy(), 
            confidences=confidences,
            metric=metric,
            retrieved_type=ckpt_args.retrieved_type, 
            retrieved_range=retrieved_range)
        medR_results.append(medRs.mean())
        print(metric)
        print('[' + ', '.join([f'{x:4.1f}' for x in medRs]) + '],')
        print()
    diffs.append(medR_results[2]-medR_results[1])

# plt.plot(betas, diffs)
# plt.savefig('find_best_temperature.jpg')

print(f'\nShow Average Precisions')
labels = torch.cat(labels, dim=0).cpu().numpy() # [N, num_classes]
# mtr = meter.APMeter()
# mtr.add(probs, labels)
# APs = mtr.value()
# for relation, ap in zip(relations, APs):
#     print(f'{relation:>10s} = {ap:<.4f}')

# T = 0.5
# print(f'\nShow F1 (threshold={T:.2f})')
# preds = probs>T
# num_pred = (preds>0).sum(1)
# num_true = (labels>0).sum(1)
# print('num_pred = {:.2f}'.format(num_pred.mean()))
# print('num_true = {:.2f}'.format(num_true.mean()))


# # retrieval scores on each question
# positive_cnt = labels.sum(axis=0) # [valid_questions]
# negative_cnt = labels.shape[0] - positive_cnt
# # for fairness, we need to make sure all questions are using the same retrieval range
# maximum_retrieval_range = int(min(positive_cnt.min(), negative_cnt.min()))
# print('positive count', positive_cnt)
# print('negative count', negative_cnt)
# print(f'maximum_retrieval_range = {maximum_retrieval_range}')

# # now compute MedR and Recalls for each subset (i.e. each question with only positive or only negative)
# for i in range(labels.shape[1]):
#     print(f'\n==> Results for Q_{i}')
#     print('positive subset')
#     txt_sub = txt_feats[labels[:,i]==1]
#     img_sub = img_feats[labels[:,i]==1]
#     medR, medR_std, recalls = utils.rank(txt_sub, img_sub, retrieved_type='image', retrieved_range=maximum_retrieval_range)
#     print('negative subset')
#     txt_sub = txt_feats[labels[:,i]==0]
#     img_sub = img_feats[labels[:,i]==0]
#     medR, medR_std, recalls = utils.rank(txt_sub, img_sub, retrieved_type='image', retrieved_range=maximum_retrieval_range)


# Compare MedR and Recalls for each question over the entire test set
ids_range = np.random.choice(txt_feats.shape[0], retrieved_range)
labels_range = labels[ids_range]
txt_range = txt_feats[ids_range].cpu().numpy()
img_range = img_feats[ids_range].cpu().numpy()
CIs_range = confidences[ids_range, ids_range]
sim_mat = utils.compute_sim_mat(txt_range, img_range, CIs_range)
positions = utils.get_positions(sim_mat)
# pdb.set_trace()
for i in range(labels_range.shape[1]):
    q_id = i+2 if args.data_source=='cite' else i
    print(f'\n==> Results for Q_{q_id}')
    ranks_sub = positions[labels_range[:,i]==1]
    N = len(ranks_sub)
    print(f'positive subset({N}/{retrieved_range}): MedR={int(np.median(ranks_sub)):>3d}, Recall@1={(ranks_sub==1).sum()/N:.2f}, Recall@5={(ranks_sub<=5).sum()/N:.2f}, Recall@10={(ranks_sub<=10).sum()/N:.2f}')
    ranks_sub = positions[labels_range[:,i]==0]
    N = len(ranks_sub)
    print(f'negative subset({N}/{retrieved_range}): MedR={int(np.median(ranks_sub)):>3d}, Recall@1={(ranks_sub==1).sum()/N:.2f}, Recall@5={(ranks_sub<=5).sum()/N:.2f}, Recall@10={(ranks_sub<=10).sum()/N:.2f}')



# # draw attentions 
# sims = np.dot(txt_feats, img_feats.T)
# ranks = []
# recipes = test_set.recipes
# ranks = []
# # loop through the N similarities for images
# for ii in range(sims.shape[0]):
#     # get a column of similarities for image ii
#     sim = sims[ii,:]
#     # sort indices in descending order
#     sorting = np.argsort(sim)[::-1].tolist()
#     # find where the index of the pair sample ended up in the sorting
#     pos = sorting.index(ii)
#     ranks.append(pos)

# ranks = np.array(ranks)
# np.save('{}/ranks.npy'.format(save_dir), ranks)

# if ckpt_args.with_attention:
#     print('save attention information')
#     i2w = {i:w for w,i in test_set.w2i.items()}
#     attn_dir = os.path.join(save_dir, 'attns')
#     os.makedirs(attn_dir, exist_ok=True)
#     for i in tqdm(range(len(test_set))[:100]):
#         save_attention_result(i, test_set, i2w, ranks, txt_attns, attn_dir)
