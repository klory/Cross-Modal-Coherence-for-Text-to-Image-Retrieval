import json, pickle
import os
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torch.utils import data
from gensim.models.keyedvectors import KeyedVectors
from PIL import Image
import pdb
import json
from io import BytesIO
import requests
import urllib

import sys
sys.path.append('../')
from LSTM_Resnet50_retrieval_model.utils import load_recipes, get_caption_wordvec, get_discourse_vec


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    normalize
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

def default_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        return im
    except:
        # print('error to open image:', path)
        return Image.new('RGB', (224, 224), 'white')

class CoherenceDataset(data.Dataset):
    def __init__(self, part, datasource, word2vec_file, max_len=200,
                 dataset_q=0, transform=None):
        assert part in ('train', 'test'), 'part must be one in [train, test]'
        if datasource == 'cite':
            self.imgheader = 'qimg'
            self.capheader = 'stim_txt'
            recipe_file = './../data/RecipeQA/q2-8_{}_dis_11-08.csv'.format(
                    part)
            self.relations = ['q2_resp', 'q3_resp', 'q4_resp', 'q5_resp',
                              'q6_resp', 'q7_resp', 'q8_resp']
            self.img_dir = f'../data/RecipeQA/images-qa/{part}/images-qa/'
            self.img_dir = f'../data/RecipeQA/images-qa/train/images-qa/'
        else:
            self.imgheader = 'url'
            self.capheader = 'caption'
            recipe_file = './../data/conceptual/conceptual_{}_dis.csv'.format(part)
            # recipe_file = './../data/conceptual/conceptual_train_dis.csv'.format(part)
            self.relations = ['Visible', 'Subjective', 'Action', 'Story',
                              'Meta', 'Irrelevant']
            self.img2id = json.load(open(
                    './../data/conceptual/img2idxmap.json', 'r'))
            self.img_dir = f'../data/conceptual/images/'
            
        # pdb.set_trace()
        wv = KeyedVectors.load(word2vec_file, mmap='r')
        w2i = {w: i+2 for i, w in enumerate(wv.index_to_key)}
        w2i['<other>'] = 1
        self.w2i = w2i
        print('vocab size =', len(self.w2i))

        self.transform = transform
        self.max_len = max_len
        self.recipes, self.n2p = load_recipes(recipe_file,
                                              self.relations,
                                              dataset_q=dataset_q)
        self.data_source = datasource

    def _prepare_one_recipe(self, index):
        rcp = self.recipes.iloc[index]
        # read and process the caption
        caption, n_words_in_caption = get_caption_wordvec(
                rcp[self.capheader], self.w2i, max_len=self.max_len)  # np.int [max_len]
        # read and process the image
        if self.data_source == 'clue':
            imgpath = self.img2id[rcp[self.imgheader]]
            pil_img = default_loader(
                    './../data/conceptual/images/{}.jpg'.format(imgpath))
#            with urllib.request.urlopen(rcp[self.imgheader]) as url:
#                f = BytesIO(url.read())
#
#            pil_img = Image.open(f).convert('RGB')
#            pil_img = Image.open(BytesIO(requests.get(
#                        rcp[self.imgheader]).content)).convert('RGB')
        else:
            img_path = os.path.join(self.img_dir, rcp[self.imgheader])
            pil_img = default_loader(img_path)
        img = self.transform(pil_img)
        # read and process the discourse_vec
        dis_vec = torch.FloatTensor(get_discourse_vec(rcp, self.relations))
        return caption, n_words_in_caption, img, dis_vec

    def __getitem__(self, index):
        txt, txtlen, img, dis_vec = self._prepare_one_recipe(index)
        return txt, txtlen, img, dis_vec

    def __len__(self):
        return len(self.recipes)


if __name__ == '__main__':
    from tqdm import tqdm
    print('first line of script')
    datasource = 'cite'
    datasource = 'clue'
    part = 'test'
    
    train_set = CoherenceDataset(
        part=part,
        datasource=datasource,
        word2vec_file=f'models/word2vec_{datasource}.bin',
        transform=train_transform)

    test_set = CoherenceDataset(
        part=part,
        datasource=datasource,
        word2vec_file=f'models/word2vec_{datasource}.bin',
        transform=val_transform)

    print(len(train_set), len(test_set))

    loader = torch.utils.data.DataLoader(test_set, batch_size=64, num_workers=8)
    for batch in tqdm(loader):
        txt, n_words_in_txt, img, dis_vec = batch
        print(txt.shape, n_words_in_txt.shape, img.shape, dis_vec.shape)
