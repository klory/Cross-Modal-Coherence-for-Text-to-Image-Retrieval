import json
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import rnn
from torchvision import models
import torch.utils.model_zoo as model_zoo
from gensim.models.keyedvectors import KeyedVectors
import pdb
import torchvision
import math

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, with_attention):
        super(AttentionLayer, self).__init__()
        self.u = torch.nn.Parameter(torch.randn(input_dim)) # u = [2*hid_dim]
        self.u.requires_grad = True
        self.fc = nn.Linear(input_dim, input_dim)
        self.with_attention = with_attention
    def forward(self, x):
        # x = [BS, max_len, 2*hid_dim]
        # a trick used to find the mask for the softmax
        mask = (x!=0)
        mask = mask[:,:,0].bool()
        h = torch.tanh(self.fc(x)) # h = [BS, max_len, 2*hid_dim]
        if self.with_attention == 1: # softmax
            scores = h @ self.u # scores = [BS, max_len], unnormalized importance
        elif self.with_attention == 2: # Transformer
            scores = h @ self.u / math.sqrt(h.shape[-1]) # scores = [BS, max_len], unnormalized importance
        masked_scores = scores.masked_fill(~mask, -1e32)
        alpha = F.softmax(masked_scores, dim=1) # alpha = [BS, max_len], normalized importance

        alpha = alpha.unsqueeze(-1) # alpha = [BS, max_len, 1]
        out = x * alpha # out = [BS, max_len, 2*hid_dim]
        out = out.sum(dim=1) # out = [BS, 2*hid_dim]
        # pdb.set_trace()
        return out, alpha.squeeze(-1)

class SentenceEncoder(nn.Module):
    def __init__(
        self, 
        emb_dim, 
        hid_dim, 
        z_dim, 
        max_len=200,
        word2vec_file='./models/word2vec.bin',
        with_attention=False):
        
        super(SentenceEncoder, self).__init__()
        wv = KeyedVectors.load(word2vec_file, mmap='r')
        vec = torch.from_numpy(wv.vectors).float()
        # first two index has special meaning, see load_dict() in utils.py
        emb = nn.Embedding(vec.shape[0]+2, vec.shape[1], padding_idx=0)
        emb.weight.data[2:].copy_(vec)
        self.embed_layer = emb
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            bidirectional=True,
            batch_first=True)

        self.with_attention = with_attention
        self.max_len = max_len
        if with_attention:
            self.atten_layer = AttentionLayer(2*hid_dim, with_attention)
    
    def forward(self, sent_list, lengths):
        # sent_list [BS, max_len]
        # lengths [BS]
        x = self.embed_layer(sent_list) # x=[BS, max_len, emb_dim]
        sorted_len, sorted_idx = lengths.sort(0, descending=True) # sorted_idx=[BS], for sorting
        _, original_idx = sorted_idx.sort(0, descending=False) # original_idx=[BS], for unsorting
        index_sorted_idx = sorted_idx.view(-1,1,1).expand_as(x) # sorted_idx=[BS, max_len, emb_dim]
        sorted_inputs = x.gather(0, index_sorted_idx.long()) # sort by num_words
        packed_seq = rnn.pack_padded_sequence(
            sorted_inputs, sorted_len.cpu().numpy(), batch_first=True)
        self.rnn.flatten_parameters()
        if self.with_attention:
            out, _ = self.rnn(packed_seq)
            # pdb.set_trace()
            y, _ = rnn.pad_packed_sequence(
                out, batch_first=True, total_length=self.max_len) # y=[BS, max_len, 2*hid_dim], currently in WRONG order!
            unsorted_idx = original_idx.view(-1,1,1).expand_as(y)
            output = y.gather(0, unsorted_idx).contiguous() # [BS, max_len, 2*hid_dim], now in correct order
            feat, alpha = self.atten_layer(output) # [BS, 2*hid_dim]
            # print('sent', feat.shape) # [BS, 2*hid_dim]
            return feat, alpha
        else:
            _, (h,_) = self.rnn(packed_seq) # [2, BS, hid_dim], currently in WRONG order!
            h = h.transpose(0,1) # [BS, 2, hid_dim], still in WRONG order!
            # unsort the output
            unsorted_idx = original_idx.view(-1,1,1).expand_as(h)
            output = h.gather(0, unsorted_idx).contiguous() # [BS, 2, hid_dim], now in correct order
            feat = output.view(output.size(0), output.size(1)*output.size(2)) # [BS, 2*hid_dim]
            # pdb.set_trace()
            return feat, None


class TextEncoder(nn.Module):
    """Encode text into latent space
    """
    def __init__(self, emb_dim, hid_dim, z_dim, max_len=200, word2vec_file=None, with_attention=False):
        super(TextEncoder, self).__init__()
        self.with_attention=with_attention
        self.sent_encoder = SentenceEncoder(
            emb_dim=emb_dim, 
            hid_dim=hid_dim, 
            z_dim=z_dim, 
            max_len=max_len,
            word2vec_file=word2vec_file,
            with_attention=with_attention)
        self.bn = nn.BatchNorm1d(2*hid_dim)
        self.fc = nn.Linear(2*hid_dim, z_dim)
    
    def forward(self, ingrs, n_ingrs):
        # ingrs [BS, max_len]
        # n_ingrs [BS]
        feat, attn = self.sent_encoder(ingrs, n_ingrs) # [BS, 2*hid_dim]
        feat = self.fc(self.bn(feat))
        feat = F.normalize(feat, p=2, dim=1)
        return feat, attn


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        num_feat = resnet.fc.in_features
        resnet.fc = nn.Linear(num_feat, 101)
        modules = list(resnet.children())[:-1]  # we do not use the last fc layer.
        self.encoder = nn.Sequential(*modules)
    
    def forward(self, image_list):
        BS = image_list.shape[0]
        return self.encoder(image_list).view(BS, -1)

class ImageEncoder(nn.Module):
    def __init__(self, z_dim):
        super(ImageEncoder, self).__init__()
        self.resnet = Resnet()
        self.bottleneck = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Linear(2048, z_dim),
        )
    
    def forward(self, image_list):
        feat = self.resnet(image_list)
        feat = self.bottleneck(feat)
        # print('image', feat.shape)
        return F.normalize(feat, p=2, dim=1)

class DiscourseClassifier(nn.Module):
    def __init__(self, num_discourse, z_dim):
        super(DiscourseClassifier, self).__init__()
        self.fc = nn.Linear(z_dim*2, num_discourse)
        
    def forward(self, textfeat, imagefeat):
        x = torch.cat((textfeat, imagefeat), 1)
        out = self.fc(x)
        return out