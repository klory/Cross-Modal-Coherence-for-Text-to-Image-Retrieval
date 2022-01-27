# Cross-Modal Coherence for Text-to-Image Retrieval -- Official PyTorch Implementation
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![TensorFlow 1.7](https://img.shields.io/badge/pytorch-1.7-green.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-CC_BY--NC-green.svg?style=plastic)

This code is tested on Ubuntu 20.04 with Pytorch 1.7.1 (Python 3.8) and currently tested with the LSTM-Resnet50 retrieval model.

> **Cross-Modal Coherence for Text-to-Image Retrieval**<br>
> Malihe Alikhani *1, Fangda Han *2, Hareesh Ravi *2, Mubbasir Kapadia 2 , Vladimir Pavlovic 2<br>
> *: Equal contribution

> 1. University of Pittsburgh<br>
> 2. Rutgers University<br>

> https://arxiv.org/abs/2109.11047
>
> **Abstract:** *Common image-text joint understanding techniques presume that images and the associated text can universally be characterized by a single implicit model. However, co-occurring images and text can be related in qualitatively different ways, and explicitly modeling it could improve the performance of current joint understanding models. In this paper, we train a Cross-Modal Coherence Modelfor text-to-image retrieval task. Our analysis shows that models trained with image--text coherence relations can retrieve images originally paired with target text more often than coherence-agnostic models. We also show via human evaluation that images retrieved by the proposed coherence-aware model are preferred over a coherence-agnostic baseline by a huge margin. Our findings provide insights into the ways that different modalities communicate and the role of coherence relations in capturing commonsense inferences in text and imagery.*

Material related to our paper is available via the following links:

- Paper: https://arxiv.org/abs/2109.11047
- Video: TBD
- Code: https://github.com/klory/Cross-Modal-Coherence-for-Text-to-Image-Retrieval
- CITE++: https://drive.google.com/drive/folders/1iCecxh3Np8Yu8ATAf5uCiaOEah-4Pb-M?usp=sharing
- Clue: https://drive.google.com/drive/folders/10fLm0mb2l8vUj_FFP1J5gygD91cDa9nN?usp=sharing

# Dataset

Download datasets from links above.

## CITE++ Dataset
* `data/RecipeQA/q2-8_train_dis_11-08.csv`
* `data/RecipeQA/q2-8_test_dis_11-08.csv`
* `data/RecipeQA/images/`

## Clue Dataset
* `data/conceptual/conceptual_train_dis.csv`
* `data/conceptual/conceptual_test_dis.csv`
* `data/conceptual/img2idxmap.json`
* `data/conceptual/images/`

# Setup Virtual Environment
Create new environment using conda
```bash
conda create -n discourse python=3.8
source activate discourse
pip install -r requirements.txt
```

# Train word2vec model

cd to `LSTM_Resnet50_retrieval_model/`, and run the following command:

## CITE++ dataset
```bash
python train_word2vec.py --data_source='cite'
```

## Clue dataset
```bash
python train_word2vec.py --data_source='clue'
```

Please check `train_word2vec.py` for hyperparameters. The generated models are saved as `models/word2vec_cite.bin` and `models/word2vec_clue.bin`.

# Train retrieval model

cd to `LSTM_Resnet50_retrieval_model/`, and run the following command:

## CITE++
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --data_source='cite'
```

## Clue
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --data_source='clue'
```

# Test models with confidence score
See top lines in `test.py` for how to test models with confidence score.

You can remove WandB monitoring by set `--wandb=0`.

