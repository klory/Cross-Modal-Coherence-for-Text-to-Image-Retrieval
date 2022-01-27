from utils import load_recipes, clean_caption
from tqdm import tqdm
import os
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import re
import pdb
import numpy as np
from matplotlib import pyplot as plt
from args import get_args


def train_w2v():
    print('Load documents...')
    args = get_args()
    if args.data_source == 'cite':
        capheader = 'stim_txt'
        recipe_file = './../data/RecipeQA/q2-8_train_dis_11-08.csv'
        relations = ['q2_resp', 'q3_resp', 'q4_resp', 'q5_resp',
                     'q6_resp', 'q7_resp', 'q8_resp']
    else:
        capheader = 'caption'
        recipe_file = './../data/conceptual/conceptual_train_dis.csv'
        relations = ['Visible', 'Subjective', 'Action', 'Story',
                     'Meta', 'Irrelevant']
    recipes, n2p = load_recipes(recipe_file, relations)
    print(f'number of recipes = {len(recipes)}')
    print('Tokenize...')
    all_sentences = []
    for i in tqdm(range(len(recipes))):
        cap = recipes[capheader].iloc[i]
        cap = clean_caption(cap)
        all_sentences.append(re.split(r'\\n| ', cap))
    print('number of sentences =', len(all_sentences))
    lens = np.array([len(x) for x in all_sentences])
    plt.hist(lens, bins=100)
    plt.show()
    print('sentence lengths: {:.2f} ({:.2f})'.format(lens.mean(), lens.std()))
    print('Train Word2Vec model...')

    class EpochLogger(CallbackAny2Vec):
        '''Callback to log information about training'''
        def __init__(self):
            self.epoch = 0

        def on_epoch_begin(self, model):
            print('-' * 40)
            print("Epoch #{} start".format(self.epoch))
            print('vocab_size = {}'.format(len(model.wv.index_to_key)))

        def on_epoch_end(self, model):
            print('total_train_time = {:.2f} s'.format(model.total_train_time))
            print('loss = {:.2f}'.format(model.get_latest_training_loss()))
            print("Epoch #{} end".format(self.epoch))
            self.epoch += 1

    epoch_logger = EpochLogger()
    model = Word2Vec(
        all_sentences, vector_size=300, window=10, min_count=1,
        workers=20, epochs=50, callbacks=[epoch_logger],
        compute_loss=True)

    if not os.path.exists('models'):
        os.makedirs('models')

    model.wv.save(os.path.join(f'./models/word2vec_{args.data_source}.bin'))


if __name__ == '__main__':
    train_w2v()
