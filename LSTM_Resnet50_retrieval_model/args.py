import argparse

def get_args():
    parser = argparse.ArgumentParser(description='retrieval model parameters')
    parser.add_argument('--seed', default=8, type=int)
    parser.add_argument("--device", default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--data_source', default='cite', type=str, choices=['cite', 'clue'])
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--ckpt_path', default='')
    parser.add_argument('--wandb', type=int, default=1, choices=[0, 1])


    parser.add_argument('--word2vec_dim', default=300, type=int)
    parser.add_argument('--rnn_hid_dim', default=300, type=int)
    parser.add_argument('--feature_dim', default=1024, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--margin', default=0.3, type=float)
    parser.add_argument('--retrieved_type', default='image', choices=['recipe', 'image'])
    parser.add_argument('--retrieved_range', default=500, type=int)
    
    parser.add_argument('--num_batches', default=10000, type=int)
    parser.add_argument('--val_freq', default=100, type=int)
    parser.add_argument('--save_freq', default=500, type=int)


    parser.add_argument('--weight_retrieval', default=1.0, type=float)
    parser.add_argument('--weight_classification', default=0.1, type=float)

    parser.add_argument('--reweight', default=1, choices=[0, 1], help='Use this flag to reweight the pos_weight of the classification loss when the pos_weight is too small/large')
    parser.add_argument('--reweight_limit', default=1000, type=float)

    parser.add_argument('--weight_decay', default=0.0, type=float, help='L2 normalization')
    parser.add_argument("--with_attention", type=int, default=2, choices=[0, 1, 2],
                        help='0: no attention, 1: vannila attention, 2: Transformer-like attention')


    # in debug mode
    parser.add_argument("--debug", type=int, default=0,
                        help="in debug mode or not")

    # ignore for now
    parser.add_argument('--dataset_q', default=0, type=int,
                        help='ignore it for now')

    args = parser.parse_args()
    return args
