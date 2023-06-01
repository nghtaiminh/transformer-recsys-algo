import torch
import numpy as np
import pickle
from model import SASRec
import argparse

with open('data/itemmap.pkl', 'rb') as f:
    itemmap = pickle.load(f)

with open('data/usermap.pkl', 'rb') as f:
    usermap = pickle.load(f)


def recommend(model, user_interacted_item_seq, args, candidate_indices=None, k=12, exclude_interacted=True):
    """Recommend based on a sequence of interacted items

    Args:
        user_interacted_item_seq (list): a sequence of interacted items
        candidate_indices (list, optional): a list of candidate items. Defaults to None.  If None, all items will be used as candidates.
        k (int, optional): number of items to recommend. Defaults to 12.
        exclude_interacted (bool, optional): whether to exclude interacted items from the recommendation. Defaults to True.
    Returns:
        recommendations: a list of recommended items, ranked from most relevant to least relevant
    """

    # map the user's interacted items to their indices
    user_interacted_item_seq = [itemmap[x]
                                for x in user_interacted_item_seq]
    # initialize the sequence with zeros
    seq = np.zeros([args.maxlen], dtype=np.int32)
    idx = args.maxlen - 1
    # fill the sequence with the user's interacted items
    for i in reversed(user_interacted_item_seq):
        seq[idx] = i
        idx -= 1
        if idx == -1:
            break

    # if candidate are provided, map them to their indices, otherwise use all items
    if candidate_indices:
        item_idx = [itemmap[x] for x in candidate_indices]
    else:
        item_idx = [x for x in range(1, len(itemmap)+1)]
    # if exclude interacted items, remove them from the candidate list
    if exclude_interacted:
        item_idx = [x for x in item_idx if x not in user_interacted_item_seq]

    u = 1  # dummy user id
    predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
    # predict the revelance score
    predictions = predictions[0]  # - for 1st argsort DESC
    # rank the items
    rank = predictions.argsort()[:k].tolist()
    # map the indices to their item ids
    def get_key(val):
        for key, value in itemmap.items():
            if val == value:
                return key

    recommendations = [get_key(item) for item in rank]

    return recommendations


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AmazonBooks_new', type=str)
parser.add_argument('--train_dir', default='default', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=100, type=int)
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), type=str)
parser.add_argument('--inference_only', default=True, type=str2bool)
parser.add_argument('--state_dict_path',
                    default='SASRec.epoch=100.lr=0.001.layer=2.head=1.hidden=64.maxlen=100.units=64.dropt=0.4.blocks=2.pth', type=str)


args = parser.parse_args(args=[])
model = SASRec(len(usermap), len(itemmap), args)
model.eval()

input_ids = ['1511688335', '0743271106', '1519177054']

print(recommend(model, input_ids, args, k=12))

