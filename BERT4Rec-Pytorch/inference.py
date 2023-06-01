import torch
import numpy as np
from models import model_factory
from models.bert import BERTModel
import pickle

# open itemmap
with open('saved_mapping/bert_itemmap.pkl', 'rb') as f:
    itemmap = pickle.load(f)

def recommend(model, input_ids, args, candidate_items=None, k=10, exclude_interacted=True):
    """
    
    Args:
        model: BERT model
        input_ids (list): 
            List of item ids sorted by timestamp in ascending order [oldest, ..., newest]
        args: 
            Model parameters.
        candidate_items: (list, optional): 
            List of candidate item ids.
        k (int, optional): Number of recommendations.
        exclude_interacted: bool, (optional)
            Whether to exclude interacted items.
    Returns:
        recommendations (list): List of recommended item ids. Ranked from most relevant to least relevant.
    """

    # map input_ids to item ids
    input_ids = [itemmap[i] for i in input_ids]

    # Prepare inputs
    input_ids = input_ids + [args.num_items + 1] # Add mask token at the end of the input
    input_ids = input_ids[-args.bert_max_len:] # Truncate input to max length
    padding_len = args.bert_max_len - len(input_ids) # Calculate padding length
    input_ids = [0] * padding_len + input_ids # 0 padding if input is shorter than max length
    input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)

    # Scoring
    with torch.no_grad():
        scores = model(input_ids)
        print("Scores: ", scores)
    scores = scores[:, -1, :]

    # exclude interacted items
    if exclude_interacted:
        if candidate_items:
            candidates = [x for x in range(1, len(itemmap)+1)
                          if x not in input_ids and x in candidate_items]
        else:
            candidates = [x for x in range(1, len(itemmap)+1) if x not in input_ids]
        
        candidate = torch.LongTensor(candidates).unsqueeze(0).to(scores.device)
        scores = scores.gather(1, candidate)

    print("Scores: ", scores)
    # Rank 
    # rank = (-scores).argsort(dim=1)
    topk = torch.topk(scores, k).indices.squeeze(0).tolist()
    print("Topk: ", topk)

    def get_key(val):
        for key, value in itemmap.items():
            if val == value:
                return key
    recommendations = [get_key(i) for i in topk]
    print("Recommendations: ", recommendations)
    return recommendations



# Define model parameters

class Args:
    def __init__(self):
        self.bert_dropout = 0.4
        self.num_items = 51022 # len(itemmap)
        self.bert_hidden_units = 256
        self.bert_mask_prob = 0.5
        self.bert_max_len = 100
        self.bert_num_blocks = 2
        self.bert_num_heads = 4
        self.model_init_seed = 0


args = Args()
model = BERTModel(args)

# Load pretrained weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = 'experiments/test_2023-03-11_0/models/best_acc_model.pth'
model_state_dict = torch.load(path, map_location =device)['model_state_dict']
model.load_state_dict(model_state_dict)
model.eval()

input_ids = ['1511688335', '0743271106', '1519177054']

# Get recommendations
recommend(model, input_ids, args, k=10)