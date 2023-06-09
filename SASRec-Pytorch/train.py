import os
import numpy as np
SEED = 12345678
np.random.seed(SEED)
import time
import torch
import argparse
import sys
import pickle
from model import SASRec
from utils import *
from torch.utils.tensorboard import SummaryWriter


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Books', type=str, help='The preprocess data file name, e.g. the Books.txt file will be Books')
parser.add_argument('--train_dir', default='default', type=str, help='The directory to save the trained model. The directory will be named as: dataset_train_dir')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
parser.add_argument('--maxlen', default=100, type=int, help='Maximum length for the interaction sequence')
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--num_blocks', default=2, type=int, help='Number of self-attention blocks in the model')
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--num_heads', default=1, type=int, help='Number of heads in the self-attention layer')
parser.add_argument('--dropout_rate', default=0.4, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--tensorboard_log_dir', default="final_run", type=str)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

writer = SummaryWriter(args.tensorboard_log_dir)

# num_early_stop = 20
# count_early_stop = 0

if __name__ == '__main__':

    # global dataset
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('Average Sequence Length: %.2f' % (cc / len(user_train)))
    
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log' + '_units=' + str(args.hidden_units) + '_dropout=' + str(args.dropout_rate) + '_blocks=' + str(args.num_blocks) + '.txt'), 'w')
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=4)
    model = SASRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    
    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    
    model.train() # enable model training
    
    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
    
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('Test (NDCG@10: %.4f, HR@10: %.4f, NDCG@20: %.4f, HR@20: %.4f)' % (t_test[0], t_test[1], t_test[2], t_test[3]))
    
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...

    lr_step = 15
    lr_decay = 1.0
    lr = args.lr
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(adam_optimizer, step_size=lr_step, gamma=lr_decay)
    
    T = 0.0
    t0 = time.time()

    ndcg_last = 0.0
    hr_last = 0.0
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):

        if args.inference_only: 
            break # just to decrease identition

        lr = adam_optimizer.param_groups[0]['lr']
        
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): 
                loss += args.l2_emb * torch.norm(param)
            writer.add_scalar("Loss/Train", loss, epoch)
            loss.backward()
            adam_optimizer.step()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + " [%2d] Lr: %5f, Training: %.2f%%, " %(epoch, lr, 100 * ((step + 1) / num_batch)) + " Loss: %.5f \r" %(loss.item()))
            sys.stderr.flush()

        sys.stdout.write("\n")

        if epoch % 5 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print("\n", end="")
            print('Epoch: %d, Time: %f(s), Valid (NDCG@10: %.4f, HR@10: %.4f, NDCG@20: %.4f, HR@20: %.4f), Test (NDCG@10: %.4f, HR@10: %.4f, NDCG@20: %.4f, HR@20: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_valid[2], t_valid[3], t_test[0], t_test[1], t_test[2], t_test[3]))
            
            # log to tensorboard
            writer.add_scalar("NDCG@10_validation", t_valid[0], epoch)
            writer.add_scalar("HR@10_validation", t_valid[1], epoch)
            writer.add_scalar("NDCG@20_validation", t_valid[2], epoch)
            writer.add_scalar("HR@20_validation", t_valid[3], epoch)
            writer.add_scalar("NDCG@10_test", t_test[0], epoch)
            writer.add_scalar("HR@10_test", t_test[1], epoch)
            writer.add_scalar("NDCG@20_test", t_test[2], epoch)
            writer.add_scalar("HR@20_test", t_test[3], epoch)
            t0 = time.time()
            model.train()
            
            folder = args.dataset + '_' + args.train_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.units={}.dropt={}.blocks={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen, args.hidden_units, args.dropout_rate, args.num_blocks)
            
            if t_valid[0] > ndcg_last and t_valid[1] > hr_last:
                print("Found best ndcg and hr score on validation dataset, Saving model ...........")
                torch.save(model.state_dict(), os.path.join(folder, fname))
                ndcg_last = t_valid[0]
                hr_last = t_valid[1]
                f.write(str(t_valid) + ' ' + str(t_test) + '\n')
                f.flush()
            # elif t_valid[0] <= ndcg_last:
            #     count_early_stop += 1
            #     if count_early_stop == num_early_stop:
            #         print("NDCG@10 not decrease in 20 epoch ... Stop training !!!")
            #         break

        scheduler.step()
    
    f.close()
    sampler.close()
    writer.flush()
    writer.close()
    print("Done")
