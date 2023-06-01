# BERT4Rec

## Requirements

- A requirement file is available

## Configurations

### Dataset

- The raw dataset (rating file) can be dowloaded [here](https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Books.csv).
- The interaction file is placed in `Data/AmazonBooks/ratings.csv`, the current format:
    > uid, sid(item id), rating, timestamp
- (Note that `ratings.csv` has been removed items and users that have less 30 interactions, so the preprocess in this code only change the index and the amount of interaction is the same as `Books.txt` in SASRec-Pytorch folder).


### Training the model

```bash
python main.py --template train_bert
```

The trained models and tensorboard log are exported at experiments/test_{date}/models/

## Reference

Link to the original repo: [here](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch)

```TeX
@inproceedings{Sun:2019:BSR:3357384.3357895,
 author = {Sun, Fei and Liu, Jun and Wu, Jian and Pei, Changhua and Lin, Xiao and Ou, Wenwu and Jiang, Peng},
 title = {BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer},
 booktitle = {Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
 series = {CIKM '19},
 year = {2019},
 isbn = {978-1-4503-6976-3},
 location = {Beijing, China},
 pages = {1441--1450},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/3357384.3357895},
 doi = {10.1145/3357384.3357895},
 acmid = {3357895},
 publisher = {ACM},
 address = {New York, NY, USA}
} 
```