# SASRec: Self-Attentive for Sequential Recommendation

## Requirements
- Python 3
- numpy
- torch
- [recommenders](https://github.com/microsoft/recommenders) lib for preprocess only

## Configurations

### Dataset

- The raw dataset (rating file) can be dowload [here](https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Books.csv).
- The interaction file `Books.csv` will be placed in the `data` folder.
- The preprocessed data is in `data/Books.txt`, the file have the current format:
  > user_id      item_id


### Train the SASRec Model

```bash
python train.py 
```

# Reference

Link to the original repo: [here](https://github.com/pmixer/SASRec.pytorch)

```
@inproceedings{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={197--206},
  year={2018},
  organization={IEEE}
}
```