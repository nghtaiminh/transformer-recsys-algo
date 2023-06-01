import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from recommenders.datasets.split_utils import filter_k_core
print("Loading data...")
df = pd.read_csv("Books.csv", names=["itemID","userID","rating","timestamp"])
print("Remove interactions with rating < 3")
df = df[df['rating'] >= 3]
# filter users and items have less than 30 interactions
print("Filter users and items have less than 30 interactions")
df = filter_k_core(df, 30)

# export preprocessed ratings file
columns_titles = ["userID","itemID", "rating", "timestamp"]
df = df.reindex(columns=columns_titles)
print("Export preprocessed ratings file")
df.to_csv("ratings.csv", header=False, index=False)
 # mapping user and item ID
user_set, item_set = set(df['userID'].unique()), set(df['itemID'].unique())
user_map = dict()
item_map = dict()
# item ID, start at 1
for u, user in enumerate(user_set):
    user_map[user] = u+1
for i, item in enumerate(item_set):
    item_map[item] = i+1
    
df["userID"] = df["userID"].apply(lambda x: user_map[x])
df["itemID"] = df["itemID"].apply(lambda x: item_map[x])
# sort by userID and timestamp
df = df.sort_values(by=["userID", "timestamp"])
df.drop(columns=["timestamp", "rating"], inplace=True)
df = df[['userID', 'itemID']]

df.to_csv("Books.txt", sep="\t", header=False, index=False)
print("Export mapping file")
# save mapping for later use
with open('itemmap.pkl', 'wb') as f:
    pickle.dump(item_map, f)

with open('usermap.pkl', 'wb') as f:
    pickle.dump(user_map, f)

print("Done!")