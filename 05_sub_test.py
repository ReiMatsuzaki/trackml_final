import os
import pandas as pd
from glob import glob
import numpy as np
import sys
from tqdm import tqdm

path_to_trackml = os.path.expanduser("~/trackml")
path_to_trackmllib = os.path.join(path_to_trackml, "trackml-library")
sys.path.append(path_to_trackmllib)
from trackml.dataset import load_dataset
from trackml.score import score_event

from util_func import get_path_to_submission

n = sys.argv[1]
max_track_id = 0
submission_list = []
maxint = 2147483648
path_to_input = os.path.join(path_to_trackml, "test")
out_dir = os.path.join("out_"+sys.argv[0].split(".")[0], n)
if(not os.path.exists(out_dir)):
    os.makedirs(out_dir)
out_name = os.path.join(out_dir, "trackml_submit_{0}_{1}.csv".format(sys.argv[0].split(".")[0], n))
print(out_name)

submission_list = []
for (event_id, hits) in tqdm(load_dataset(path_to_input, parts=["hits"]), total=125):

    path_to_sub = get_path_to_submission(os.path.join("out_04_merge_test", n), event_id)    
    submission = pd.read_csv(path_to_sub)
            
    tids = submission["track_id"].values
    uniq, indeces, counts = np.unique(tids, return_inverse=True, return_counts=True)
    tids = np.where( (uniq[indeces]>-1) & (counts[indeces]>1), 1+indeces+max_track_id, 0)

    if(np.any(tids>maxint)):
       print("found too much integer")
       sys.exit()

    submission["event_id"] = event_id
    submission["track_id"] = tids
    
    max_track_id = np.max(submission["track_id"].values)
    submission_list.append(submission)

df_all = pd.concat(submission_list)
df_all.to_csv(out_name, index=None)



