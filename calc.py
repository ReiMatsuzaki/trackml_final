import sys
import os
join = os.path.join

import numpy as np
import pandas as pd
import datetime

path_to_trackml = os.path.expanduser("~/trackml")
path_to_trackmllib = os.path.join(path_to_trackml, "trackml-library")
sys.path.append(path_to_trackmllib)
from trackml.dataset import load_dataset
from trackml.score import score_event
from tqdm import tqdm

import model_bin as mod_model
from util_func import get_path_to_submission

import argparse

# score memo
# N_bins_theta = 500, N_bins_w0v0inv = 500, lim_w0v0inv = 0.002
#
# N_seed      | score   |
    
def get_model(args):
    model  = mod_model.Clusterer(mode       = args.mode,
                                 merge      = args.merge,
                                 N_seed     = args.n_seed,
                                 std_z0     = args.std_z0,
                                 beta_r0inv = args.beta_r0inv,
                                 N_bins_theta0xy = args.n_bins_theta0xy,
                                 N_bins_theta0z  = args.n_bins_theta0z,
                                 func_theta0z = args.func_theta0z,
                                 i0=args.i0,
                                 max_num      = args.max_num,
                                 residual_merge_min_num = args.residual_merge_min_num,
                                 submission_root=args.submission_root)
    return model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default="bin")
    parser.add_argument('-merge', type=str, default="length")
    parser.add_argument('-in_path',  type=str, default="none")
    parser.add_argument('-out_path', type=str)
    parser.add_argument("-test", type=bool, default=False)    
    parser.add_argument('-num', type=int, default=1)
    parser.add_argument('-idx', type=int, default=0)
    parser.add_argument('-i0', type=int, default=0)

    parser.add_argument('-n_seed',   type=int, default=100)
    parser.add_argument('-beta_r0inv', type=float, default=0.0006)
    parser.add_argument('-std_z0',     type=float, default=5.5)
    parser.add_argument('-n_bins_theta0xy', type=int, default=500)
    parser.add_argument('-n_bins_theta0z',  type=int, default=500)
    parser.add_argument('-max_num', type=int, default=20)
    parser.add_argument('-func_theta0z', type=str, default="gaussplus")
    parser.add_argument('-residual_merge_min_num', type=int, default=3)
        
    parser.add_argument('-submission_root', type=str, default=None)


    
    args = parser.parse_args()
    return args

def run():
    print(datetime.datetime.now(), sys.argv[0], " begin")
    
    args = get_args()
    
    if(not os.path.exists(args.out_path)):
        os.makedirs(args.out_path)        
        
    if(args.test):        
        print("calculation begin for test")        
        path_to_input = os.path.join(path_to_trackml, "test")
        nevents = 125
        parts = ["hits"]
        
    else:
        print("calculation begin for train_1")
        path_to_input = os.path.join(path_to_trackml, "train_1")
        nevents = 1
        parts = ["hits", "truth"]

    for (i, data) in tqdm(enumerate(load_dataset(path_to_input, parts=parts, nevents=nevents)),
                          total=nevents):

        if(len(data)==3):
            (event_id, hits, truth) = data
        elif(len(data)==2):
            (event_id, hits) = data
        else:
            print(data)
            raise RuntimeError("not match")
        
        if(np.mod(i, args.num) == args.idx):
            print("go calc for event_id=", event_id)
        else:
            print("pass calc for event_id=", event_id)
            continue
        
        print("len(hits): ", len(hits))
        hits["event_id"] = event_id

        model = get_model(args)
        path_to_sub = get_path_to_submission(args.in_path, event_id)
        if(path_to_sub is None):
            print("submission set None")
        else:
            print("submission read from  {0}".format(path_to_sub))
            model.set_submission(pd.read_csv(path_to_sub))
        model.fit(hits)
        submission = model.submission
        
        path_to_sub = get_path_to_submission(args.out_path, event_id)
        if(not os.path.exists(os.path.dirname(path_to_sub))):
            os.makedirs(os.path.dirname(path_to_sub))
        submission.to_csv(path_to_sub, index=None)
        if(len(data)==3):
            score = score_event(truth, model.submission)
            print("score: ", score)

    print(datetime.datetime.now(), sys.argv[0], " end")
    
if __name__=="__main__":
    run()

    
