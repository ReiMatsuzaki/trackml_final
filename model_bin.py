import sys
import os
join = os.path.join

import numpy as np
import scipy.special
import pandas as pd
import datetime

path_to_trackml = os.path.expanduser("~/trackml")
path_to_trackmllib = os.path.join(path_to_trackml, "trackml-library")
sys.path.append(path_to_trackmllib)
from trackml.dataset import load_dataset
from trackml.score import score_event
from tqdm import tqdm
from itertools import product

def merge_by_residual(sub_list, dfh, min_num=3):

    dfh["R"]      = np.sqrt(dfh["x"].values**2 + dfh["y"].values**2)
    dfh["Phi"]    = np.arctan2(dfh["y"].values,  dfh["x"].values)
    dfh["CosPhi"] = np.cos(dfh["Phi"])
    dfh["SinPhi"] = np.sin(dfh["Phi"])

    num = len(dfh)
    best_candidate_id = np.zeros(num, dtype=int)
    best_candidate_re = np.ones( num, dtype=float)*100
    best_candidate_num= np.ones( num, dtype=int)
    
    for sub in sub_list:

        dfall = pd.merge(dfh, sub).copy()

        candidate_id = sub["track_id"].values        
        tids, indeces, counts = np.unique(candidate_id, return_inverse=True, return_counts=True)
        tids = tids[ (tids>0) & (counts>min_num) ]
        candidate_num = np.where(candidate_id<1, 1, counts[indeces])

        candidate_re = np.ones(num, dtype=float)*100
        for tid in tqdm(tids):
            df = dfall[dfall.track_id==tid].copy()
            df.sort_values(by="z")

            ntrack = len(df)

            # Solve Hough equation
            #     r = 2 r0 cos(phi - theta0xy)
            #       = 2 r0 cos(phi) cos(theta0xy) + 2 r0 sin(phi) sin(theta0xy)
            sol = np.linalg.lstsq(df[["CosPhi", "SinPhi"]].values, df["R"].values, rcond=-1)
            if(len(sol[1])==0):
                candidate_re[dfall.track_id==tid] = 0.0
                continue
            
            residual1 = sol[1][0]
            
            # solve :  tan[theta0z]  = (z-z0) / {r0(pi - 2(phi-theta0xy))}
            df["one"] = 1.0
            sol = np.linalg.lstsq(df[["Phi", "one"]].values, df["z"], rcond=-1)
            residual2 = sol[1][0]

            # update residual
            candidate_re[dfall.track_id==tid] = residual2*residual1

        max_id = np.max(best_candidate_id)
        mask = ( ( candidate_num >  best_candidate_num) |
                 ((candidate_num == best_candidate_num) & (candidate_re < best_candidate_re)) )
        best_candidate_id = np.where(mask, candidate_id+max_id, best_candidate_id)
        best_candidate_num= np.where(mask, candidate_num,       best_candidate_num)
        best_candidate_re = np.where(mask, candidate_re,        best_candidate_re)

    return pd.DataFrame({"event_id": sub_list[0]["event_id"].values,
                         "hit_id": dfh["hit_id"],
                         "track_id": best_candidate_id})

def merge_by_residual2(sub1, sub2, dfh, max_num, residual_merge_min_num):
    
    dfh["R"]      = np.sqrt(dfh["x"].values**2 + dfh["y"].values**2)
    dfh["Phi"]    = np.arctan2(dfh["y"].values,  dfh["x"].values)
    dfh["CosPhi"] = np.cos(dfh["Phi"])
    dfh["SinPhi"] = np.sin(dfh["Phi"])

    for sub in [sub1, sub2]:
        
        dfall = pd.merge(dfh, sub, on="hit_id").copy()
        
        unique, indeces, counts = np.unique(sub["track_id"].values,
                                            return_counts=True, return_inverse=True)
        mask = ((counts[indeces]==1) |
                (counts[indeces]>max_num) |
                (sub["track_id"].values==0))
        sub["track_id"] = np.where(mask, 0, indeces)
        sub["nhits"]    = np.where(mask, 1, counts[indeces])

        tids = unique[ (unique>0) & (counts>residual_merge_min_num) ]
        sub["residual"] = 1000.0
        for tid in tqdm(tids):
            df = dfall[dfall.track_id==tid].copy()
            df.sort_values(by="z")
            
            # Solve Hough equation
            #     r = 2 r0 cos(phi - theta0xy)
            #       = 2 r0 cos(phi) cos(theta0xy) + 2 r0 sin(phi) sin(theta0xy)
            sol = np.linalg.lstsq(df[["CosPhi", "SinPhi"]].values, df["R"].values, rcond=-1)
            if(len(sol[1])==0):
                candidate_re[dfall.track_id==tid] = 0.0
                continue
            residual1 = sol[1][0]
            
            # solve :  tan[theta0z]  = (z-z0) / {r0(pi - 2(phi-theta0xy))}
            df["one"] = 1.0
            sol = np.linalg.lstsq(df[["Phi", "one"]].values, df["z"], rcond=-1)
            residual2 = sol[1][0]

            # update residual
            sub.loc[sub.track_id==tid, "residual"] = residual2*residual1
            
    max_id1 = np.max(sub1["track_id"].values)
    mask = ( ( sub1["nhits"].values >  sub2["nhits"].values) |
             ((sub1["nhits"].values == sub2["nhits"].values) &
              (sub1["residual"].values < sub2["residual"].values)))
    sub1["track_id"] = np.where(mask, sub1["track_id"], sub2["track_id"]+max_id1)
    sub1["nhits"]    = np.where(mask, sub1["nhits"],    sub2["nhits"])
    return sub1

def merge_by_length2(sub1, sub2, max_num):
    
    for sub in [sub1, sub2]:
        #if("nhits" in sub.columns):
        #    continue
        unique, indeces, counts = np.unique(sub["track_id"].values,
                                            return_counts=True, return_inverse=True)
        mask = ((counts[indeces]==1) |
                (counts[indeces]>max_num) |
                (sub["track_id"].values==0))
        sub["track_id"] = np.where(mask, 0, indeces)
        sub["nhits"]    = np.where(mask, 1, counts[indeces])

    max_id1 = np.max(sub1["track_id"].values)
    mask = sub1["nhits"].values > sub2["nhits"].values
    sub1["track_id"] = np.where(mask, sub1["track_id"], sub2["track_id"]+max_id1)
    sub1["nhits"]    = np.where(mask, sub1["nhits"],    sub2["nhits"])

    if(len(sub1)!=len(sub2)):
        raise RuntimeError("Error after mergin: {0} {1}".format(len(sub1), len(sub2)))
    
    return sub1
    
def uniform_slaterplus(in_xs):
    xs = np.append(in_xs, 1.0)
    eps = 0.25/(np.pi/2)
    n = 2
    A = 15.0
    B = 110.0
    c = 7.0
    fs = A * ( (xs-1-eps)**(-n+1) * 1/(-n+1) +
               (xs+1+eps)**(-n+1) * 1/(-n+1) )
    fs+= B * np.sign(xs) / c * (1 - np.exp(-c*np.abs(xs)))
    fs = fs/fs[-1]
    return fs[:-1]            
    
def digitize_column(df, col, N, minval=None, maxval=None):
    if(minval is None):
        minval = np.min(df[col].values)
    if(maxval is None):
        maxval = np.max(df[col].values)
    mask = (minval < df[col].values) & (df[col].values < maxval)
    ddf = df[~mask]
    sval = (df[col].values-minval)/(maxval-minval)
    df[col+"Digi"] = np.where(mask, (N*sval).astype(int), 0)
    return df

def combine_digi(df, columns, Ns):
    mask = np.any(df[columns].values==0, axis=1)
    cumsum = 0
    NNs = [1]
    for N in Ns:
        NNs.append(NNs[-1] * (N+2))
    for i_col, acol in enumerate(columns):
        cumsum += (df[acol] * NNs[i_col]).astype(int)

    df["ComboDigi"] = np.where(mask, -1, cumsum)

class Clusterer():
    def __init__(self, mode="bin", merge="length",
                 N_seed=100,
                 std_z0 = 14.0,
                 beta_r0inv = 0.0006818,
                 N_bins_theta0xy = 500,
                 N_bins_theta0z  = 500,
                 func_theta0z = None,
                 verbose=True,
                 max_num=20,
                 residual_merge_min_num = 3,
                 submission=None,
                 submission_root = None,
                 c_logtan=0.23,
                 d_logtan=1.0,
                 i0=0,
                 B0=0.0006, dBdz=8e-8):
        self.mode = mode
        self.i0=i0
        self.merge = merge
        self.N_seed = N_seed
        self.std_z0 = std_z0
        self.beta_r0inv = beta_r0inv
        self.N_bins_theta0xy = N_bins_theta0xy
        self.N_bins_theta0z  = N_bins_theta0z
        self.func_theta0z = func_theta0z
        self.verbose = verbose
        self.max_num = max_num
        self.residual_merge_min_num = residual_merge_min_num
        self.submission = submission
        self.c_logtan = c_logtan
        self.d_logtan = d_logtan
        self.submission_root = submission_root
        self.B0 = B0
        self.dBdz = dBdz

    def init_hough(self, dfh):
        
        hough = dfh[["hit_id", "x", "y", "z"]].copy()
        hough.columns = ["HitID", "X", "Y", "Z"]

        # -- cylindrical transformation --
        hough["R"] = np.sqrt(hough["X"]**2 + hough["Y"]**2)
        hough["Phi"] = np.arctan2(hough["Y"], hough["X"])
        hough["B"] = self.B0
        z0 = 1500.0
        sz = np.sign(hough["Z"].values)
        hough["B"] = np.where( np.abs(hough["Z"])<z0, self.B0,
                               -sz*self.dBdz*(hough["Z"]-sz*z0) + self.B0)
        self.hough = hough
        self.hough["Theta0XY"] = 0.0
        self.hough["Theta0Z"] = 0.0
        return hough

    def hough_transform(self, r0inv, z0, q):

        ## use 1/r0 = B/p0
        r0inv = r0inv*(self.hough["B"]/self.B0)
        
        r_2r0inv = self.hough["R"] * r0inv / 2        
        mask = (r_2r0inv < 1.0 + 1.0e-10)
        self.hough["mask"] = mask

        phi = self.hough.loc[mask,"Phi"]
        dphi = np.arccos(r_2r0inv[mask])       # dphi >0 for anytime
        theta0xy = np.mod(phi - q*dphi, 2*np.pi)
        theta0z  = np.arctan2(r0inv[mask] * (self.hough.loc[mask,"Z"].values-z0), np.pi-2*dphi)
        theta0z /= (np.pi/2)
        if(self.func_theta0z=="square"):
            theta0z = np.sign(theta0z) * theta0z**2
        elif(self.func_theta0z == "poly"):
            xs1 = theta0z
            xs2 = xs1*xs1
            xs3 = xs1*xs2
            xs5 = xs3*xs2
            C1=18.75
            C2=-14.42460
            C3=8.0
            theta0z = (C1*xs5 + C2*xs3 + C3*xs1) / (C1+C2+C3)
        elif(self.func_theta0z == "gaussplus"):
            erf = scipy.special.erf
            xs = np.append(theta0z, 1.0)
            eps = 0.25/(np.pi/2)
            n = 2
            A = 15.0
            B = 50.0
            c = 30.0
            fs = A * ( (xs-1-eps)**(-n+1) * 1/(-n+1) +
                       (xs+1+eps)**(-n+1) * 1/(-n+1) )
            fs+= B * np.sqrt(np.pi)/2 * erf(np.sqrt(c)*xs) / (2*np.sqrt(c))
            fs = fs/fs[-1]
            theta0z = fs[:-1]
            
        elif(self.func_theta0z == "slaterplus"):
            theta0z = uniform_slaterplus(theta0z)
        elif(self.func_theta0z == "tan"):
            theta0z = tan(theta0z*np.pi/2)
            
        elif(self.func_theta0z == "logtan"):

            invc = 1/self.c_logtan
            d = self.d_logtan
            t1 = 88.0*np.pi/180.0
            theta0z_prime = np.sign(theta0z) * np.log(1 + invc*np.tan(d*np.abs(theta0z)*np.pi/2))
            limval = np.log(1 + (invc*np.tan(d*t1)))
            theta0z = theta0z_prime / limval
            
        elif(self.func_theta0z is None):
            theta0z = theta0z*1.0
            
        else:
            raise RuntimeError("invalid func_theta0z")

        self.hough.loc[mask, "Theta0XY"] = theta0xy
        self.hough.loc[mask, "Theta0Z"] = theta0z
        return self.hough
    
    def predict(self, dfh):

        # -- init --
        self.init_hough(dfh)

        n_pow = int(np.log2(self.N_seed))+1
        subs_list = [[] for _ in range(n_pow)]
        for i in tqdm(range(self.N_seed), total=self.N_seed, disable=not self.verbose):

            if(self.mode == "bin"):
                sub = self.predict_by_bin(dfh)
            elif(self.mode == "read"):
                sub = self.predict_by_read(dfh["event_id"].values[0], i+self.i0*self.N_seed)
            else:
                raise RuntimeError("not impl")
            subs_list[0].append(sub)

            # -- merging --
            while True:
                find_pair = False
                for n in range(n_pow):
                    if(len(subs_list[n]) == 2):
                        if(self.merge=="length"):
                            sub = merge_by_length2(subs_list[n][0], subs_list[n][1], self.max_num)
                        elif(self.merge=="residual"):
                            sub = merge_by_residual2(subs_list[n][0], subs_list[n][1],
                                                     dfh, self.max_num, self.residual_merge_min_num)
                        else:
                            raise RuntimeError("not supported")
                        
                        subs_list[n] = []
                        subs_list[n+1].append(sub)
                        find_pair = True
                        continue
                    elif(len(subs_list[n]) > 2):
                        raise RuntimeError("too long subs_list")
                if(not find_pair):
                    break
                
        labels = subs_list[-1][0]["track_id"].values
        return labels
    
    def predict_by_bin(self, dfh):
        # -- produce part of helix parameters --
        z0    = np.random.normal(0.0, self.std_z0)
        r0inv = np.random.gamma(2,  self.beta_r0inv)
        q     = np.random.choice([+1.0, -1.0])
        d1     = np.random.uniform(-np.pi, np.pi)/self.N_bins_theta0xy
        d2     = np.random.uniform(-1.0, 1.0)/self.N_bins_theta0z
        
        # -- compute other helix parameters --
        self.hough_transform(r0inv, z0, q)
        self.hough["Theta0XY"] += d1
        self.hough["Theta0Z"]  += d2
        digitize_column(self.hough, "Theta0XY", self.N_bins_theta0xy, 0.0, 2*np.pi)
        digitize_column(self.hough, "Theta0Z",  self.N_bins_theta0z, -1, 1)
        combine_digi(self.hough, ["Theta0XYDigi", "Theta0ZDigi"],
                     [self.N_bins_theta0xy, self.N_bins_theta0z])
        
        # -- clustering --
        sub = pd.DataFrame({"event_id": 0,
                            "hit_id":   dfh.hit_id.values,
                            "track_id": self.hough["ComboDigi"]})

        return sub

    def predict_by_read(self, event_id, i):
        if(self.submission_root is None):
            raise RuntimeError("submission_root is None")
        from util_func import get_path_to_submission
        path_to_submission = get_path_to_submission(os.path.join(self.submission_root, "{0:0>3}".format(i)), event_id)
        return pd.read_csv(path_to_submission)
        
    def fit(self, dfh, y=None):
        labels = self.predict(dfh)
        self.submission = pd.DataFrame({"event_id": dfh["event_id"].values,
                                        "hit_id": dfh["hit_id"].values,
                                        "track_id": labels})
        return self
