from scipy.stats.stats import pearsonr
from joblib import Parallel, delayed
import numpy as np
import pandas as pd


def rank_evaluator(pred, true):
    metrics, r, e = backtest_protefolio(pred, true)
    metrics["MRR"] = MRR(pred, true)
    metrics["MSE"] = np.mean((true - pred) ** 2)
    return metrics, r , e


def daily_metirc_protefolio(pred_d, true_d, K = 30):
    ret_df = pd.DataFrame({"true":true_d,"pred":pred_d})
    rank_df = ret_df.rank(pct=True)
    rank_df["true_ret"] = true_d
    long = rank_df.nlargest(K, columns='pred').true_ret.mean()
    IC =  ret_df.true.corr(ret_df.pred)
    r = long
    exceed = r - true_d.mean()
    return pd.Series({"IC":IC, "R":r, "E":exceed})

def backtest_protefolio(pred, true):
    dates = list(range(true.shape[0]))
    res = Parallel(n_jobs=20)(delayed(daily_metirc_protefolio)(pred[d], true[d]) for d in dates)
    res = {
       dates[i]: res[i]
       for i in range(len(dates))
    }
    res = pd.DataFrame(res).T
    r = res['R'].copy()
    e = res['E'].copy()
    return {
    'AAR': (1 + r.to_numpy()).cumprod()[-1] ** (252 / true.shape[0]) - 1,
    'AV': r.std() * 252 ** 0.5,
    'MDD': (r.cumsum().cummax() - r.cumsum()).max(),
    'EAAR': ((1 + r.to_numpy()).cumprod()[-1] ** (252 / true.shape[0]) - 1) - ((1 + true.mean(axis = 1)).cumprod()[-1] ** (252 / true.shape[0]) - 1),
    'EAV': e.std()  * 252 ** 0.5,
    'EMDD': (e.cumsum().cummax() - e.cumsum()).max()
    }, r, e

def topK_MAP(pred,true, k):
    #shape: t * n
    topk_index_gt = np.argsort(true, axis = -1)[:,::-1][:,:k]
    pred_rank = (-pred).argsort().argsort() + 1
    topk_rank_in_pred = np.take_along_axis(pred_rank, topk_index_gt, axis=1)
    MAP = np.mean((np.arange(k) + 1) / np.sort(topk_rank_in_pred, axis = -1))
    return MAP

def MRR(pred,true, k = 1):
    #Mean Reciprocal Rank in a given period
    topk_index_gt = np.argsort(true, axis = -1)[:,::-1][:,:k]
    pred_rank = (-pred).argsort().argsort() + 1
    topk_rank_in_pred = np.take_along_axis(pred_rank, topk_index_gt, axis=1)
    MAP = np.mean((np.arange(k) + 1) / np.sort(topk_rank_in_pred, axis = -1))
    return MAP

def rowwise_cor(A,B):
    diag_pear_coef = [pearsonr(A[i, :], B[i, :])[0] for i in range(A.shape[0])]
    return diag_pear_coef


def topK_IRR(pred, true, k):
    ## Investment Retrun Ratio
    pred_rank_index = np.argsort(pred, axis = -1)[:,::-1][:,:k]
    selected_real_returns = np.take_along_axis(true, pred_rank_index, axis=1)
    selected_real_return = np.sum(selected_real_returns)
    return selected_real_return

