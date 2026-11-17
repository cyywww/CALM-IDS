r"""Shared pieces of CALM-IDS: data loading, the piecewise-linear encoding (PLE) and the evaluation metrics.

Data (thesis Section 3.3), in <repo>/datasets/ (see README):
  NSL-KDD/PKDDTrain+.csv, PKDDTest+.csv            NSL-KDD as preprocessed in the AOC-IDS repository (its NSL_pre_data/)
  UNSW-NB15/UNSWTrain.csv, UNSWTest.csv            UNSW-NB15 as preprocessed in the AOC-IDS repository (its UNSW_pre_data/)
  UNSW-NB15/UNSW_NB15_{training,testing}-set.csv   the original UNSW-NB15 files; only their attack categories are used, in the evaluation
The 20/80 split into the initial labelled set D0 and the data stream is made by the caller (train_test_split(test_size=0.8, random_state=seed)).
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve

DATA = Path(__file__).resolve().parent.parent / "datasets"
_trapz = getattr(np, "trapezoid", None) or np.trapz          # numpy >= 2.0 renamed trapz


def load(ds):
    """Return (x_tr, y_tr, fam_tr, x_te, y_te, fam_te) for ds in {'nsl', 'unsw'}: raw features (float), binary labels (1 = attack) and
    attack-category names ('normal' for normal flows) of the training and the test file."""
    if ds == "nsl":
        tr, te = (pd.read_csv(DATA / f"NSL-KDD/PKDD{n}+.csv") for n in ("Train", "Test"))
        y_tr, y_te = ((d["labels2"] != "normal").to_numpy().astype(int) for d in (tr, te))
        fam_tr, fam_te = (d["labels5"].to_numpy().astype(str) for d in (tr, te))
        drop = ["labels5", "labels2"]
    else:
        tr, te = (pd.read_csv(DATA / f"UNSW-NB15/UNSW{n}.csv") for n in ("Train", "Test"))
        y_tr, y_te = (d["label"].to_numpy().astype(int) for d in (tr, te))
        cat = [pd.read_csv(DATA / f"UNSW-NB15/UNSW_NB15_{n}-set.csv", usecols=["attack_cat"])["attack_cat"].to_numpy().astype(str)
               for n in ("training", "testing")]
        fam_tr, fam_te = (np.where(c == "Normal", "normal", c) for c in cat)
        drop = ["label"]
    return tr.drop(drop, axis=1).to_numpy(float), y_tr, fam_tr, te.drop(drop, axis=1).to_numpy(float), y_te, fam_te


class PLE:
    """Piecewise-linear encoding (Gorishniy et al. 2022): numerical columns (> 2 distinct values in the training file) are cut into
    T bins at the quantiles of the labelled set D0; binary columns pass through. Bin edges are computed on D0 only."""
    def __init__(self, x_all, x_d0, T):
        self.num = np.array([len(np.unique(x_all[:, j])) > 2 for j in range(x_all.shape[1])])
        self.edges = []
        for j in np.where(self.num)[0]:
            e = np.unique(np.quantile(x_d0[:, j], np.linspace(0, 1, T + 1)))
            self.edges.append(e if len(e) >= 2 else np.array([x_d0[:, j].min(), x_d0[:, j].min() + 1.0]))
        self.dim = int((~self.num).sum()) + sum(len(e) - 1 for e in self.edges)
    def __call__(self, x):
        out = [x[:, ~self.num]]
        for j, e in zip(np.where(self.num)[0], self.edges):
            lo, hi = e[:-1][None, :], e[1:][None, :]
            out.append(np.clip((x[:, j][:, None] - lo) / (hi - lo), 0.0, 1.0))
        return np.concatenate(out, 1).astype(np.float32)


# ---------------------------------- evaluation metrics ----------------------------------
def pauc(S, y, b):
    """pAUC[0, b] in percent: the trapezoidal ROC area over FPR <= b, divided by b (the average TPR over the FPRs up to b)."""
    f_, t_, _ = roc_curve(y, S); m = f_ <= b
    return float(_trapz(np.append(t_[m], np.interp(b, f_, t_)), np.append(f_[m], b)) / b) * 100


def metrics(s_te, y_te, thr):
    """Operating point of the threshold thr: accuracy, precision, recall (TPR) and F1 in percent, FPR as a fraction."""
    pred = (s_te > thr).astype(int); tp = int(((pred == 1) & (y_te == 1)).sum()); fp = int(((pred == 1) & (y_te == 0)).sum())
    fn = int(((pred == 0) & (y_te == 1)).sum()); tn = int(((pred == 0) & (y_te == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(acc=100 * (tp + tn) / len(y_te), prec=100 * prec, rec=100 * rec, f1=100 * (2 * prec * rec / (prec + rec) if prec + rec else 0.0),
                fpr=fp / max(tn + fp, 1), thr=float(thr))
