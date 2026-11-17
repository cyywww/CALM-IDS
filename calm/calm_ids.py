r"""CALM-IDS (Calibrated Adaptive Latent Mixtures for Intrusion Detection) in one program (thesis Chapter 4).
The defaults are the method; the switches are the variants used in the experiments of the thesis (see run_experiments.sh).

  Score   piecewise-linear encoding (PLE, 16 quantile bins fitted on the initial labelled set D0) -> TabM-mini adapters (4 members) ->
          shared spectrally normalised network (projection d -> 32 with bound 3, two residual blocks with ||W||_2 <= 0.9); per member a
          normal and an attack Gaussian mixture in the latent space: K=8 normal and J=8 attack unit-variance components, plus the central
          component on the attack side (weight eps=0.05, variance 4 times the mean latent variance); score s = mean over the members of
          log p_A - log p_N. Joint training with the classification term (cost weight kappa on normal flows, fed back from the target
          FPR alpha) plus lambda=0.1 times the density term; then the encoder is frozen and the mixtures are refitted to the frozen
          latents by a labelled EM.
  Decide  a flow is classified as an attack when its score exceeds the conformal (1-alpha) quantile of the scores of the calibration set
          (20% of the normal flows of D0, held out from training), alpha=0.01; the same rule on the test set and on every batch.
  Adapt   every batch: score with the current model, classify, and let the flows classified as attacks move the attack component means
          by a running-average (online EM) step with t_0 = |training flows| / |batch|; everything else stays fixed.
  Output  <out>/<tag>.json with one record per seed: pAUC up to 1/2/5/10/20% FPR, AUC and operating points on the test set, and the
          per-batch FPR and TPR on the data stream; with --dump DIR also the test-set scores, labels and attack categories of every seed
          (DIR/<tag>_s<seed>.npz: S_te, yte, fam_te).
Read-only switch (it records, it changes nothing): --eval_every N (test metrics every N batches).
"""
import argparse, json, sys, time, math
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); from common import load, metrics, pauc, PLE

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="nsl", choices=["nsl", "unsw"])
ap.add_argument("--seed0", type=int, default=5009); ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--mode", default="online", choices=["offline", "online"])                               # offline = no data stream (the initial model)
ap.add_argument("--epochs", type=int, default=None)                                                        # default 30 (NSL-KDD) / 300 (UNSW-NB15)
ap.add_argument("--alpha", type=float, default=0.01)                                                       # target FPR alpha of the calibrated threshold; also the target of the cost weight kappa
# ablation (Table 5.7): --c 0 --c0 0 (without spectral normalisation), --K 1 --J 1 (one Gaussian per class), --eps 1e-6 (without the central component)
ap.add_argument("--c", type=float, default=0.9); ap.add_argument("--c0", type=float, default=3.0)          # spectral-norm bounds of the residual blocks and of the projection (0 = none)
ap.add_argument("--K", type=int, default=8); ap.add_argument("--J", type=int, default=8)                   # normal and attack components per member
ap.add_argument("--eps", type=float, default=0.05)                                                         # weight of the central component
# variant that also retrains the encoder on the data stream with soft pseudo-labels (Table 5.6): --enc_online 1
ap.add_argument("--enc_online", type=int, default=0)
# zero-day attacks (Fig. 5.4): --lofo <category> --lofo_mode a|b (a = absent from D0 and from the data stream, b = absent from D0 only);
# --centre_update 0 = the online update switched off
ap.add_argument("--lofo", default=""); ap.add_argument("--lofo_mode", default="b", choices=["a", "b"]); ap.add_argument("--centre_update", type=int, default=1)
# read-only evaluation switch and output
ap.add_argument("--eval_every", type=int, default=0)
ap.add_argument("--tag", default="calm"); ap.add_argument("--out", default=str(HERE.parent / "results" / "main")); ap.add_argument("--dump", default="")
a = ap.parse_args() if __name__ == "__main__" else ap.parse_args([])
if a.epochs is None: a.epochs = {"nsl": 30, "unsw": 300}[a.dataset]
# the constants of the method (Table 5.2 of the thesis)
M, BLOCKS, MEMBERS, BINS = 32, 2, 4, 16               # latent width, residual blocks, ensemble members, quantile bins of the encoding
GAMMA_O = 4.0                                          # variance of the central component, as a multiple of the mean latent variance
LAM = 0.1                                              # weight lambda of the density term
KAPPA_MAX, KAPPA_WARM = 50.0, 2                        # upper bound of the cost weight kappa and the epoch from which it is adjusted
LR, BS = 1e-3, 256                                     # Adam learning rate and mini-batch size
REF_SPLIT = 0.2                                        # share of the normal flows of D0 held out as the calibration set
E1, LR_ONLINE = 1, 1e-4                                # the encoder-retraining variant: epochs per batch and learning rate
INTERVAL = {"nsl": 2000, "unsw": 2784}[a.dataset]      # batch size of the stream (the reference paper's)
ALPHAS = (0.01, 0.05, 0.10, 0.20)                      # operating points reported on the test file
torch.set_num_threads(2)
LOG2PI = math.log(2 * math.pi)

def mdist(z, c, v):
    """Diagonal Mahalanobis squared distance sum_d (z_d - c_d)^2 / v_d, shape (B, K)."""
    return ((z.pow(2) / v).sum(1)[:, None] + (c.pow(2) / v).sum(1)[None, :] - 2 * z @ (c / v).t()).clamp(min=0.0)

# ───────────────────────────── encoder ─────────────────────────────
class SNLinear(nn.Module):
    """W_eff = W * min(1, c / sigma_max(W)) (soft spectral normalisation; sigma_max exact, power iteration if the SVD fails)."""
    def __init__(self, din, dout, c):
        super().__init__(); self.lin = nn.Linear(din, dout); self.c = c
    def weight(self):
        W = self.lin.weight
        if self.c is None or self.c <= 0: return W
        try: sig = torch.linalg.matrix_norm(W, ord=2)
        except RuntimeError:
            with torch.no_grad():
                v = torch.randn(W.shape[1]); v = v / v.norm()
                for _ in range(30): u = W @ v; u = u / (u.norm() + 1e-12); v = W.t() @ u; v = v / (v.norm() + 1e-12)
            sig = (u @ (W @ v)).abs()
        return W * torch.clamp(self.c / sig, max=1.0)
    def forward(self, x): return F.linear(x, self.weight(), self.lin.bias)
    def sigma(self): return float(torch.linalg.matrix_norm(self.weight().detach(), ord=2))

class Encoder(nn.Module):
    """Returns (B, k, m). k > 1 = TabM-mini: k input adapters R_i initialised at +-1, one shared trunk."""
    def __init__(self, d, m, L, c, c0, k=1):
        super().__init__(); self.k = k
        if k > 1: self.R = nn.Parameter(torch.sign(torch.randn(k, d)))
        self.proj = SNLinear(d, m, c0)
        self.blocks = nn.ModuleList([nn.ModuleList([SNLinear(m, m, c), SNLinear(m, m, c)]) for _ in range(L)])
    def body(self, x):
        h = self.proj(x)
        for w1, w2 in self.blocks: h = h + w2(F.leaky_relu(w1(h), 0.1))
        return h
    def forward(self, x):
        if self.k > 1: return self.body((x[:, None, :] * self.R[None]).reshape(-1, x.shape[1])).reshape(x.shape[0], self.k, -1)
        return self.body(x)[:, None, :]
    def sigmas(self): return [self.proj.sigma()] + [w.sigma() for blk in self.blocks for w in blk]

# ───────────────────────────── latent mixture of one member ─────────────────────────────
class Mixture(nn.Module):
    """Columns of the responsibility matrix: K normal components | J attack components | the central component (index iO)."""
    def __init__(self, m, K, J, eps):
        super().__init__(); self.m, self.K, self.J, self.eps = m, K, J, eps; self.iO = K + J; self.C = K + J + 1
        self.mu = nn.Parameter(torch.randn(K, m)); self.nu = nn.Parameter(torch.randn(J, m))
        self.register_buffer("vN", torch.ones(m, dtype=torch.float64)); self.register_buffer("vA", torch.ones(m, dtype=torch.float64))
        self.register_buffer("logw", torch.full((K,), -math.log(K))); self.register_buffer("logv", torch.full((J,), -math.log(J)))
        self.register_buffer("c_o", torch.zeros(m)); self.register_buffer("var_o", torch.tensor(10.0)); self.pi = 0.5
    def set_open(self, z):
        z = z.detach(); self.c_o.copy_(z.mean(0)); self.var_o.copy_((z - self.c_o).pow(2).sum(1).mean() / self.m * GAMMA_O)
    def comp_logp(self, z):
        z = z.double()
        lN = -mdist(z, self.mu.double(), self.vN) / 2 - 0.5 * (LOG2PI * self.m + torch.log(self.vN).sum())
        lK = -mdist(z, self.nu.double(), self.vA) / 2 - 0.5 * (LOG2PI * self.m + torch.log(self.vA).sum())
        lO = -(z - self.c_o.double()).pow(2).sum(1) / (2 * self.var_o.double()) - 0.5 * self.m * (LOG2PI + torch.log(self.var_o.double()))
        return lN, lK, lO
    def logp(self, z):
        lN, lK, lO = self.comp_logp(z)
        log_pN = torch.logsumexp(lN + self.logw, 1); log_pK = torch.logsumexp(lK + self.logv, 1)
        return log_pN, torch.logaddexp(math.log(1 - self.eps) + log_pK, math.log(self.eps) + lO)
    def score(self, z):
        lN_, lA_ = self.logp(z); return lA_ - lN_
    def resp(self, z):
        """Responsibilities of all components (the posterior probability that each component generated the flow)."""
        lN, lK, lO = self.comp_logp(z); pi = min(max(self.pi, 1e-6), 1 - 1e-6)
        return torch.softmax(torch.cat([lN + self.logw + math.log(1 - pi), lK + self.logv + math.log(pi * (1 - self.eps)), (lO + math.log(pi * self.eps))[:, None]], 1), 1)
    def resp_within(self, z, y):
        """Responsibilities within the labelled class (the labelled EM on the labelled set D0)."""
        lN, lK, lO = self.comp_logp(z); r = torch.zeros(len(z), self.C, dtype=torch.float64); b = y == 0
        r[b, :self.K] = torch.softmax(lN[b] + self.logw, 1)
        r[~b, self.K:] = torch.softmax(torch.cat([lK[~b] + self.logv + math.log(1 - self.eps), (lO[~b] + math.log(self.eps))[:, None]], 1), 1)
        return r
    def attack_side(self, r):
        """Responsibilities restricted to the attack side (attack components and the central component) and renormalised."""
        r = r.clone(); r[:, :self.K] = 0
        return r / r.sum(1, keepdim=True).clamp(min=1e-12)
    def suff(self, z, r):
        z = z.double(); return dict(n=r.mean(0), m=(r.t() @ z) / len(z))
    def mstep(self, S, weights=True, benign_centres=True, centres=True):
        K, J = self.K, self.J; n, m = S["n"], S["m"]
        if weights:
            nb, nk, no = n[:K], n[K:K + J], n[K + J:].sum()
            self.logw.copy_(torch.log(nb.clamp(min=1e-12) / nb.sum()).float()); self.logv.copy_(torch.log(nk.clamp(min=1e-12) / nk.sum()).float())
            self.pi = float((nk.sum() + no) / n.sum())
        if not centres: return
        lb, lk = n[:K] > 1e-9, n[K:K + J] > 1e-9
        if benign_centres: self.mu.data[lb] = (m[:K][lb] / n[:K][lb][:, None]).to(self.mu.dtype)
        self.nu.data[lk] = (m[K:K + J][lk] / n[K:K + J][lk][:, None]).to(self.nu.dtype)
class Ens:
    """The k members; the score is the mean of their log-likelihood ratios. Z has shape (B, k, m)."""
    def __init__(self, mixes): self.mixes = mixes; self.k = len(mixes)
    def score(self, Z): return torch.stack([mx.score(Z[:, i]) for i, mx in enumerate(self.mixes)]).mean(0)
    def set_open(self, Z):
        for i, mx in enumerate(self.mixes): mx.set_open(Z[:, i])
    def params(self): return [p for mx in self.mixes for p in mx.parameters()]

def latents(enc, x, chunk=20000):
    enc.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(x), chunk): out.append(enc(x[i:i + chunk]))
    enc.train(); return torch.cat(out)

def conformal_threshold(s_ref_sorted, alpha):
    """The conformal (1 - alpha) quantile of the sorted reference scores."""
    return float(s_ref_sorted[min(int(np.ceil((1 - alpha) * (len(s_ref_sorted) + 1))) - 1, len(s_ref_sorted) - 1)])

# ───────────────────────────── one seed ─────────────────────────────
def run_seed(s, x_tr, y_tr, fam_tr, x_te, y_te, fam_te):
    torch.manual_seed(s); np.random.seed(s); rng = np.random.default_rng(s)
    idx = np.arange(len(x_tr)); i0, i1 = train_test_split(idx, test_size=0.8, random_state=s)
    ox, oy, ofam, sx, sy, sfam = x_tr[i0], y_tr[i0], fam_tr[i0], x_tr[i1], y_tr[i1], fam_tr[i1]
    info = {}
    if a.lofo:
        drop_o = (ofam == a.lofo) & (oy == 1); info["lofo"] = dict(family=a.lofo, mode=a.lofo_mode, n_f_D0=int(drop_o.sum()),
                    n_f_stream=int(((sfam == a.lofo) & (sy == 1)).sum()), n_f_test=int(((fam_te == a.lofo) & (y_te == 1)).sum()))
        ox, oy, ofam = ox[~drop_o], oy[~drop_o], ofam[~drop_o]
        if a.lofo_mode == "a":
            drop_s = (sfam == a.lofo) & (sy == 1); sx, sy, sfam = sx[~drop_s], sy[~drop_s], sfam[~drop_s]
    enc_ = PLE(x_tr, ox, BINS); tf = lambda v: torch.from_numpy(enc_(v))
    OX, SX, XTE = tf(ox), tf(sx), tf(x_te)
    ben_idx, att_idx = np.nonzero(oy == 0)[0], np.nonzero(oy == 1)[0]
    pb = rng.permutation(len(ben_idx)); nb_ref = int(round(len(ben_idx) * REF_SPLIT)); ref_b, fit_b = ben_idx[pb[:nb_ref]], ben_idx[pb[nb_ref:]]
    fit_a = att_idx[rng.permutation(len(att_idx))]
    tr_i = np.concatenate([fit_b, fit_a]); XT, YT = OX[tr_i], torch.from_numpy(oy[tr_i]); B_ref = OX[ref_b]   # training flows; calibration set
    pi0 = float(YT.float().mean()); logit_pi0 = math.log(pi0 / (1 - pi0))
    k = MEMBERS
    enc = Encoder(OX.shape[1], M, BLOCKS, a.c, a.c0, k); ens = Ens([Mixture(M, a.K, a.J, a.eps) for _ in range(k)])
    for mx in ens.mixes: mx.pi = pi0
    opt = torch.optim.Adam(list(enc.parameters()) + ens.params(), lr=LR)
    kappa = 1.0; k_traj, f_traj = [], []

    def loss_fn(xb, yb):
        """Classification term (cross-entropy with weight kappa on normal flows) + LAM x density term, averaged over the members."""
        Z = enc(xb); ens.set_open(Z); yf = yb.float(); tot = 0.0; us = []
        for i, mx in enumerate(ens.mixes):
            lN_, lA_ = mx.logp(Z[:, i]); u = (lA_ - lN_) + logit_pi0
            ce = (yf * F.softplus(-u) + kappa * (1 - yf) * F.softplus(u)).mean(); nll = -(yf * lA_ + (1 - yf) * lN_).mean()
            tot = tot + ce + LAM * nll; us.append(u)
        return tot / k, torch.stack(us).mean(0)

    # ── stage 1, joint training on the training flows; kappa is fed back from the normal training flows after every epoch ──
    t0 = time.time(); n_tr = len(XT); nbatch = max(1, n_tr // BS); total_steps = a.epochs * nbatch; step = 0
    def kmeans_init():
        from sklearn.cluster import KMeans
        with torch.no_grad():
            Z = latents(enc, XT).numpy()
            for i, mx in enumerate(ens.mixes):
                for cen, cls, kk in ((mx.mu, 0, a.K), (mx.nu, 1, a.J)):
                    cen.data.copy_(torch.from_numpy(KMeans(kk, n_init=1, random_state=s + 1000 * i).fit(Z[(YT == cls).numpy(), i]).cluster_centers_).float())
    for ep in range(a.epochs):
        perm = torch.from_numpy(rng.permutation(n_tr))
        for i in range(nbatch):
            for g_ in opt.param_groups: g_["lr"] = LR * 0.5 * (1 + math.cos(math.pi * step / total_steps))
            bi = perm[i * BS:(i + 1) * BS]; loss, _ = loss_fn(XT[bi], YT[bi])
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
        if ep == 0: kmeans_init()
        if ep + 1 >= KAPPA_WARM:                                                                        # the cost-weight feedback towards the target FPR alpha
            with torch.no_grad():
                ens.set_open(latents(enc, XT)); fhat = float(((ens.score(latents(enc, XT[YT == 0])) + logit_pi0) > 0).float().mean())
            kappa = float(np.clip(kappa * math.exp(np.clip((fhat - a.alpha) / a.alpha, -1, 1)), 1.0, KAPPA_MAX))
            k_traj.append(round(kappa, 3)); f_traj.append(round(fhat, 4))
    sec_train = time.time() - t0
    # ── stage 2, refit: the encoder is frozen and the mixtures are refitted to the frozen latents of the training flows by a labelled EM ──
    enc.eval()
    for p_ in enc.parameters(): p_.requires_grad_(False)
    ZT = latents(enc, XT); ens.set_open(ZT); S0 = []
    with torch.no_grad():
        for i, mx in enumerate(ens.mixes):
            z = ZT[:, i]
            for it in range(100):                                                                 # labelled batch EM on the frozen latents (at most 100 iterations)
                old = torch.cat([mx.mu.data.flatten(), mx.nu.data.flatten()]).clone()
                S = mx.suff(z, mx.resp_within(z, YT)); mx.mstep(S)
                if float((torch.cat([mx.mu.data.flatten(), mx.nu.data.flatten()]) - old).abs().max()) < 1e-6: break
            info["refit_iters"] = it + 1
            S0.append({kk: v.clone() for kk, v in S.items()})
    S_run = [{kk: v.clone() for kk, v in S.items()} for S in S0]                                    # running statistics of the attack component means
    info.update(sigmas=[round(v, 3) for v in enc.sigmas()], var_o=float(np.mean([float(mx.var_o) for mx in ens.mixes])), pi0=pi0, kappa_final=kappa,
                kappa_traj=k_traj, fhat_traj=f_traj, sec_train=round(sec_train, 1))
    ZB = latents(enc, B_ref)
    # ── read-only test evaluation with the current psi_t (touches no parameter, statistic or random number) ──
    curve = []; ZTE_cache = None
    def curve_eval(t):
        nonlocal ZTE_cache
        with torch.no_grad():
            if ZTE_cache is None or a.enc_online: ZTE_cache = latents(enc, XTE)
            s_t = ens.score(ZTE_cache).numpy(); r_t = np.sort(ens.score(ZB).numpy())
        row = dict(round=t, auc=float(roc_auc_score(y_te, s_t)), pauc={str(b): pauc(s_t, y_te, b) for b in (0.01, 0.02, 0.05, 0.10, 0.20)})
        for al in sorted({a.alpha, 0.05, 0.2}):                                                             # the rule's level and two looser ones
            tau_q = conformal_threshold(r_t, al); row[f"q{al}"] = dict(metrics(s_t, y_te, tau_q), tau=tau_q)
        curve.append(row)
    # ── the data stream, batch by batch: (1) score and classify, (2) the flows classified as attacks move the attack component means,
    #    (3) the threshold is recalculated on the calibration set at the next batch ──
    rounds = []; t_0 = n_tr / INTERVAL
    if a.mode == "online":
        if a.eval_every: curve_eval(0)
        for t, i in enumerate(range(0, len(SX), INTERVAL), start=1):
            cx, cy, cf = SX[i:i + INTERVAL], sy[i:i + INTERVAL], sfam[i:i + INTERVAL]
            with torch.no_grad():
                zc = latents(enc, cx); s_ref = np.sort(ens.score(ZB).numpy()); s_c = ens.score(zc).numpy()             # scores with the current model
                rej = s_c > conformal_threshold(s_ref, a.alpha)                                                        # above the (1-alpha) quantile of the calibration scores = attack
                tU = torch.from_numpy(rej); update = dict(n=int(rej.sum()), fdr=float((rej & (cy == 0)).sum() / max(rej.sum(), 1)))
                g = 1.0 / (t + t_0)
                for j, mx in enumerate(ens.mixes):                                                                     # then the online update, per member: only the flows classified as attacks
                    r = mx.resp(zc[:, j])
                    rm = torch.zeros_like(r)
                    if rej.any(): rm[tU] = mx.attack_side(r[tU])
                    Sb = mx.suff(zc[:, j], rm)
                    for kk in S_run[j]: S_run[j][kk] = S_run[j][kk] + g * (Sb[kk] - S_run[j][kk])
                    if a.centre_update: mx.mstep(S_run[j], weights=False, benign_centres=False)
            if a.enc_online:                                                                                           # the variant: the data stream also retrains the encoder
                for p_ in enc.parameters(): p_.requires_grad_(True)
                enc.train(); opt2 = torch.optim.Adam(enc.parameters(), lr=LR_ONLINE)
                with torch.no_grad(): rs = [mx.resp(zc[:, j]).float() for j, mx in enumerate(ens.mixes)]
                for _ in range(E1):
                    perm = torch.from_numpy(rng.permutation(len(cx)))
                    for jj in range(0, len(cx), BS):
                        bi = perm[jj:jj + BS]; Z = enc(cx[bi]); l_em = 0.0
                        for j, mx in enumerate(ens.mixes):
                            cen = torch.cat([mx.mu.data, mx.nu.data]); rb = rs[j][bi]; KJ = a.K + a.J
                            l_em = l_em + (rb[:, :KJ] * mdist(Z[:, j].double(), cen.double(), torch.ones(M, dtype=torch.float64)) / 2).sum(1).mean() + (rb[:, mx.iO] * (Z[:, j] - mx.c_o).pow(2).sum(1) / (2 * mx.var_o)).mean()
                        ri = torch.from_numpy(rng.integers(0, n_tr, len(bi))); l_rep, _ = loss_fn(XT[ri], YT[ri])
                        opt2.zero_grad(); (l_em / k + l_rep).backward(); opt2.step()
                enc.eval()
                for p_ in enc.parameters(): p_.requires_grad_(False)
                ZB = latents(enc, B_ref); ens.set_open(latents(enc, XT))
            def stat(rj):
                n_alert = int(rj.sum()); fd = int((rj & (cy == 0)).sum())
                return dict(alerts=n_alert, fdr=(fd / n_alert if n_alert else 0.0), power=float(rj[cy == 1].mean()) if (cy == 1).any() else np.nan,
                            fpr=float(rj[cy == 0].mean()) if (cy == 0).any() else np.nan)
            rd = dict(round=t, n=int(len(cx)), attack_frac=float(cy.mean()), alerts=stat(rej), update=update)
            if a.lofo:
                mf = (cf == a.lofo) & (cy == 1); rd["lofo_n"] = int(mf.sum()); rd["lofo_rec"] = float(rej[mf].mean()) if mf.any() else np.nan
                rd["lofo_open_resp"] = float(np.mean([float(mx.resp(zc[torch.from_numpy(mf), j])[:, mx.iO].mean()) for j, mx in enumerate(ens.mixes)])) if mf.any() else np.nan
            rounds.append(rd)
            if a.eval_every and (t % a.eval_every == 0 or i + INTERVAL >= len(SX)): curve_eval(t)
    sec = time.time() - t0
    # ── evaluation on the test set ──
    with torch.no_grad():
        s_te = ens.score(latents(enc, XTE)).numpy(); s_ref = np.sort(ens.score(ZB).numpy())
    out = {"seed": s, "sec": round(sec, 1), "n_ref": int(len(B_ref)), "auc": float(roc_auc_score(y_te, s_te)),
           "pauc": {str(b): pauc(s_te, y_te, b) for b in (0.01, 0.02, 0.05, 0.10, 0.20)}, "info": info}
    for al in ALPHAS:
        tau = conformal_threshold(s_ref, al); mc = metrics(s_te, y_te, tau); out[f"a{al}"] = dict(mc, p_alpha=mc["fpr"] / al, tau=tau)
    if curve: out["curve"] = curve
    if rounds:
        out["rounds"] = rounds
        out["fpr_mean"] = float(np.mean([r["alerts"]["fpr"] for r in rounds])); out["fdp_mean"] = float(np.mean([r["alerts"]["fdr"] for r in rounds]))
        out["power_mean"] = float(np.nanmean([r["alerts"]["power"] for r in rounds])); out["fpr_viol"] = float(np.mean([r["alerts"]["fpr"] > a.alpha for r in rounds]))
        out["update_fdp_mean"] = float(np.mean([r["update"]["fdr"] for r in rounds]))
    if a.dump:
        d = Path(a.dump); d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f"{a.tag}_s{s}.npz", S_te=s_te.astype(np.float32), yte=y_te.astype(np.int8), fam_te=fam_te)
    return out

def main():
    x_tr, y_tr, fam_tr, x_te, y_te, fam_te = load(a.dataset); runs = []
    for i in range(a.seeds):
        r = run_seed(a.seed0 + i, x_tr, y_tr, fam_tr, x_te, y_te, fam_te); runs.append(r)
        extra = f"  stream FPR {r['fpr_mean']:.3f} FDP {r['fdp_mean']:.3f} power {r['power_mean']:.3f}" if "rounds" in r else ""
        print(f"  seed {r['seed']}  AUC {100*r['auc']:.2f}  pAUC@1% {r['pauc']['0.01']:.2f} @20% {r['pauc']['0.2']:.2f}  F1@a0.05 {r['a0.05']['f1']:.2f} (FPR {100*r['a0.05']['fpr']:.1f}%){extra}  {r['sec']}s", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump({"args": vars(a), "runs": runs}, open(Path(a.out) / f"{a.tag}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
