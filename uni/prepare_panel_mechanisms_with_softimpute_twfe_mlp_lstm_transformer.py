import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------
# Masking mechanisms
# ---------------------------

def make_mcar_mask(X, p, seed=0):
    """MCAR: missing completely at random."""
    rng = np.random.default_rng(seed)
    obs = np.argwhere(~np.isnan(X))
    k = int(len(obs) * p)
    if k <= 0:
        return np.zeros_like(X, dtype=bool)
    test_idx = obs[rng.choice(len(obs), size=k, replace=False)]
    mask = np.zeros_like(X, dtype=bool)
    mask[test_idx[:, 0], test_idx[:, 1]] = True
    return mask


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _standardize(v):
    v = v.astype(float)
    m = np.nanmean(v)
    s = np.nanstd(v)
    if s < 1e-12:
        return np.zeros_like(v)
    return (v - m) / s


def _calibrate_intercept(z, target_p, beta=1.0, lo=-20.0, hi=20.0, iters=60):
    """Pick a0 so that mean(sigmoid(a0 + beta*z)) ≈ target_p."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        p_mid = _sigmoid(mid + beta * z).mean()
        if p_mid < target_p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_mar_mask(X, p, seed=0, beta=1.5):
    """
    MAR: missingness depends on an OBSERVABLE proxy.

    Proxy used: row mean over observed years (country average level).
    Cells in countries with higher/lower proxy can be made more likely missing
    depending on sign of beta.
    """
    rng = np.random.default_rng(seed)
    obs = np.argwhere(~np.isnan(X))
    if len(obs) == 0:
        return np.zeros_like(X, dtype=bool)

    row_mean = np.nanmean(X, axis=1)  # observable country level proxy
    z = _standardize(row_mean[obs[:, 0]])
    a0 = _calibrate_intercept(z, target_p=p, beta=beta)

    prob = _sigmoid(a0 + beta * z)
    chosen = rng.random(size=len(obs)) < prob

    mask = np.zeros_like(X, dtype=bool)
    idx = obs[chosen]
    mask[idx[:, 0], idx[:, 1]] = True
    return mask


def make_mnar_mask(X_true, p, seed=0, beta=1.5, tail="low"):
    """
    MNAR: missingness depends on the TRUE value in the cell.

    tail:
      - "low"  : low values more likely missing
      - "high" : high values more likely missing
      - "both" : extremes more likely missing
    """
    rng = np.random.default_rng(seed)
    obs = np.argwhere(~np.isnan(X_true))
    if len(obs) == 0:
        return np.zeros_like(X_true, dtype=bool)

    vals = X_true[obs[:, 0], obs[:, 1]]
    z = _standardize(vals)

    if tail == "low":
        score = -z
    elif tail == "high":
        score = z
    elif tail == "both":
        score = np.abs(z)
    else:
        raise ValueError("tail must be one of: low, high, both")

    a0 = _calibrate_intercept(score, target_p=p, beta=beta)
    prob = _sigmoid(a0 + beta * score)

    chosen = rng.random(size=len(obs)) < prob

    mask = np.zeros_like(X_true, dtype=bool)
    idx = obs[chosen]
    mask[idx[:, 0], idx[:, 1]] = True
    return mask


def make_block_mask(X, p, seed=0, avg_block_len=5):
    """Block missingness: contiguous missing years for some countries."""
    rng = np.random.default_rng(seed)
    N, T = X.shape
    mask = np.zeros_like(X, dtype=bool)

    obs_cells = np.argwhere(~np.isnan(X))
    total_obs = len(obs_cells)
    target = int(total_obs * p)
    if target <= 0:
        return mask

    current = 0
    while current < target:
        i = rng.integers(0, N)
        L = max(1, int(rng.poisson(avg_block_len)))
        t0 = rng.integers(0, max(1, T - L + 1))
        t1 = min(T, t0 + L)

        block_obs = (~np.isnan(X[i, t0:t1])) & (~mask[i, t0:t1])
        add = int(block_obs.sum())
        mask[i, t0:t1][block_obs] = True
        current += add

    return mask


# ---------------------------
# Imputation methods
# ---------------------------

def mean_by_year_impute(X):
    """Column-wise (year-wise) mean imputation."""
    Xhat = X.copy()
    col_means = np.nanmean(Xhat, axis=0)
    inds = np.where(np.isnan(Xhat))
    Xhat[inds] = col_means[inds[1]]
    return Xhat


def iterative_svd_impute(X, r=3, n_iter=50, tol=1e-6):
    """Iterative low-rank imputation using truncated SVD."""
    Xhat = mean_by_year_impute(X)
    observed = ~np.isnan(X)

    for _ in range(n_iter):
        U, s, Vt = np.linalg.svd(Xhat, full_matrices=False)
        s[r:] = 0.0
        Xnew = (U * s) @ Vt
        Xnew[observed] = X[observed]

        err = np.linalg.norm(Xnew - Xhat) / (np.linalg.norm(Xhat) + 1e-12)
        Xhat = Xnew
        if err < tol:
            break

    return Xhat




def twfe_impute(X, n_iter=30, tol=1e-7):
    """
    Two-way fixed effects imputation:
        x_it ≈ alpha_i + gamma_t

    Iterative version: alternates estimating alpha (row effects) and gamma (col effects)
    using current filled matrix, while keeping observed entries fixed.
    """
    Xhat = mean_by_year_impute(X)
    observed = ~np.isnan(X)

    for _ in range(n_iter):
        X_prev = Xhat.copy()

        # Row effects (country)
        alpha = np.nanmean(Xhat, axis=1, keepdims=True)

        # Column effects (year), on residuals
        gamma = np.nanmean(Xhat - alpha, axis=0, keepdims=True)

        Xnew = alpha + gamma
        Xnew[observed] = X[observed]

        err = np.linalg.norm(Xnew - X_prev) / (np.linalg.norm(X_prev) + 1e-12)
        Xhat = Xnew
        if err < tol:
            break

    return Xhat


def soft_impute(X, lam=None, max_rank=None, n_iter=100, tol=1e-6):
    """
    SoftImpute (nuclear norm regularization) for matrix completion.

    Solves (approximately):
        min_Z 0.5 ||P_Ω(X - Z)||_F^2 + lam * ||Z||_*
    via iterative singular value thresholding.

    Parameters
    ----------
    lam : float
        Shrinkage parameter (λ). Larger -> lower effective rank (more regularization).
        If None, a data-driven default is used.
    max_rank : int or None
        Optional cap on rank during SVD reconstruction (speed). If None, full.
    """
    Xhat = mean_by_year_impute(X)
    observed = ~np.isnan(X)

    # data-driven default lambda: proportional to top singular value
    if lam is None:
        U0, s0, Vt0 = np.linalg.svd(Xhat, full_matrices=False)
        lam = 0.2 * s0[0]  # conservative default; tune if needed

    for _ in range(n_iter):
        X_prev = Xhat.copy()

        U, s, Vt = np.linalg.svd(Xhat, full_matrices=False)
        s_shrunk = np.maximum(s - lam, 0.0)

        if max_rank is not None:
            r = min(max_rank, len(s_shrunk))
            U = U[:, :r]
            Vt = Vt[:r, :]
            s_shrunk = s_shrunk[:r]

        Xnew = (U * s_shrunk) @ Vt
        Xnew[observed] = X[observed]

        err = np.linalg.norm(Xnew - X_prev) / (np.linalg.norm(X_prev) + 1e-12)
        Xhat = Xnew
        if err < tol:
            break

    return Xhat


def eval_metrics(X_true, X_hat, mask):
    diff = (X_hat - X_true)[mask]
    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))
    return rmse, mae


# ---------------------------
# MLP (feed-forward neural network) imputation
# ---------------------------

class _PanelDataset(Dataset):
    def __init__(self, X_in, X_true, mask_loss, mu, sigma):
        # Standardize (for training stability); metrics are computed in original (log) scale.
        self.X_in = torch.tensor((X_in - mu) / sigma, dtype=torch.float32)
        self.X_true = torch.tensor((X_true - mu) / sigma, dtype=torch.float32)
        self.mask = torch.tensor(mask_loss, dtype=torch.bool)

    def __len__(self):
        return self.X_true.shape[0]

    def __getitem__(self, idx):
        return self.X_in[idx], self.X_true[idx], self.mask[idx]


class MLPImputer(nn.Module):
    def __init__(self, d, hidden=(128, 64), dropout=0.1):
        super().__init__()
        layers = []
        prev = d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, d)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class BiLSTMImputer(nn.Module):
    """Bidirectional LSTM imputer over years (sequence length T=21).

    Input per timestep: [value, missing_flag]
      - value: standardized log-GDP after mean-by-year imputation
      - missing_flag: 1.0 if the value was artificially masked, else 0.0

    Output per timestep: predicted standardized value.
    """

    def __init__(self, hidden_size=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):  # x: (B, T, 2)
        h, _ = self.lstm(x)     # (B, T, 2H)
        y = self.head(h)        # (B, T, 1)
        return y.squeeze(-1)    # (B, T)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, T, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, d_model)
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TransformerImputer(nn.Module):
    """
    Transformer imputer over years.

    Input per timestep: [value, missing_flag]
    Output per timestep: predicted standardized value.
    """

    def __init__(
        self,
        input_dim=2,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=512)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        # x: (B, T, 2)
        h = self.input_proj(x)      # (B, T, d_model)
        h = self.pos_encoder(h)     # (B, T, d_model)
        h = self.encoder(h)         # (B, T, d_model)
        y = self.head(h)            # (B, T, 1)
        return y.squeeze(-1)        # (B, T)


class _SeqDataset(Dataset):
    def __init__(self, X_in, X_true, mask, mu, sigma):
        X_in_std = (X_in - mu) / sigma
        X_true_std = (X_true - mu) / sigma
        miss = mask.astype(np.float32)

        # sequence input: (value, missing_flag)
        self.X_seq = torch.tensor(np.stack([X_in_std, miss], axis=-1), dtype=torch.float32)  # (N,T,2)
        self.Y = torch.tensor(X_true_std, dtype=torch.float32)                               # (N,T)
        self.M = torch.tensor(mask, dtype=torch.bool)                                        # (N,T)

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return self.X_seq[idx], self.Y[idx], self.M[idx]



def _masked_mse_torch(y_pred, y_true, mask):
    diff2 = (y_pred - y_true) ** 2
    diff2 = diff2[mask]
    if diff2.numel() == 0:
        return torch.tensor(0.0, device=y_pred.device)
    return diff2.mean()


def _mlp_fit_predict_out_of_sample(
    X_true,
    X_masked,
    mask_train_loss,
    train_rows,
    val_rows,
    seed=0,
    epochs=200,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    patience=30,
    hidden=(128, 64),
    dropout=0.1,
    device=None,
):
    '''
    Train MLP on TRAIN rows only, with supervised loss ONLY on artificially masked cells in TRAIN.
    Evaluate/predict on VAL rows (no leakage).

    Returns:
      X_hat_log for VAL rows only (same shape as X_true[val_rows])
    '''
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Input is mean-imputed masked matrix (same baseline preprocessing as other methods)
    X_in = mean_by_year_impute(X_masked)

    Xtr_in = X_in[train_rows]
    Xtr_true = X_true[train_rows]
    Mtr = mask_train_loss[train_rows]

    Xva_in = X_in[val_rows]
    Xva_true = X_true[val_rows]
    Mva = mask_train_loss[val_rows]  # not used for training, only to keep shapes consistent

    # Standardize using TRAIN inputs (in log scale)
    mu = Xtr_in.mean(axis=0, keepdims=True)
    sigma = Xtr_in.std(axis=0, keepdims=True) + 1e-6

    ds_tr = _PanelDataset(Xtr_in, Xtr_true, Mtr, mu, sigma)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

    # We will predict on VAL in one forward pass (no loss)
    Xva_in_std = torch.tensor((Xva_in - mu) / sigma, dtype=torch.float32)

    torch.manual_seed(seed)
    model = MLPImputer(d=X_true.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = float("inf")
    best_state = None
    bad = 0

    for _ep in range(1, epochs + 1):
        model.train()
        for x_in_b, x_true_b, m_b in dl_tr:
            x_in_b = x_in_b.to(device)
            x_true_b = x_true_b.to(device)
            m_b = m_b.to(device)

            opt.zero_grad()
            y_b = model(x_in_b)
            loss = _masked_mse_torch(y_b, x_true_b, m_b)
            if torch.isnan(loss):
                raise RuntimeError("MLP loss is NaN. Check that X_true has no NaNs after preprocessing.")
            loss.backward()
            opt.step()

        # cheap validation on TRAIN masked cells (early stopping heuristic)
        model.eval()
        with torch.no_grad():
            # compute train RMSE in standardized scale
            train_mses = []
            for x_in_b, x_true_b, m_b in dl_tr:
                x_in_b = x_in_b.to(device)
                x_true_b = x_true_b.to(device)
                m_b = m_b.to(device)
                y_b = model(x_in_b)
                train_mses.append(_masked_mse_torch(y_b, x_true_b, m_b).item())
            rmse_tr_std = math.sqrt(float(np.mean(train_mses))) if len(train_mses) else float("inf")

        if rmse_tr_std < best - 1e-7:
            best = rmse_tr_std
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict on VAL rows, then de-standardize back to log scale
    model.eval()
    with torch.no_grad():
        Yhat_std = model(Xva_in_std.to(device)).cpu().numpy()
    Yhat_log = Yhat_std * sigma + mu
    return Yhat_log




def _lstm_fit_predict_out_of_sample(
    X_true,
    X_masked,
    mask_train_loss,
    train_rows,
    val_rows,
    seed=0,
    epochs=200,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    patience=30,
    hidden_size=64,
    num_layers=1,
    dropout=0.1,
    device=None,
):
    '''
    Train BiLSTM on TRAIN rows only, with supervised loss ONLY on artificially masked cells in TRAIN.
    Evaluate/predict on VAL rows, returning predictions in LOG scale for VAL rows only.
    '''
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # input = mean-by-year imputation (same baseline as other methods)
    X_in = mean_by_year_impute(X_masked)

    # split matrices
    Xtr_true = X_true[train_rows]
    Xva_true = X_true[val_rows]
    Xtr_in = X_in[train_rows]
    Xva_in = X_in[val_rows]

    mask_tr = mask_train_loss[train_rows].copy()
    mask_va = mask_train_loss[val_rows].copy()

    # standardize using TRAIN inputs (log scale)
    mu = Xtr_in.mean(axis=0, keepdims=True)
    sigma = Xtr_in.std(axis=0, keepdims=True) + 1e-6

    ds_tr = _SeqDataset(Xtr_in, Xtr_true, mask_tr, mu, sigma)
    ds_va = _SeqDataset(Xva_in, Xva_true, mask_va, mu, sigma)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False)

    torch.manual_seed(seed)
    model = BiLSTMImputer(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = float("inf")
    best_state = None
    bad = 0

    for _ in range(epochs):
        model.train()
        for x_seq_b, y_true_b, m_b in dl_tr:
            x_seq_b = x_seq_b.to(device)
            y_true_b = y_true_b.to(device)
            m_b = m_b.to(device)

            opt.zero_grad()
            y_b = model(x_seq_b)
            loss = _masked_mse_torch(y_b, y_true_b, m_b)
            if torch.isnan(loss):
                raise RuntimeError("LSTM loss is NaN. Check preprocessing and inputs.")
            loss.backward()
            opt.step()

        # cheap early stopping on TRAIN masked cells (same heuristic as MLP)
        model.eval()
        with torch.no_grad():
            train_mses = []
            for x_seq_b, y_true_b, m_b in dl_tr:
                x_seq_b = x_seq_b.to(device)
                y_true_b = y_true_b.to(device)
                m_b = m_b.to(device)
                y_b = model(x_seq_b)
                train_mses.append(_masked_mse_torch(y_b, y_true_b, m_b).item())
            rmse_tr_std = math.sqrt(float(np.mean(train_mses))) if len(train_mses) else float("inf")

        if rmse_tr_std < best - 1e-7:
            best = rmse_tr_std
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # predict on VAL rows, de-standardize back to log scale
    model.eval()
    with torch.no_grad():
        # Build VAL input sequence directly (no dataloader)
        Xva_in_std = (Xva_in - mu) / sigma
        miss_va = mask_va.astype(np.float32)
        Xva_seq = torch.tensor(np.stack([Xva_in_std, miss_va], axis=-1), dtype=torch.float32).to(device)  # (Nva,T,2)
        Yhat_std = model(Xva_seq).cpu().numpy()
    Yhat_log = Yhat_std * sigma + mu
    return Yhat_log


def _transformer_fit_predict_out_of_sample(
    X_true,
    X_masked,
    mask_train_loss,
    train_rows,
    val_rows,
    seed=0,
    epochs=200,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    patience=30,
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    device=None,
):
    """
    Train Transformer on TRAIN rows only, with supervised loss ONLY on artificially masked cells in TRAIN.
    Evaluate/predict on VAL rows, returning predictions in LOG scale for VAL rows only.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # input = mean-by-year imputation
    X_in = mean_by_year_impute(X_masked)

    # split matrices
    Xtr_true = X_true[train_rows]
    Xva_true = X_true[val_rows]
    Xtr_in = X_in[train_rows]
    Xva_in = X_in[val_rows]

    mask_tr = mask_train_loss[train_rows].copy()
    mask_va = mask_train_loss[val_rows].copy()

    # standardize using TRAIN inputs (log scale)
    mu = Xtr_in.mean(axis=0, keepdims=True)
    sigma = Xtr_in.std(axis=0, keepdims=True) + 1e-6

    ds_tr = _SeqDataset(Xtr_in, Xtr_true, mask_tr, mu, sigma)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = TransformerImputer(
        input_dim=2,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = float("inf")
    best_state = None
    bad = 0

    for _ in range(epochs):
        model.train()
        for x_seq_b, y_true_b, m_b in dl_tr:
            x_seq_b = x_seq_b.to(device)
            y_true_b = y_true_b.to(device)
            m_b = m_b.to(device)

            opt.zero_grad()
            y_b = model(x_seq_b)
            loss = _masked_mse_torch(y_b, y_true_b, m_b)
            if torch.isnan(loss):
                raise RuntimeError("Transformer loss is NaN. Check preprocessing and inputs.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        # cheap early stopping on TRAIN masked cells
        model.eval()
        with torch.no_grad():
            train_mses = []
            for x_seq_b, y_true_b, m_b in dl_tr:
                x_seq_b = x_seq_b.to(device)
                y_true_b = y_true_b.to(device)
                m_b = m_b.to(device)
                y_b = model(x_seq_b)
                train_mses.append(_masked_mse_torch(y_b, y_true_b, m_b).item())
            rmse_tr_std = math.sqrt(float(np.mean(train_mses))) if len(train_mses) else float("inf")

        if rmse_tr_std < best - 1e-7:
            best = rmse_tr_std
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # predict on VAL rows, de-standardize back to log scale
    model.eval()
    with torch.no_grad():
        Xva_in_std = (Xva_in - mu) / sigma
        miss_va = mask_va.astype(np.float32)
        Xva_seq = torch.tensor(np.stack([Xva_in_std, miss_va], axis=-1), dtype=torch.float32).to(device)  # (Nva,T,2)
        Yhat_std = model(Xva_seq).cpu().numpy()
    Yhat_log = Yhat_std * sigma + mu
    return Yhat_log


# ---------------------------
# Experiments
# ---------------------------

def _make_mask(X, p, seed, mechanism, mech_kwargs):
    mech_kwargs = mech_kwargs or {}

    if mechanism == "mcar":
        return make_mcar_mask(X, p=p, seed=seed)
    if mechanism == "mar":
        return make_mar_mask(X, p=p, seed=seed, **mech_kwargs)
    if mechanism == "mnar":
        # MNAR uses the true matrix values by design
        return make_mnar_mask(X, p=p, seed=seed, **mech_kwargs)
    if mechanism == "block":
        return make_block_mask(X, p=p, seed=seed, **mech_kwargs)

    raise ValueError("mechanism must be one of: mcar, mar, mnar, block")


def run_experiment(
    X,
    p_values=(0.1, 0.3, 0.5),
    seeds=range(5),
    rank=3,
    mechanism="mcar",
    mech_kwargs=None,
    softimpute_kwargs=None,
):
    """
    Runs masking + imputation experiment for a single mechanism.

    Returns a tidy table with mean RMSE/MAE (and RMSE std) across seeds for:
      - Mean (year-wise)
      - TWFE
      - Iterative SVD (rank=r)
      - Nuclear Norm (SoftImpute)
      - MLP (feed-forward neural network)
      - BiLSTM
      - Transformer
    """
    softimpute_kwargs = softimpute_kwargs or {}

    rows = []
    for p in p_values:
        rmse_mean_list, mae_mean_list = [], []
        rmse_twfe_list, mae_twfe_list = [], []
        rmse_svd_list,  mae_svd_list  = [], []
        rmse_nuc_list,  mae_nuc_list  = [], []
        rmse_mlp_list,  mae_mlp_list  = [], []
        rmse_lstm_list, mae_lstm_list = [], []
        rmse_trf_list,  mae_trf_list  = [], []

        for seed in seeds:
            mask = _make_mask(X, p=p, seed=seed, mechanism=mechanism, mech_kwargs=mech_kwargs)

            # row split (countries) for out-of-sample evaluation, same for all methods
            rng = np.random.default_rng(seed)
            N, T = X.shape
            idx = np.arange(N)
            rng.shuffle(idx)
            n_train = int(0.8 * N)
            train_rows = idx[:n_train]
            val_rows = idx[n_train:]

            # apply mask
            X_masked = X.copy()
            X_masked[mask] = np.nan

            # evaluation mask only on VAL rows (no leakage)
            mask_val = mask.copy()
            mask_val[train_rows, :] = False

            # Mean
            X_mean = mean_by_year_impute(X_masked)
            rmse_mean, mae_mean = eval_metrics(X, X_mean, mask_val)

            # TWFE
            X_twfe = twfe_impute(X_masked)
            rmse_twfe, mae_twfe = eval_metrics(X, X_twfe, mask_val)

            # Iterative SVD
            X_svd = iterative_svd_impute(X_masked, r=rank)
            rmse_svd, mae_svd = eval_metrics(X, X_svd, mask_val)

            # Nuclear norm (SoftImpute)
            X_nuc = soft_impute(X_masked, max_rank=rank, **softimpute_kwargs)
            rmse_nuc, mae_nuc = eval_metrics(X, X_nuc, mask_val)

            # MLP (trained on TRAIN rows only; evaluated on VAL rows only)
            try:
                Yhat_val = _mlp_fit_predict_out_of_sample(
                    X_true=X,
                    X_masked=X_masked,
                    mask_train_loss=mask,
                    train_rows=train_rows,
                    val_rows=val_rows,
                    seed=seed,
                    epochs=200,
                    batch_size=64,
                    lr=1e-3,
                    weight_decay=1e-4,
                    patience=30,
                )
                # Build full matrix for metric function (only val rows relevant)
                X_mlp = X.copy()
                X_mlp[val_rows, :] = Yhat_val
                rmse_mlp, mae_mlp = eval_metrics(X, X_mlp, mask_val)
            except Exception as e:
                raise RuntimeError(f"MLP failed for mechanism={mechanism}, p={p}, seed={seed}: {e}")


            # BiLSTM (trained on TRAIN rows only; evaluated on VAL rows only)
            try:
                Yhat_val_lstm = _lstm_fit_predict_out_of_sample(
                    X_true=X,
                    X_masked=X_masked,
                    mask_train_loss=mask,
                    train_rows=train_rows,
                    val_rows=val_rows,
                    seed=seed,
                    epochs=200,
                    batch_size=64,
                    lr=1e-3,
                    weight_decay=1e-4,
                    patience=30,
                    hidden_size=64,
                    num_layers=1,
                    dropout=0.1,
                )
                X_lstm = X.copy()
                X_lstm[val_rows, :] = Yhat_val_lstm
                rmse_lstm, mae_lstm = eval_metrics(X, X_lstm, mask_val)
            except Exception as e:
                raise RuntimeError(f"LSTM failed for mechanism={mechanism}, p={p}, seed={seed}: {e}")


            rmse_mean_list.append(rmse_mean); mae_mean_list.append(mae_mean)
            rmse_twfe_list.append(rmse_twfe); mae_twfe_list.append(mae_twfe)
            rmse_svd_list.append(rmse_svd);  mae_svd_list.append(mae_svd)
            rmse_nuc_list.append(rmse_nuc);  mae_nuc_list.append(mae_nuc)
            rmse_mlp_list.append(rmse_mlp);  mae_mlp_list.append(mae_mlp)
            rmse_lstm_list.append(rmse_lstm); mae_lstm_list.append(mae_lstm)

            # Transformer (trained on TRAIN rows only; evaluated on VAL rows only)
            try:
                Yhat_val_trf = _transformer_fit_predict_out_of_sample(
                    X_true=X,
                    X_masked=X_masked,
                    mask_train_loss=mask,
                    train_rows=train_rows,
                    val_rows=val_rows,
                    seed=seed,
                    epochs=200,
                    batch_size=64,
                    lr=1e-3,
                    weight_decay=1e-4,
                    patience=30,
                    d_model=64,
                    nhead=4,
                    num_layers=2,
                    dim_feedforward=128,
                    dropout=0.1,
                )
                X_trf = X.copy()
                X_trf[val_rows, :] = Yhat_val_trf
                rmse_trf, mae_trf = eval_metrics(X, X_trf, mask_val)
            except Exception as e:
                raise RuntimeError(f"Transformer failed for mechanism={mechanism}, p={p}, seed={seed}: {e}")

            rmse_trf_list.append(rmse_trf); mae_trf_list.append(mae_trf)

        rows.append({
            "mechanism": mechanism,
            "missing_fraction": p,

            "RMSE_mean": float(np.mean(rmse_mean_list)),
            "RMSE_twfe": float(np.mean(rmse_twfe_list)),
            "RMSE_svd":  float(np.mean(rmse_svd_list)),
            "RMSE_nuclear": float(np.mean(rmse_nuc_list)),
            "RMSE_mlp": float(np.mean(rmse_mlp_list)),
            "RMSE_lstm": float(np.mean(rmse_lstm_list)),
            "RMSE_transformer": float(np.mean(rmse_trf_list)),

            "MAE_mean": float(np.mean(mae_mean_list)),
            "MAE_twfe": float(np.mean(mae_twfe_list)),
            "MAE_svd":  float(np.mean(mae_svd_list)),
            "MAE_nuclear": float(np.mean(mae_nuc_list)),
            "MAE_mlp": float(np.mean(mae_mlp_list)),
            "MAE_lstm": float(np.mean(mae_lstm_list)),
            "MAE_transformer": float(np.mean(mae_trf_list)),

            "RMSE_mean_std": float(np.std(rmse_mean_list)),
            "RMSE_twfe_std": float(np.std(rmse_twfe_list)),
            "RMSE_svd_std":  float(np.std(rmse_svd_list)),
            "RMSE_nuclear_std": float(np.std(rmse_nuc_list)),
            "RMSE_mlp_std": float(np.std(rmse_mlp_list)),
            "RMSE_lstm_std": float(np.std(rmse_lstm_list)),
            "RMSE_transformer_std": float(np.std(rmse_trf_list)),
        })

    return pd.DataFrame(rows)


def run_all_mechanisms(X, p_values=(0.1, 0.3, 0.5), seeds=range(5), rank=3):
    dfs = []
    dfs.append(run_experiment(X, p_values, seeds, rank, mechanism="mcar"))
    dfs.append(run_experiment(X, p_values, seeds, rank, mechanism="mar", mech_kwargs={"beta": 2.0}))
    dfs.append(run_experiment(X, p_values, seeds, rank, mechanism="mnar", mech_kwargs={"beta": 2.0, "tail": "low"}))
    dfs.append(run_experiment(X, p_values, seeds, rank, mechanism="block", mech_kwargs={"avg_block_len": 6}))
    return pd.concat(dfs, ignore_index=True)


def plot_results(all_results, out_png="imputation_by_mechanism.png"):
    """One figure per mechanism (keeps plots readable)."""
    mechanisms = list(all_results["mechanism"].unique())

    for mech in mechanisms:
        dfm = all_results[all_results["mechanism"] == mech].sort_values("missing_fraction")

        plt.figure(figsize=(7, 4.5))
        plt.plot(dfm["missing_fraction"], dfm["RMSE_mean"], marker="o", label="Mean")
        plt.plot(dfm["missing_fraction"], dfm["RMSE_twfe"], marker="o", label="TWFE")
        plt.plot(dfm["missing_fraction"], dfm["RMSE_svd"], marker="o", label="Iterative SVD")
        plt.plot(dfm["missing_fraction"], dfm["RMSE_nuclear"], marker="o", label="Nuclear norm (SoftImpute)")
        if "RMSE_mlp" in dfm.columns:
            plt.plot(dfm["missing_fraction"], dfm["RMSE_mlp"], marker="o", label="MLP")
        if "RMSE_lstm" in dfm.columns:
            plt.plot(dfm["missing_fraction"], dfm["RMSE_lstm"], marker="o", label="BiLSTM")
        if "RMSE_transformer" in dfm.columns:
            plt.plot(dfm["missing_fraction"], dfm["RMSE_transformer"], marker="o", label="Transformer")

        plt.xlabel("Missing fraction")
        plt.ylabel("RMSE")
        plt.title(f"Imputation accuracy vs missingness ({mech.upper()})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        fname = out_png.replace(".png", f"_{mech}.png")
        plt.savefig(fname, dpi=200)
        plt.show()


# ---------------------------
# Main script (adjust the CSV name if needed)
# ---------------------------

if __name__ == "__main__":
    # Expect a wide CSV with columns for years: "2000" ... "2020"
    df = pd.read_csv("gdp_pc_2000_2020.csv")

    year_cols = [c for c in df.columns if c.isdigit()]
    if not year_cols:
        raise ValueError("No year columns found (expected columns like '2000', '2001', ...).")

    # For a clean 'ground truth' experiment, keep only complete rows for year columns
    df_full = df.dropna(subset=year_cols).copy()

    X = df_full[year_cols].to_numpy(float)

    # Optional: log transform GDP per capita (common for macro variables)
    X = np.log(X)

    print("Full panel shape (rows x years):", X.shape)
    print("Years:", min(year_cols), "...", max(year_cols))

    all_results = run_all_mechanisms(X, p_values=(0.1, 0.3, 0.5), seeds=range(10), rank=3)
    print("\nResults (mean over seeds):")
    print(all_results)

    all_results.to_csv("imputation_results_by_mechanism.csv", index=False)
    print("\nSaved table: imputation_results_by_mechanism.csv")

    plot_results(all_results, out_png="imputation_by_mechanism.png")