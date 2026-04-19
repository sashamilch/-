import os
import math
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEFAULT_DEVICE = get_default_device()
print("Using device:", DEFAULT_DEVICE)
if DEFAULT_DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# =========================================================
# 1. Data loading and preprocessing
# =========================================================

def load_pwt_csv(csv_path):
    df = pd.read_csv(csv_path)

    year_cols = [c for c in df.columns if str(c).isdigit()]
    if not year_cols:
        raise ValueError("No year columns found.")

    id_cols = ["ISO code", "Country", "Variable code", "Variable name"]
    missing_id = [c for c in id_cols if c not in df.columns]
    if missing_id:
        raise ValueError("Missing required columns: {0}".format(missing_id))

    long_df = df.melt(
        id_vars=id_cols,
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )

    long_df["year"] = long_df["year"].astype(int)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")

    long_df = long_df.rename(columns={
        "ISO code": "iso",
        "Country": "country",
        "Variable code": "var_code",
        "Variable name": "var_name",
    })
    return long_df


def filter_variables(long_df, variable_codes, start_year=1999, end_year=2023):
    out = long_df[
        (long_df["var_code"].isin(variable_codes)) &
        (long_df["year"] >= start_year) &
        (long_df["year"] <= end_year)
    ].copy()
    if out.empty:
        raise ValueError("No data left after filtering.")
    return out


def apply_variablewise_log(long_df, log_vars=None, use_log1p=False):
    out = long_df.copy()
    out["value_transformed"] = out["value"]

    if not log_vars:
        return out

    mask = out["var_code"].isin(log_vars)
    if use_log1p:
        good = mask & (out["value"] >= 0)
        out.loc[good, "value_transformed"] = np.log1p(out.loc[good, "value"])
        out.loc[mask & (out["value"] < 0), "value_transformed"] = np.nan
    else:
        good = mask & (out["value"] > 0)
        out.loc[good, "value_transformed"] = np.log(out.loc[good, "value"])
        out.loc[mask & (out["value"] <= 0), "value_transformed"] = np.nan
    return out


def fit_standardization_params(long_df, train_countries=None, train_years=None, value_col="value_transformed"):
    df = long_df.copy()
    if train_countries is not None:
        df = df[df["iso"].isin(train_countries)]
    if train_years is not None:
        df = df[df["year"].isin(train_years)]

    stats = (
        df.groupby("var_code")[value_col]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mu", "std": "sigma"})
    )
    stats["sigma"] = stats["sigma"].replace(0.0, np.nan).fillna(1.0)
    return stats


def apply_standardization(long_df, stats_df, value_col="value_transformed"):
    out = long_df.merge(stats_df, on="var_code", how="left")
    out["value_scaled"] = (out[value_col] - out["mu"]) / out["sigma"]
    return out


def build_balanced_tensor(long_df, variables, value_col="value_scaled", min_nonmissing_share=0.8):
    matrices = {}
    for var in variables:
        sub = long_df[long_df["var_code"] == var].copy()
        X = sub.pivot_table(index="iso", columns="year", values=value_col, aggfunc="mean")
        X = X.sort_index(axis=0).sort_index(axis=1)
        row_share = X.notna().mean(axis=1)
        X = X.loc[row_share >= min_nonmissing_share]
        matrices[var] = X

    common_countries = sorted(set.intersection(*[set(m.index) for m in matrices.values()]))
    common_years = sorted(set.intersection(*[set(m.columns) for m in matrices.values()]))
    if not common_countries or not common_years:
        raise ValueError("No common countries/years across selected variables.")

    arrs = []
    for var in variables:
        Xi = matrices[var].loc[common_countries, common_years]
        arrs.append(Xi.to_numpy(dtype=float))
    tensor = np.stack(arrs, axis=-1)  # (N, T, K)

    return {
        "tensor": tensor,
        "countries": common_countries,
        "years": common_years,
        "variables": list(variables),
        "matrices": matrices,
    }

# =========================================================
# 2. Missingness mechanisms on tensors
# =========================================================

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
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        p_mid = _sigmoid(mid + beta * z).mean()
        if p_mid < target_p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_mcar_mask_tensor(X, p, seed=0):
    rng = np.random.default_rng(seed)
    obs = np.argwhere(~np.isnan(X))
    k = int(len(obs) * p)
    mask = np.zeros_like(X, dtype=bool)
    if k <= 0 or len(obs) == 0:
        return mask
    chosen = obs[rng.choice(len(obs), size=k, replace=False)]
    mask[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True
    return mask


def make_mar_mask_tensor(X, p, seed=0, beta=1.5):
    rng = np.random.default_rng(seed)
    obs = np.argwhere(~np.isnan(X))
    mask = np.zeros_like(X, dtype=bool)
    if len(obs) == 0:
        return mask

    # observable proxy: row-time average across variables
    proxy = np.nanmean(X, axis=2)  # (N, T)
    z = _standardize(proxy[obs[:, 0], obs[:, 1]])
    a0 = _calibrate_intercept(z, target_p=p, beta=beta)
    prob = _sigmoid(a0 + beta * z)
    chosen = rng.random(size=len(obs)) < prob
    idx = obs[chosen]
    mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return mask


def make_mnar_mask_tensor(X, p, seed=0, beta=1.5, tail="low"):
    rng = np.random.default_rng(seed)
    obs = np.argwhere(~np.isnan(X))
    mask = np.zeros_like(X, dtype=bool)
    if len(obs) == 0:
        return mask

    vals = X[obs[:, 0], obs[:, 1], obs[:, 2]]
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
    idx = obs[chosen]
    mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return mask


def make_block_mask_tensor(X, p, seed=0, avg_block_len=5):
    rng = np.random.default_rng(seed)
    N, T, K = X.shape
    mask = np.zeros_like(X, dtype=bool)

    obs_cells = np.argwhere(~np.isnan(X))
    total_obs = len(obs_cells)
    target = int(total_obs * p)
    if target <= 0:
        return mask

    current = 0
    while current < target:
        i = rng.integers(0, N)
        k = rng.integers(0, K)
        L = max(1, int(rng.poisson(avg_block_len)))
        t0 = rng.integers(0, max(1, T - L + 1))
        t1 = min(T, t0 + L)
        block_obs = (~np.isnan(X[i, t0:t1, k])) & (~mask[i, t0:t1, k])
        add = int(block_obs.sum())
        mask[i, t0:t1, k][block_obs] = True
        current += add
    return mask


def make_mask_tensor(X, p, seed, mechanism, mech_kwargs=None):
    mech_kwargs = mech_kwargs or {}
    if mechanism == "mcar":
        return make_mcar_mask_tensor(X, p=p, seed=seed)
    if mechanism == "mar":
        return make_mar_mask_tensor(X, p=p, seed=seed, **mech_kwargs)
    if mechanism == "mnar":
        return make_mnar_mask_tensor(X, p=p, seed=seed, **mech_kwargs)
    if mechanism == "block":
        return make_block_mask_tensor(X, p=p, seed=seed, **mech_kwargs)
    raise ValueError("mechanism must be one of: mcar, mar, mnar, block")

# =========================================================
# 3. Classical baselines on joint matrix
# =========================================================

def flatten_tensor_to_joint_matrix(X):
    N, T, K = X.shape
    X_joint = np.transpose(X, (0, 2, 1)).reshape(N, K * T)
    return X_joint


def unflatten_joint_matrix_to_tensor(X_joint, T, K):
    N = X_joint.shape[0]
    X = X_joint.reshape(N, K, T)
    X = np.transpose(X, (0, 2, 1))
    return X


def safe_mean_fill_cols(X):
    Xhat = X.copy().astype(float)
    col_means = np.nanmean(Xhat, axis=0)
    global_mean = np.nanmean(Xhat)
    if np.isnan(global_mean):
        global_mean = 0.0
    col_means = np.where(np.isnan(col_means), global_mean, col_means)
    inds = np.where(np.isnan(Xhat))
    Xhat[inds] = col_means[inds[1]]
    Xhat = np.where(np.isnan(Xhat), global_mean, Xhat)
    return Xhat


def mean_joint_impute(X_joint):
    return safe_mean_fill_cols(X_joint)


def twfe_joint_impute(X_joint, n_iter=30, tol=1e-7):
    Xhat = safe_mean_fill_cols(X_joint)
    observed = ~np.isnan(X_joint)
    global_mean = np.nanmean(Xhat)
    if np.isnan(global_mean):
        global_mean = 0.0

    for _ in range(n_iter):
        X_prev = Xhat.copy()
        alpha = np.nanmean(Xhat, axis=1, keepdims=True)
        alpha = np.where(np.isnan(alpha), global_mean, alpha)
        gamma = np.nanmean(Xhat - alpha, axis=0, keepdims=True)
        gamma = np.where(np.isnan(gamma), 0.0, gamma)
        Xnew = alpha + gamma
        Xnew[observed] = X_joint[observed]
        Xnew = np.where(np.isnan(Xnew), global_mean, Xnew)
        err = np.linalg.norm(Xnew - X_prev) / (np.linalg.norm(X_prev) + 1e-12)
        Xhat = Xnew
        if err < tol:
            break
    return Xhat


def iterative_svd_joint_impute(X_joint, r=3, n_iter=50, tol=1e-6):
    Xhat = safe_mean_fill_cols(X_joint)
    observed = ~np.isnan(X_joint)
    max_rank = max(1, min(Xhat.shape) - 1)
    r = min(r, max_rank)
    global_mean = np.nanmean(Xhat)
    if np.isnan(global_mean):
        global_mean = 0.0

    for _ in range(n_iter):
        Xhat = np.where(np.isnan(Xhat), global_mean, Xhat)
        U, s, Vt = np.linalg.svd(Xhat, full_matrices=False)
        s[r:] = 0.0
        Xnew = (U * s) @ Vt
        Xnew[observed] = X_joint[observed]
        Xnew = np.where(np.isnan(Xnew), global_mean, Xnew)
        err = np.linalg.norm(Xnew - Xhat) / (np.linalg.norm(Xhat) + 1e-12)
        Xhat = Xnew
        if err < tol:
            break
    return Xhat


def soft_impute_joint(X_joint, lam=None, max_rank=None, n_iter=100, tol=1e-6):
    Xhat = safe_mean_fill_cols(X_joint)
    observed = ~np.isnan(X_joint)
    global_mean = np.nanmean(Xhat)
    if np.isnan(global_mean):
        global_mean = 0.0
    Xhat = np.where(np.isnan(Xhat), global_mean, Xhat)

    if lam is None:
        U0, s0, Vt0 = np.linalg.svd(Xhat, full_matrices=False)
        lam = 0.2 * s0[0]

    for _ in range(n_iter):
        X_prev = Xhat.copy()
        Xhat = np.where(np.isnan(Xhat), global_mean, Xhat)
        U, s, Vt = np.linalg.svd(Xhat, full_matrices=False)
        s_shrunk = np.maximum(s - lam, 0.0)
        if max_rank is not None:
            rr = min(max_rank, len(s_shrunk))
            U = U[:, :rr]
            Vt = Vt[:rr, :]
            s_shrunk = s_shrunk[:rr]
        Xnew = (U * s_shrunk) @ Vt
        Xnew[observed] = X_joint[observed]
        Xnew = np.where(np.isnan(Xnew), global_mean, Xnew)
        err = np.linalg.norm(Xnew - X_prev) / (np.linalg.norm(X_prev) + 1e-12)
        Xhat = Xnew
        if err < tol:
            break
    return Xhat

# =========================================================
# 4. Neural multivariate models
# =========================================================

def mean_fill_tensor(X):
    N, T, K = X.shape
    Xhat = X.copy().astype(float)
    for k in range(K):
        Xhat[:, :, k] = safe_mean_fill_cols(Xhat[:, :, k])
    return Xhat


class MultivariateSeqDataset(Dataset):
    def __init__(self, X_in, X_true, mask):
        self.X_in = torch.tensor(X_in, dtype=torch.float32)
        self.X_true = torch.tensor(X_true, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)

    def __len__(self):
        return self.X_true.shape[0]

    def __getitem__(self, idx):
        return self.X_in[idx], self.X_true[idx], self.mask[idx]


def masked_mse_torch(y_pred, y_true, mask):
    diff2 = (y_pred - y_true) ** 2
    diff2 = diff2[mask]
    if diff2.numel() == 0:
        return torch.tensor(0.0, device=y_pred.device)
    return diff2.mean()


class MultivariateMLPImputer(nn.Module):
    def __init__(self, d_in, d_out, hidden=(256, 128), dropout=0.1):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MultivariateBiLSTMImputer(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size // 2),
        )

    def forward(self, x):
        h, _ = self.lstm(x)
        y = self.head(h)
        return y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]


class MultivariateTransformerImputer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
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
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x):
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        h = self.encoder(h)
        y = self.head(h)
        return y


def _prepare_multivariate_inputs(X_true, X_masked, train_rows, val_rows):
    # mean fill by variable, then concatenate mask as extra channels
    X_in = mean_fill_tensor(X_masked)

    Xtr_in = X_in[train_rows]
    Xva_in = X_in[val_rows]
    Xtr_true = X_true[train_rows]
    Xva_true = X_true[val_rows]

    # standardize by variable using train inputs
    mu = np.nanmean(Xtr_in, axis=(0, 1), keepdims=True)
    sigma = np.nanstd(Xtr_in, axis=(0, 1), keepdims=True) + 1e-6

    Xtr_in_std = (Xtr_in - mu) / sigma
    Xva_in_std = (Xva_in - mu) / sigma
    Xtr_true_std = (Xtr_true - mu) / sigma
    Xva_true_std = (Xva_true - mu) / sigma

    return Xtr_in_std, Xva_in_std, Xtr_true_std, Xva_true_std, mu, sigma


def mlp_fit_predict_multivariate(
    X_true, X_masked, mask_train_loss, train_rows, val_rows,
    seed=0, epochs=300, batch_size=32, lr=1e-3, weight_decay=1e-4, patience=40,
    hidden=(256, 128), dropout=0.1, device=None,
):
    if device is None:
        device = DEFAULT_DEVICE

    N, T, K = X_true.shape
    Xtr_in_std, Xva_in_std, Xtr_true_std, _, mu, sigma = _prepare_multivariate_inputs(X_true, X_masked, train_rows, val_rows)

    Mtr = mask_train_loss[train_rows]
    Mva = mask_train_loss[val_rows]

    Xtr_mask = Mtr.astype(np.float32)
    Xva_mask = Mva.astype(np.float32)

    Xtr_flat = np.concatenate([Xtr_in_std.reshape(len(train_rows), T * K), Xtr_mask.reshape(len(train_rows), T * K)], axis=1)
    Xva_flat = np.concatenate([Xva_in_std.reshape(len(val_rows), T * K), Xva_mask.reshape(len(val_rows), T * K)], axis=1)
    Ytr_flat = Xtr_true_std.reshape(len(train_rows), T * K)

    Xtr_tensor = torch.tensor(Xtr_flat, dtype=torch.float32)
    Ytr_tensor = torch.tensor(Ytr_flat, dtype=torch.float32)
    Mtr_tensor = torch.tensor(Mtr.reshape(len(train_rows), T * K), dtype=torch.bool)
    dl_tr = DataLoader(torch.utils.data.TensorDataset(Xtr_tensor, Ytr_tensor, Mtr_tensor), batch_size=batch_size, shuffle=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = MultivariateMLPImputer(d_in=2 * T * K, d_out=T * K, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = float("inf")
    best_state = None
    bad = 0

    for _ in range(epochs):
        model.train()
        for xb, yb, mb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = masked_mse_torch(pred, yb, mb)
            if torch.isnan(loss):
                raise RuntimeError("MLP loss is NaN")
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            train_mses = []
            for xb, yb, mb in dl_tr:
                xb = xb.to(device)
                yb = yb.to(device)
                mb = mb.to(device)
                pred = model(xb)
                train_mses.append(masked_mse_torch(pred, yb, mb).item())
            rmse_tr = math.sqrt(float(np.mean(train_mses))) if train_mses else float("inf")

        if rmse_tr < best - 1e-7:
            best = rmse_tr
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_std = model(torch.tensor(Xva_flat, dtype=torch.float32).to(device)).cpu().numpy()
    pred_std = pred_std.reshape(len(val_rows), T, K)
    pred = pred_std * sigma + mu
    return pred


def lstm_fit_predict_multivariate(
    X_true, X_masked, mask_train_loss, train_rows, val_rows,
    seed=0, epochs=300, batch_size=32, lr=1e-3, weight_decay=1e-4, patience=40,
    hidden_size=64, num_layers=2, dropout=0.1, device=None,
):
    if device is None:
        device = DEFAULT_DEVICE

    _, T, K = X_true.shape
    Xtr_in_std, Xva_in_std, Xtr_true_std, _, mu, sigma = _prepare_multivariate_inputs(X_true, X_masked, train_rows, val_rows)
    Mtr = mask_train_loss[train_rows]
    Mva = mask_train_loss[val_rows]

    Xtr_seq = np.concatenate([Xtr_in_std, Mtr.astype(np.float32)], axis=2)
    Xva_seq = np.concatenate([Xva_in_std, Mva.astype(np.float32)], axis=2)

    ds_tr = MultivariateSeqDataset(Xtr_seq, Xtr_true_std, Mtr)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = MultivariateBiLSTMImputer(input_size=2 * K, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = float("inf")
    best_state = None
    bad = 0

    for _ in range(epochs):
        model.train()
        for xb, yb, mb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = masked_mse_torch(pred, yb, mb)
            if torch.isnan(loss):
                raise RuntimeError("LSTM loss is NaN")
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            train_mses = []
            for xb, yb, mb in dl_tr:
                xb = xb.to(device)
                yb = yb.to(device)
                mb = mb.to(device)
                pred = model(xb)
                train_mses.append(masked_mse_torch(pred, yb, mb).item())
            rmse_tr = math.sqrt(float(np.mean(train_mses))) if train_mses else float("inf")

        if rmse_tr < best - 1e-7:
            best = rmse_tr
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_std = model(torch.tensor(Xva_seq, dtype=torch.float32).to(device)).cpu().numpy()
    pred = pred_std * sigma + mu
    return pred


def transformer_fit_predict_multivariate(
    X_true, X_masked, mask_train_loss, train_rows, val_rows,
    seed=0, epochs=250, batch_size=32, lr=1e-3, weight_decay=1e-4, patience=30,
    d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, device=None,
):
    if device is None:
        device = DEFAULT_DEVICE

    _, T, K = X_true.shape
    Xtr_in_std, Xva_in_std, Xtr_true_std, _, mu, sigma = _prepare_multivariate_inputs(X_true, X_masked, train_rows, val_rows)
    Mtr = mask_train_loss[train_rows]
    Mva = mask_train_loss[val_rows]

    Xtr_seq = np.concatenate([Xtr_in_std, Mtr.astype(np.float32)], axis=2)
    Xva_seq = np.concatenate([Xva_in_std, Mva.astype(np.float32)], axis=2)

    ds_tr = MultivariateSeqDataset(Xtr_seq, Xtr_true_std, Mtr)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = MultivariateTransformerImputer(
        input_dim=2 * K,
        output_dim=K,
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
        for xb, yb, mb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = masked_mse_torch(pred, yb, mb)
            if torch.isnan(loss):
                raise RuntimeError("Transformer loss is NaN")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            train_mses = []
            for xb, yb, mb in dl_tr:
                xb = xb.to(device)
                yb = yb.to(device)
                mb = mb.to(device)
                pred = model(xb)
                train_mses.append(masked_mse_torch(pred, yb, mb).item())
            rmse_tr = math.sqrt(float(np.mean(train_mses))) if train_mses else float("inf")

        if rmse_tr < best - 1e-7:
            best = rmse_tr
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_std = model(torch.tensor(Xva_seq, dtype=torch.float32).to(device)).cpu().numpy()
    pred = pred_std * sigma + mu
    return pred

# =========================================================
# 5. Metrics and experiment loop
# =========================================================

def eval_metrics_tensor(X_true, X_hat, mask):
    diff = (X_hat - X_true)[mask]
    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))
    return rmse, mae


def eval_metrics_tensor_per_variable(X_true, X_hat, mask, variable_names):
    metrics = {}
    K = X_true.shape[2]
    for k in range(K):
        mk = mask[:, :, k]
        var = variable_names[k]
        if mk.sum() == 0:
            metrics[var] = {"rmse": np.nan, "mae": np.nan}
            continue
        diff = (X_hat[:, :, k] - X_true[:, :, k])[mk]
        metrics[var] = {
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "mae": float(np.mean(np.abs(diff))),
        }
    return metrics


def safe_nanmean(x):
    return float(np.nanmean(x)) if len(x) > 0 else np.nan


def safe_nanstd(x):
    return float(np.nanstd(x)) if len(x) > 0 else np.nan


def run_multivariate_experiment(
    X,
    variable_names,
    p_values=(0.1, 0.3, 0.5),
    seeds=range(5),
    rank=3,
    mechanism="mcar",
    mech_kwargs=None,
    softimpute_kwargs=None,
    use_transformer=True,
    mlp_epochs=300,
    mlp_batch_size=32,
    mlp_lr=1e-3,
    mlp_weight_decay=1e-4,
    mlp_patience=40,
    mlp_hidden=(256, 128),
    mlp_dropout=0.1,
    lstm_epochs=300,
    lstm_batch_size=32,
    lstm_lr=1e-3,
    lstm_weight_decay=1e-4,
    lstm_patience=40,
    lstm_hidden_size=64,
    lstm_num_layers=2,
    lstm_dropout=0.1,
    trf_epochs=250,
    trf_batch_size=32,
    trf_lr=1e-3,
    trf_weight_decay=1e-4,
    trf_patience=30,
    trf_d_model=64,
    trf_nhead=4,
    trf_num_layers=2,
    trf_dim_feedforward=128,
    trf_dropout=0.1,
):
    mech_kwargs = mech_kwargs or {}
    softimpute_kwargs = softimpute_kwargs or {}
    N, T, K = X.shape

    methods = ["mean", "twfe", "svd", "nuclear", "mlp", "lstm"]
    if use_transformer:
        methods.append("transformer")

    rows = []
    per_var_rows = []

    for p in p_values:
        method_metrics = {}
        per_var_metrics = {}
        for method in methods:
            method_metrics[method] = {"rmse": [], "mae": []}
            per_var_metrics[method] = {}
            for var in variable_names:
                per_var_metrics[method][var] = {"rmse": [], "mae": []}

        success_count = 0

        for seed in seeds:
            mask = make_mask_tensor(X, p=p, seed=seed, mechanism=mechanism, mech_kwargs=mech_kwargs)
            rng = np.random.default_rng(seed)
            idx = np.arange(N)
            rng.shuffle(idx)
            n_train = int(0.8 * N)
            train_rows = idx[:n_train]
            val_rows = idx[n_train:]

            X_masked = X.copy()
            X_masked[mask] = np.nan

            mask_val = mask.copy()
            mask_val[train_rows, :, :] = False
            if mask_val.sum() == 0:
                print("[SKIP SEED] empty mask_val:", mechanism, p, seed)
                continue
            if np.any(np.all(np.isnan(X_masked[val_rows]), axis=(0, 2))):
                print("[SKIP SEED] full-NaN year on val:", mechanism, p, seed)
                continue

            X_joint = flatten_tensor_to_joint_matrix(X)
            X_masked_joint = flatten_tensor_to_joint_matrix(X_masked)

            try:
                results_this_seed = {}

                X_mean_joint = mean_joint_impute(X_masked_joint)
                results_this_seed["mean"] = unflatten_joint_matrix_to_tensor(X_mean_joint, T, K)

                X_twfe_joint = twfe_joint_impute(X_masked_joint)
                results_this_seed["twfe"] = unflatten_joint_matrix_to_tensor(X_twfe_joint, T, K)

                X_svd_joint = iterative_svd_joint_impute(X_masked_joint, r=rank)
                results_this_seed["svd"] = unflatten_joint_matrix_to_tensor(X_svd_joint, T, K)

                X_nuc_joint = soft_impute_joint(X_masked_joint, max_rank=rank, **softimpute_kwargs)
                results_this_seed["nuclear"] = unflatten_joint_matrix_to_tensor(X_nuc_joint, T, K)
            except Exception as e:
                print("[SKIP SEED] classical methods failed:", mechanism, p, seed, str(e))
                continue

            try:
                Yhat_val_mlp = mlp_fit_predict_multivariate(
                    X_true=X, X_masked=X_masked, mask_train_loss=mask,
                    train_rows=train_rows, val_rows=val_rows, seed=seed,
                    epochs=mlp_epochs, batch_size=mlp_batch_size, lr=mlp_lr,
                    weight_decay=mlp_weight_decay, patience=mlp_patience,
                    hidden=mlp_hidden, dropout=mlp_dropout,
                )
                X_mlp = X.copy(); X_mlp[val_rows, :, :] = Yhat_val_mlp
                results_this_seed["mlp"] = X_mlp
            except Exception as e:
                print("[WARN] MLP failed:", mechanism, p, seed, str(e))

            try:
                Yhat_val_lstm = lstm_fit_predict_multivariate(
                    X_true=X, X_masked=X_masked, mask_train_loss=mask,
                    train_rows=train_rows, val_rows=val_rows, seed=seed,
                    epochs=lstm_epochs, batch_size=lstm_batch_size, lr=lstm_lr,
                    weight_decay=lstm_weight_decay, patience=lstm_patience,
                    hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, dropout=lstm_dropout,
                )
                X_lstm = X.copy(); X_lstm[val_rows, :, :] = Yhat_val_lstm
                results_this_seed["lstm"] = X_lstm
            except Exception as e:
                print("[WARN] LSTM failed:", mechanism, p, seed, str(e))

            if use_transformer:
                try:
                    Yhat_val_trf = transformer_fit_predict_multivariate(
                        X_true=X, X_masked=X_masked, mask_train_loss=mask,
                        train_rows=train_rows, val_rows=val_rows, seed=seed,
                        epochs=trf_epochs, batch_size=trf_batch_size, lr=trf_lr,
                        weight_decay=trf_weight_decay, patience=trf_patience,
                        d_model=trf_d_model, nhead=trf_nhead, num_layers=trf_num_layers,
                        dim_feedforward=trf_dim_feedforward, dropout=trf_dropout,
                    )
                    X_trf = X.copy(); X_trf[val_rows, :, :] = Yhat_val_trf
                    results_this_seed["transformer"] = X_trf
                except Exception as e:
                    print("[WARN] Transformer failed:", mechanism, p, seed, str(e))

            if len(results_this_seed) < 4:
                continue

            for method, X_hat in results_this_seed.items():
                rmse, mae = eval_metrics_tensor(X, X_hat, mask_val)
                method_metrics[method]["rmse"].append(rmse)
                method_metrics[method]["mae"].append(mae)

                per_var = eval_metrics_tensor_per_variable(X, X_hat, mask_val, variable_names)
                for var in variable_names:
                    per_var_metrics[method][var]["rmse"].append(per_var[var]["rmse"])
                    per_var_metrics[method][var]["mae"].append(per_var[var]["mae"])

            success_count += 1

        row = {
            "setting": "multivariate_joint",
            "variables": ",".join(variable_names),
            "mechanism": mechanism,
            "missing_fraction": p,
            "n_success": int(success_count),
        }
        for method in methods:
            row["RMSE_" + method] = safe_nanmean(method_metrics[method]["rmse"])
            row["MAE_" + method] = safe_nanmean(method_metrics[method]["mae"])
            row["RMSE_" + method + "_std"] = safe_nanstd(method_metrics[method]["rmse"])
            for var in variable_names:
                row["RMSE_" + method + "_" + var] = safe_nanmean(per_var_metrics[method][var]["rmse"])
                row["MAE_" + method + "_" + var] = safe_nanmean(per_var_metrics[method][var]["mae"])

                per_var_rows.append({
                    "setting": "multivariate_joint",
                    "variables": ",".join(variable_names),
                    "mechanism": mechanism,
                    "missing_fraction": p,
                    "n_success": int(success_count),
                    "method": method,
                    "variable": var,
                    "RMSE": safe_nanmean(per_var_metrics[method][var]["rmse"]),
                    "MAE": safe_nanmean(per_var_metrics[method][var]["mae"]),
                    "RMSE_std": safe_nanstd(per_var_metrics[method][var]["rmse"]),
                })

        rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(per_var_rows)


def run_all_mechanisms_multivariate(X, variable_names, p_values=(0.1, 0.3, 0.5), seeds=range(5), rank=3, softimpute_kwargs=None, use_transformer=True):
    dfs = []
    per_var_dfs = []
    for mechanism, mech_kwargs in [
        ("mcar", {}),
        ("mar", {"beta": 2.0}),
        ("mnar", {"beta": 2.0, "tail": "low"}),
        ("block", {"avg_block_len": 6}),
    ]:
        df_overall, df_per_var = run_multivariate_experiment(
            X,
            variable_names,
            p_values,
            seeds,
            rank,
            mechanism=mechanism,
            mech_kwargs=mech_kwargs,
            softimpute_kwargs=softimpute_kwargs,
            use_transformer=use_transformer,
        )
        dfs.append(df_overall)
        per_var_dfs.append(df_per_var)
    return pd.concat(dfs, ignore_index=True), pd.concat(per_var_dfs, ignore_index=True)


def plot_overall_results(all_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    methods = [
        ("RMSE_mean", "Mean"),
        ("RMSE_twfe", "TWFE"),
        ("RMSE_svd", "Iterative SVD"),
        ("RMSE_nuclear", "SoftImpute"),
        ("RMSE_mlp", "MLP"),
        ("RMSE_lstm", "BiLSTM"),
    ]
    if "RMSE_transformer" in all_results.columns:
        methods.append(("RMSE_transformer", "Transformer"))

    for mech in sorted(all_results["mechanism"].unique()):
        dfm = all_results[all_results["mechanism"] == mech].sort_values("missing_fraction")
        plt.figure(figsize=(7, 4.5))
        for col, label in methods:
            if col in dfm.columns:
                plt.plot(dfm["missing_fraction"], dfm[col], marker="o", label=label)
        plt.xlabel("Missing fraction")
        plt.ylabel("RMSE (overall)")
        plt.title("Multivariate imputation: overall RMSE ({0})".format(mech.upper()))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "overall_{0}.png".format(mech)), dpi=200)
        plt.close()


def plot_per_variable_results(per_var_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    label_map = {
        "mean": "Mean",
        "twfe": "TWFE",
        "svd": "Iterative SVD",
        "nuclear": "SoftImpute",
        "mlp": "MLP",
        "lstm": "BiLSTM",
        "transformer": "Transformer",
    }

    for var in sorted(per_var_results["variable"].unique()):
        for mech in sorted(per_var_results["mechanism"].unique()):
            dfm = per_var_results[(per_var_results["variable"] == var) & (per_var_results["mechanism"] == mech)].copy()
            if dfm.empty:
                continue
            plt.figure(figsize=(7, 4.5))
            for method in ["mean", "twfe", "svd", "nuclear", "mlp", "lstm", "transformer"]:
                dmm = dfm[dfm["method"] == method].sort_values("missing_fraction")
                if dmm.empty:
                    continue
                plt.plot(dmm["missing_fraction"], dmm["RMSE"], marker="o", label=label_map.get(method, method))
            plt.xlabel("Missing fraction")
            plt.ylabel("RMSE")
            plt.title("Multivariate imputation for {0} ({1})".format(var, mech.upper()))
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "{0}_{1}.png".format(var, mech)), dpi=200)
            plt.close()


# =========================================================
# 6. Main
# =========================================================

if __name__ == "__main__":
    csv_path = "2026-04-18T10-10_export.csv"
    variables = ["ccon", "cn", "emp", "hc"]
    log_vars = ["ccon", "cn", "emp"]

    long_df = load_pwt_csv(csv_path)
    print("Loaded rows:", len(long_df))
    print("Years:", long_df["year"].min(), "-", long_df["year"].max())
    print("Available variables:", sorted(long_df["var_code"].dropna().unique()))

    df_sub = filter_variables(long_df, variable_codes=variables, start_year=1999, end_year=2023)
    df_sub = apply_variablewise_log(df_sub, log_vars=log_vars, use_log1p=False)

    all_countries = sorted(df_sub["iso"].dropna().unique())
    rng = np.random.default_rng(42)
    shuffled = np.array(all_countries)
    rng.shuffle(shuffled)
    n_train = int(0.8 * len(shuffled))
    train_countries = shuffled[:n_train].tolist()

    stats_df = fit_standardization_params(df_sub, train_countries=train_countries, value_col="value_transformed")
    df_sub = apply_standardization(df_sub, stats_df, value_col="value_transformed")

    tensor_info = build_balanced_tensor(df_sub, variables=variables, value_col="value_scaled", min_nonmissing_share=0.8)
    X = tensor_info["tensor"]
    print("Tensor shape (N, T, K):", X.shape)
    print("Variables used:", tensor_info["variables"])
    print("Countries used:", len(tensor_info["countries"]))
    print("Years used:", len(tensor_info["years"]))

    all_results, per_var_results = run_all_mechanisms_multivariate(
        X=X,
        variable_names=tensor_info["variables"],
        p_values=(0.1, 0.3, 0.5),
        seeds=range(5),
        rank=3,
        softimpute_kwargs={},
        use_transformer=True,
    )

    print("\n=== MULTIVARIATE RESULTS: OVERALL ===")
    print(all_results)
    print("\n=== MULTIVARIATE RESULTS: PER VARIABLE ===")
    print(per_var_results.head(20))

    out_dir = "results_pwt_multivariate_per_variable"
    plot_dir_overall = os.path.join(out_dir, "plots_overall")
    plot_dir_per_var = os.path.join(out_dir, "plots_per_variable")
    os.makedirs(out_dir, exist_ok=True)

    all_results.to_csv(os.path.join(out_dir, "all_results_multivariate_overall.csv"), index=False)
    per_var_results.to_csv(os.path.join(out_dir, "all_results_multivariate_per_variable.csv"), index=False)

    plot_overall_results(all_results, plot_dir_overall)
    plot_per_variable_results(per_var_results, plot_dir_per_var)

    print("\nSaved overall results:", os.path.join(out_dir, "all_results_multivariate_overall.csv"))
    print("Saved per-variable results:", os.path.join(out_dir, "all_results_multivariate_per_variable.csv"))
    print("Saved overall plots to:", plot_dir_overall)
    print("Saved per-variable plots to:", plot_dir_per_var)