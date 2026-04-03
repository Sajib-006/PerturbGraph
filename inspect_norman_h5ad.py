#!/usr/bin/env python3

import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import scanpy as sc


def to_dense_preview(x, n_rows=5, n_cols=5):
    if hasattr(x, "toarray"):
        x = x[:n_rows, :n_cols].toarray()
    else:
        x = np.asarray(x[:n_rows, :n_cols])
    return x


def safe_series_summary(s: pd.Series, top_n: int = 20):
    out = {}
    out["dtype"] = str(s.dtype)
    out["n_unique"] = int(s.nunique(dropna=False))
    vc = s.value_counts(dropna=False).head(top_n)
    out["top_values"] = {str(k): int(v) for k, v in vc.items()}
    return out


def summarize_matrix(adata):
    X = adata.X
    n_obs, n_vars = adata.shape

    if hasattr(X, "nnz"):
        nnz = int(X.nnz)
    else:
        nnz = int(np.count_nonzero(X))

    density = nnz / (n_obs * n_vars)

    obs_sum = np.asarray(X.sum(axis=1)).ravel()
    var_sum = np.asarray(X.sum(axis=0)).ravel()

    summary = {
        "shape": [int(n_obs), int(n_vars)],
        "nnz": nnz,
        "density": float(density),
        "obs_sum_describe": pd.Series(obs_sum).describe().to_dict(),
        "var_sum_describe": pd.Series(var_sum).describe().to_dict(),
        "preview_block": to_dense_preview(X).tolist(),
    }
    return summary


def summarize_obs_var(df: pd.DataFrame, name: str, top_n_cols: int = 50):
    out = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns[:top_n_cols]),
        "column_summaries": {},
    }

    for col in df.columns[:top_n_cols]:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            out["column_summaries"][col] = {
                "dtype": str(s.dtype),
                "describe": s.describe().to_dict(),
                "n_missing": int(s.isna().sum()),
            }
        else:
            out["column_summaries"][col] = safe_series_summary(s.astype("object"))
    return out


def find_candidate_perturbation_columns(obs: pd.DataFrame):
    candidates = []
    keywords = [
        "pert", "gene", "target", "guide", "gRNA", "sgRNA", "condition",
        "treatment", "label", "group", "class", "assignment"
    ]
    for col in obs.columns:
        lc = col.lower()
        if any(k.lower() in lc for k in keywords):
            candidates.append(col)
    return candidates


def detect_control_like_values(obs: pd.DataFrame, columns):
    control_words = [
        "control", "ctrl", "non-target", "nontarget", "non_target",
        "nt", "unperturbed", "vehicle", "mock"
    ]
    findings = {}
    for col in columns:
        s = obs[col].astype(str)
        vals = sorted(set(s.dropna().unique()))
        matched = [v for v in vals if any(w in v.lower() for w in control_words)]
        if matched:
            findings[col] = matched[:50]
    return findings


def check_layers_raw_counts(adata):
    info = {}
    if adata.raw is not None:
        info["has_raw"] = True
        info["raw_shape"] = list(adata.raw.shape)
        try:
            Xr = adata.raw.X
            if hasattr(Xr, "nnz"):
                info["raw_nnz"] = int(Xr.nnz)
            else:
                info["raw_nnz"] = int(np.count_nonzero(Xr))
        except Exception as e:
            info["raw_error"] = repr(e)
    else:
        info["has_raw"] = False

    info["layers"] = list(adata.layers.keys())
    layer_stats = {}
    for layer in adata.layers.keys():
        try:
            X = adata.layers[layer]
            if hasattr(X, "nnz"):
                nnz = int(X.nnz)
            else:
                nnz = int(np.count_nonzero(X))
            layer_stats[layer] = {
                "shape": list(X.shape),
                "nnz": nnz,
            }
        except Exception as e:
            layer_stats[layer] = {"error": repr(e)}
    info["layer_stats"] = layer_stats
    return info


def inspect_h5ad(path, out_prefix="norman_inspect"):
    print("=" * 100)
    print(f"Reading: {path}")
    print("=" * 100)

    adata = sc.read_h5ad(path)

    print("\nBasic AnnData summary")
    print(adata)
    print("shape:", adata.shape)
    print("obs columns:", list(adata.obs.columns))
    print("var columns:", list(adata.var.columns))
    print("layers:", list(adata.layers.keys()))
    print("has raw:", adata.raw is not None)

    print("\nMatrix summary")
    matrix_summary = summarize_matrix(adata)
    print(json.dumps(matrix_summary, indent=2, default=str)[:3000])

    print("\nobs summary")
    obs_summary = summarize_obs_var(adata.obs, "obs")
    print("n_obs columns:", obs_summary["n_cols"])
    print("first obs columns:", obs_summary["columns"])

    print("\nvar summary")
    var_summary = summarize_obs_var(adata.var, "var")
    print("n_var columns:", var_summary["n_cols"])
    print("first var columns:", var_summary["columns"])

    print("\nCandidate perturbation-related obs columns")
    pert_cols = find_candidate_perturbation_columns(adata.obs)
    print(pert_cols)

    print("\nControl-like values in candidate columns")
    control_hits = detect_control_like_values(adata.obs, pert_cols)
    print(json.dumps(control_hits, indent=2))

    print("\nRaw/layer info")
    raw_layer_info = check_layers_raw_counts(adata)
    print(json.dumps(raw_layer_info, indent=2, default=str))

    # gene naming check
    print("\nGene naming preview")
    print("var_names[:20]:", adata.var_names[:20].tolist())
    if "gene_name" in adata.var.columns:
        print("gene_name[:20]:", adata.var["gene_name"].astype(str).head(20).tolist())
    if "feature_types" in adata.var.columns:
        print("feature_types counts:")
        print(adata.var["feature_types"].value_counts(dropna=False).head(20))

    # save detailed reports
    obs_path = f"{out_prefix}_obs_summary.json"
    var_path = f"{out_prefix}_var_summary.json"
    meta_path = f"{out_prefix}_meta_summary.json"

    with open(obs_path, "w") as f:
        json.dump(obs_summary, f, indent=2, default=str)
    with open(var_path, "w") as f:
        json.dump(var_summary, f, indent=2, default=str)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "matrix_summary": matrix_summary,
                "candidate_perturbation_columns": pert_cols,
                "control_hits": control_hits,
                "raw_layer_info": raw_layer_info,
            },
            f,
            indent=2,
            default=str,
        )

    # save quick previews
    adata.obs.head(50).to_csv(f"{out_prefix}_obs_head.csv")
    adata.var.head(200).to_csv(f"{out_prefix}_var_head.csv")

    # if candidate perturb columns exist, save value counts
    for col in pert_cols:
        vc = adata.obs[col].astype(str).value_counts(dropna=False).head(500)
        vc.to_csv(f"{out_prefix}_value_counts_{col}.csv", header=["count"])

    print("\nSaved:")
    print(f"  {obs_path}")
    print(f"  {var_path}")
    print(f"  {meta_path}")
    print(f"  {out_prefix}_obs_head.csv")
    print(f"  {out_prefix}_var_head.csv")
    for col in pert_cols:
        print(f"  {out_prefix}_value_counts_{col}.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5ad",
        type=str,
        required=True,
        help="Path to Norman h5ad file",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="norman_inspect",
        help="Prefix for saved inspection outputs",
    )
    args = parser.parse_args()

    inspect_h5ad(args.h5ad, args.out_prefix)


if __name__ == "__main__":
    main()