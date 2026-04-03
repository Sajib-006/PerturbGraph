#!/usr/bin/env python3

import os
import json
import argparse
import random

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from node2vec import Node2Vec
from torch_geometric.nn import GCNConv

from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


# ============================================================
# utils
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

def build_go_embedding(
    go_file,
    node_list,
    gene_col="gene",
    go_col="go_id",
    sep="\t",
    min_genes_per_go=5,
    max_genes_per_go=500,
    go_svd_dim=64,
    random_state=42,
):
    """
    Build GO embedding for genes in node_list.

    Expected file format:
      gene    go_id

    Returns:
      go_feat_df: DataFrame indexed by gene, shape [len(node_list), go_svd_dim]
    """
    print(f"\nLoading GO annotations from: {go_file}")
    df = pd.read_csv(go_file, sep=sep)

    if gene_col not in df.columns or go_col not in df.columns:
        raise ValueError(
            f"GO file must contain columns '{gene_col}' and '{go_col}'. "
            f"Found: {list(df.columns)}"
        )

    df[gene_col] = df[gene_col].astype(str)
    df[go_col] = df[go_col].astype(str)

    print("Raw GO annotation rows:", df.shape)

    node_set = set(node_list)
    df = df[df[gene_col].isin(node_set)].copy()

    print("Rows after filtering to node_list:", df.shape)
    print("Unique genes with GO:", df[gene_col].nunique())
    print("Unique GO terms before filtering:", df[go_col].nunique())

    # filter GO terms by size
    go_sizes = df.groupby(go_col)[gene_col].nunique()
    keep_go = go_sizes[(go_sizes >= min_genes_per_go) & (go_sizes <= max_genes_per_go)].index
    df = df[df[go_col].isin(keep_go)].copy()

    print("Unique GO terms after filtering:", df[go_col].nunique())
    print("Rows after GO-size filtering:", df.shape)

    if df.empty:
        print("Warning: no GO annotations left after filtering. Returning zeros.")
        cols = [f"go_emb_{i}" for i in range(go_svd_dim)]
        return pd.DataFrame(0.0, index=node_list, columns=cols)

    # categorical encoding
    gene_cat = pd.Categorical(df[gene_col], categories=node_list)
    go_terms = sorted(df[go_col].unique())
    go_cat = pd.Categorical(df[go_col], categories=go_terms)

    gene_codes = gene_cat.codes
    go_codes = go_cat.codes

    valid = (gene_codes >= 0) & (go_codes >= 0)
    gene_codes = gene_codes[valid]
    go_codes = go_codes[valid]

    n_genes = len(node_list)
    n_go = len(go_terms)

    # build binary matrix
    M = np.zeros((n_genes, n_go), dtype=np.float32)
    M[gene_codes, go_codes] = 1.0

    print("GO matrix shape:", M.shape)

    # choose effective dim
    eff_dim = min(go_svd_dim, max(1, min(M.shape[0] - 1, M.shape[1] - 1)))
    if eff_dim < 1:
        print("Warning: GO matrix too small. Returning zeros.")
        cols = [f"go_emb_{i}" for i in range(go_svd_dim)]
        return pd.DataFrame(0.0, index=node_list, columns=cols)

    svd = TruncatedSVD(n_components=eff_dim, random_state=random_state)
    G = svd.fit_transform(M).astype(np.float32)

    print("GO embedding shape before padding:", G.shape)
    print("GO explained variance sum:", float(svd.explained_variance_ratio_.sum()))

    # pad if eff_dim < requested dim
    if eff_dim < go_svd_dim:
        G_pad = np.zeros((n_genes, go_svd_dim), dtype=np.float32)
        G_pad[:, :eff_dim] = G
        G = G_pad

    # standardize
    scaler = StandardScaler()
    G = scaler.fit_transform(G)

    cols = [f"go_emb_{i}" for i in range(go_svd_dim)]
    go_feat_df = pd.DataFrame(G, index=node_list, columns=cols)

    return go_feat_df

def safe_spearman(a, b, eps=1e-12):
    if np.std(a) < eps or np.std(b) < eps:
        return 0.0
    rho = spearmanr(a, b).correlation
    return 0.0 if np.isnan(rho) else float(rho)


def precision_at_k(true_vec, pred_vec, k):
    true_up = set(np.argsort(true_vec)[-k:])
    pred_up = set(np.argsort(pred_vec)[-k:])
    true_dn = set(np.argsort(true_vec)[:k])
    pred_dn = set(np.argsort(pred_vec)[:k])
    return len(true_up & pred_up) / k, len(true_dn & pred_dn) / k


def eval_program_space(H_true, H_pred):
    r2 = r2_score(H_true, H_pred, multioutput="uniform_average")
    mse = float(np.mean((H_true - H_pred) ** 2))
    cos = float(np.mean(np.diag(cosine_similarity(H_true, H_pred))))
    return {
        "R2_H": float(r2),
        "Cos_H": float(cos),
        "MSE_H": float(mse),
    }


def eval_delta_space(X_true, X_pred, k_list=(25, 50, 100), sample_genes=2000, seed=0):
    n_test, n_genes = X_true.shape

    if sample_genes is not None and sample_genes < n_genes:
        rng = np.random.default_rng(seed)
        cols = rng.choice(n_genes, size=sample_genes, replace=False)
        Xt = X_true[:, cols]
        Xp = X_pred[:, cols]
    else:
        Xt, Xp = X_true, X_pred

    spears, coses = [], []
    for i in range(n_test):
        spears.append(safe_spearman(Xt[i], Xp[i]))
        a = X_true[i]
        b = X_pred[i]
        coses.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)))

    out = {
        "Spearman_Delta_mean": float(np.mean(spears)),
        "Spearman_Delta_median": float(np.median(spears)),
        "Cos_Delta_mean": float(np.mean(coses)),
        "Cos_Delta_median": float(np.median(coses)),
        "MSE_Delta": float(np.mean((X_true - X_pred) ** 2)),
    }

    for k in k_list:
        pus, pds = [], []
        for i in range(n_test):
            pu, pd = precision_at_k(X_true[i], X_pred[i], k=k)
            pus.append(pu)
            pds.append(pd)
        out[f"Prec@{k}_UP"] = float(np.mean(pus))
        out[f"Prec@{k}_DN"] = float(np.mean(pds))

    return out


def evaluate_predictions(H_true, H_pred, X_true, Vt, seed=42):
    prog_metrics = eval_program_space(H_true, H_pred)
    X_pred = H_pred @ Vt
    delta_metrics = eval_delta_space(X_true, X_pred, seed=seed)
    out = {}
    out.update(prog_metrics)
    out.update(delta_metrics)
    return out


# ============================================================
# data loading
# ============================================================
def load_string_edges_from_raw(string_links_path, string_info_path):
    print(f"Reading STRING links: {string_links_path}")
    print(f"Reading STRING info : {string_info_path}")

    links = pd.read_csv(string_links_path, sep=r"\s+")
    info = pd.read_csv(string_info_path, sep="\t")

    print("Raw STRING edges:", links.shape)
    print("STRING info rows:", info.shape)

    protein_to_gene = dict(zip(info["#string_protein_id"], info["preferred_name"]))

    g1 = links["protein1"].map(protein_to_gene)
    g2 = links["protein2"].map(protein_to_gene)
    score = links["combined_score"].astype(int)

    edges_gene = pd.DataFrame({
        "gene1": g1,
        "gene2": g2,
        "score": score,
    }).dropna()

    edges_gene = edges_gene[edges_gene["gene1"] != edges_gene["gene2"]].copy()

    print("Edges after mapping:", edges_gene.shape)
    print("STRING gene nodes:", len(set(edges_gene["gene1"]) | set(edges_gene["gene2"])))
    return edges_gene


def build_pseudobulk_by_perturbation(
    adata,
    obs_gene_col="gene",
    ctrl_label="non-targeting",
    min_cells_per_pert=50,
):
    print("\nBuilding perturbation-level pseudo-bulk delta matrix...")

    adata.obs[obs_gene_col] = adata.obs[obs_gene_col].astype(str)
    labels_all = adata.obs[obs_gene_col].values

    ctrl_mask = labels_all == ctrl_label
    if ctrl_mask.sum() == 0:
        raise ValueError(f"No control cells found with label '{ctrl_label}'")

    control_mean = np.asarray(adata[ctrl_mask].X.mean(axis=0)).ravel()

    pert_vectors = []
    pert_labels = []

    for g in np.unique(labels_all):
        mask = labels_all == g
        if g == ctrl_label:
            continue
        if mask.sum() < min_cells_per_pert:
            continue

        expr = np.asarray(adata[mask].X.mean(axis=0)).ravel()
        delta = expr - control_mean
        pert_vectors.append(delta.astype(np.float32))
        pert_labels.append(str(g))

    X_pert = np.vstack(pert_vectors).astype(np.float32)
    pert_labels = np.asarray(pert_labels)

    print("Perturbation matrix:", X_pert.shape)
    print("Control cells:", int(ctrl_mask.sum()))
    return X_pert, pert_labels, control_mean


def make_node_splits(pert_labels_arr, test_frac=0.2, val_frac=0.15, random_state=42):
    unique_genes = np.unique(pert_labels_arr)

    train_genes_full, test_genes = train_test_split(
        unique_genes, test_size=test_frac, random_state=random_state
    )
    train_genes, val_genes = train_test_split(
        train_genes_full, test_size=val_frac, random_state=random_state
    )

    return set(train_genes), set(val_genes), set(test_genes)


def build_svd_targets_train_only(X_pert, train_row_mask, n_components=128, seed=42):
    print(f"\nFitting TruncatedSVD on TRAIN ONLY with n_components={n_components}")
    svd = TruncatedSVD(n_components=n_components, random_state=seed)

    H_train = svd.fit_transform(X_pert[train_row_mask]).astype(np.float32)
    H_all = svd.transform(X_pert).astype(np.float32)

    print("H_train shape:", H_train.shape)
    print("ExplainedVarSum:", float(svd.explained_variance_ratio_.sum()))
    return svd, H_all


# ============================================================
# graph builders
# ============================================================
def build_string_edges(edges_gene, pert_set, min_score=700):
    e = edges_gene[
        (edges_gene["score"] >= min_score) &
        (edges_gene["gene1"].isin(pert_set)) &
        (edges_gene["gene2"].isin(pert_set))
    ][["gene1", "gene2", "score"]].copy()

    e["gmin"] = e[["gene1", "gene2"]].min(axis=1)
    e["gmax"] = e[["gene1", "gene2"]].max(axis=1)
    e = (
        e.groupby(["gmin", "gmax"], as_index=False)["score"]
        .max()
        .rename(columns={"gmin": "gene1", "gmax": "gene2"})
    )

    print("STRING edges:", e.shape)
    return e


def build_coexpression_edges(
    adata,
    pert_genes,
    ctrl_label="non-targeting",
    obs_gene_col="gene",
    var_gene_col="gene_name",
    top_k=20,
    min_abs_corr=0.08,
    max_ctrl_cells=8000,
    edge_score_scale=(700, 950),
    positive_only=False,
):
    print("\nBuilding coexpression edges from control cells...")

    ctrl_mask = adata.obs[obs_gene_col].astype(str).values == ctrl_label
    ctrl_idx = np.where(ctrl_mask)[0]
    print("Control cells available:", len(ctrl_idx))

    if max_ctrl_cells is not None and len(ctrl_idx) > max_ctrl_cells:
        rng = np.random.default_rng(0)
        ctrl_idx = rng.choice(ctrl_idx, size=max_ctrl_cells, replace=False)
        print("Subsampled control cells:", len(ctrl_idx))

    gene_names = (
        adata.var[var_gene_col].astype(str).values
        if var_gene_col in adata.var.columns
        else adata.var_names.astype(str).values
    )

    sym_to_pos = {}
    for j, sym in enumerate(gene_names):
        if sym not in sym_to_pos:
            sym_to_pos[sym] = j

    kept_genes = [g for g in sorted(set(pert_genes)) if g in sym_to_pos]
    cols = [sym_to_pos[g] for g in kept_genes]

    print("Perturbation genes found in adata.var:", len(kept_genes), "/", len(set(pert_genes)))

    Xc = to_dense(adata[ctrl_idx, cols].X).astype(np.float32)

    mu = Xc.mean(axis=0, keepdims=True)
    sd = Xc.std(axis=0, keepdims=True) + 1e-6
    Xz = (Xc - mu) / sd

    corr = (Xz.T @ Xz) / max(1, (Xz.shape[0] - 1))
    corr = np.clip(corr, -1.0, 1.0)

    rows = []
    lo, hi = edge_score_scale

    for i, g in enumerate(kept_genes):
        vals = corr[i].copy()
        vals[i] = 0.0

        if positive_only:
            cand = np.where(vals >= min_abs_corr)[0]
            order = cand[np.argsort(vals[cand])[::-1]]
        else:
            cand = np.where(np.abs(vals) >= min_abs_corr)[0]
            order = cand[np.argsort(np.abs(vals[cand]))[::-1]]

        order = order[:top_k]

        for j in order:
            g2 = kept_genes[j]
            c = float(vals[j])
            score = lo + (hi - lo) * min(1.0, abs(c))
            rows.append((g, g2, score))

    e = pd.DataFrame(rows, columns=["gene1", "gene2", "score"])
    e["gmin"] = e[["gene1", "gene2"]].min(axis=1)
    e["gmax"] = e[["gene1", "gene2"]].max(axis=1)
    e = (
        e.groupby(["gmin", "gmax"], as_index=False)["score"]
        .max()
        .rename(columns={"gmin": "gene1", "gmax": "gene2"})
    )

    print("Coexpression edges:", e.shape)
    return e


def build_dorothea_tf_edges(
    dorothea_file,
    pert_set,
    mode="curated_clean",
    min_support=1,
    remove_self_loops=True,
):
    """
    mode:
      - curated_clean
      - curated_plus
      - all_noisy
      - signed_only
    """
    print(f"\nLoading DoRothEA TF edges: {dorothea_file}")
    df = pd.read_csv(dorothea_file, sep="\t")

    df["TF"] = df["TF"].astype(str)
    df["target"] = df["target"].astype(str)
    df["database"] = df["database"].astype(str)
    df["evidence"] = df["evidence"].astype(str)

    if remove_self_loops:
        df = df[df["TF"] != df["target"]].copy()

    curated_clean = {
        "trrust_signed",
        "tfact_signed",
        "oreganno_signed",
        "TFe_signed",
        "HTRIdb",
        "kegg",
        "reviews",
        "fantom4",
        "tred_via_RegNetwork",
        "IntAct",
    }

    curated_plus = curated_clean | {"PAZAR", "ReMap"}

    if mode == "curated_clean":
        df = df[df["database"].isin(curated_clean)].copy()
    elif mode == "curated_plus":
        df = df[df["database"].isin(curated_plus)].copy()
    elif mode == "all_noisy":
        drop_dbs = {"GTEx", "jaspar_2018", "hocomoco_v11"}
        df = df[~df["database"].isin(drop_dbs)].copy()
    elif mode == "signed_only":
        df = df[df["database"].isin(curated_clean)].copy()
        df = df[df["effect"] != 0].copy()
    else:
        raise ValueError(f"Unknown dorothea mode: {mode}")

    df = df[
        df["TF"].isin(pert_set) &
        df["target"].isin(pert_set)
    ].copy()

    print(f"Rows after DoRothEA filtering ({mode}):", df.shape)

    if len(df) == 0:
        raise ValueError("No TF edges left after filtering.")

    def collapse_group(g):
        n_support = len(g)
        dbs = sorted(set(g["database"]))
        n_db = len(dbs)

        pos = int((g["effect"] == 1).sum())
        neg = int((g["effect"] == -1).sum())
        zero = int((g["effect"] == 0).sum())

        if pos + neg > 0:
            effect_final = 1 if pos >= neg else -1
        else:
            effect_final = 0

        score = 700 + 30 * min(n_support, 5) + 20 * min(n_db, 4)
        if pos + neg > 0:
            score += 30
        score = min(score, 950)

        return pd.Series({
            "gene1": g["TF"].iloc[0],
            "gene2": g["target"].iloc[0],
            "score": float(score),
            "effect_final": int(effect_final),
            "n_support": int(n_support),
            "n_databases": int(n_db),
            "n_pos": int(pos),
            "n_neg": int(neg),
            "n_zero": int(zero),
        })

    edges = (
        df.groupby(["TF", "target"], as_index=False)
          .apply(collapse_group)
          .reset_index(drop=True)
    )

    edges = edges[edges["n_support"] >= min_support].copy()

    print("Collapsed TF edges:", edges.shape)
    print("Collapsed effect counts:")
    print(edges["effect_final"].value_counts(dropna=False).sort_index())

    return edges[["gene1", "gene2", "score"]].copy()


def merge_edge_tables(edge_tables):
    e = pd.concat(edge_tables, ignore_index=True).dropna()
    e["gmin"] = e[["gene1", "gene2"]].min(axis=1)
    e["gmax"] = e[["gene1", "gene2"]].max(axis=1)
    e = (
        e.groupby(["gmin", "gmax"], as_index=False)["score"]
        .max()
        .rename(columns={"gmin": "gene1", "gmax": "gene2"})
    )
    return e


# ============================================================
# biological features
# ============================================================
def build_biological_features(
    adata,
    node_list,
    string_edges=None,
    coexp_edges=None,
    tf_edges=None,
    ctrl_label="non-targeting",
    obs_gene_col="gene",
    var_gene_col="gene_name",
    max_ctrl_cells=8000,
):
    print("\nBuilding biological node features...")

    ctrl_mask = adata.obs[obs_gene_col].astype(str).values == ctrl_label
    ctrl_idx = np.where(ctrl_mask)[0]
    print("Control cells available for node features:", len(ctrl_idx))

    if max_ctrl_cells is not None and len(ctrl_idx) > max_ctrl_cells:
        rng = np.random.default_rng(0)
        ctrl_idx = rng.choice(ctrl_idx, size=max_ctrl_cells, replace=False)
        print("Subsampled control cells for node features:", len(ctrl_idx))

    gene_names = (
        adata.var[var_gene_col].astype(str).values
        if var_gene_col in adata.var.columns
        else adata.var_names.astype(str).values
    )

    sym_to_pos = {}
    for j, sym in enumerate(gene_names):
        if sym not in sym_to_pos:
            sym_to_pos[sym] = j

    # ------------------------------------------------
    # degree helpers
    # ------------------------------------------------
    def undirected_degree_map(edge_df):
        deg = {g: 0 for g in node_list}
        if edge_df is None or len(edge_df) == 0:
            return deg
        node_set = set(node_list)
        for g1, g2 in edge_df[["gene1", "gene2"]].itertuples(index=False, name=None):
            if g1 in node_set:
                deg[g1] += 1
            if g2 in node_set:
                deg[g2] += 1
        return deg

    def directed_out_map(edge_df):
        deg = {g: 0 for g in node_list}
        if edge_df is None or len(edge_df) == 0:
            return deg
        node_set = set(node_list)
        for g1, _ in edge_df[["gene1", "gene2"]].itertuples(index=False, name=None):
            if g1 in node_set:
                deg[g1] += 1
        return deg

    def directed_in_map(edge_df):
        deg = {g: 0 for g in node_list}
        if edge_df is None or len(edge_df) == 0:
            return deg
        node_set = set(node_list)
        for _, g2 in edge_df[["gene1", "gene2"]].itertuples(index=False, name=None):
            if g2 in node_set:
                deg[g2] += 1
        return deg

    string_deg = undirected_degree_map(string_edges)
    coexp_deg = undirected_degree_map(coexp_edges)
    tf_out_deg = directed_out_map(tf_edges)
    tf_in_deg = directed_in_map(tf_edges)

    # ------------------------------------------------
    # TF role feature
    # ------------------------------------------------
    tf_gene_set = set()
    if tf_edges is not None and len(tf_edges) > 0:
        tf_gene_set = set(tf_edges["gene1"].astype(str).unique())

    # ------------------------------------------------
    # control expression stats
    # ------------------------------------------------
    present_genes = [g for g in node_list if g in sym_to_pos]
    print(f"Genes found in adata.var: {len(present_genes)} / {len(node_list)}")

    ctrl_mean_map = {g: 0.0 for g in node_list}
    ctrl_var_map = {g: 0.0 for g in node_list}
    ctrl_detect_map = {g: 0.0 for g in node_list}

    if len(present_genes) > 0:
        cols = [sym_to_pos[g] for g in present_genes]
        Xc = to_dense(adata[ctrl_idx, cols].X).astype(np.float32)

        mean_expr = Xc.mean(axis=0)
        var_expr = Xc.var(axis=0)
        det_frac = (Xc > 0).mean(axis=0)

        for j, g in enumerate(present_genes):
            ctrl_mean_map[g] = float(mean_expr[j])
            ctrl_var_map[g] = float(var_expr[j])
            ctrl_detect_map[g] = float(det_frac[j])

    # ------------------------------------------------
    # STRING graph centrality + neighborhood summaries
    # ------------------------------------------------
    string_weighted_degree = {g: 0.0 for g in node_list}
    string_pagerank = {g: 0.0 for g in node_list}
    string_clustering = {g: 0.0 for g in node_list}

    neighbor_mean_ctrl_mean = {g: 0.0 for g in node_list}
    neighbor_mean_ctrl_var = {g: 0.0 for g in node_list}
    neighbor_mean_ctrl_detect = {g: 0.0 for g in node_list}

    if string_edges is not None and len(string_edges) > 0:
        Gs = nx.Graph()
        Gs.add_nodes_from(node_list)

        # add weighted edges
        for g1, g2, w in string_edges[["gene1", "gene2", "score"]].itertuples(index=False, name=None):
            Gs.add_edge(str(g1), str(g2), weight=float(w))

        # weighted degree
        for g in node_list:
            if g in Gs:
                string_weighted_degree[g] = float(Gs.degree(g, weight="weight"))

        # pagerank
        try:
            pr = nx.pagerank(Gs, weight="weight")
            for g in node_list:
                string_pagerank[g] = float(pr.get(g, 0.0))
        except Exception as e:
            print("Warning: PageRank failed, using zeros.", e)

        # clustering coefficient
        try:
            clust = nx.clustering(Gs, weight="weight")
            for g in node_list:
                string_clustering[g] = float(clust.get(g, 0.0))
        except Exception as e:
            print("Warning: clustering failed, using zeros.", e)

        # neighborhood summaries
        for g in node_list:
            nbrs = list(Gs.neighbors(g)) if g in Gs else []
            if len(nbrs) == 0:
                continue

            nbr_mean = [ctrl_mean_map[n] for n in nbrs]
            nbr_var = [ctrl_var_map[n] for n in nbrs]
            nbr_det = [ctrl_detect_map[n] for n in nbrs]

            neighbor_mean_ctrl_mean[g] = float(np.mean(nbr_mean))
            neighbor_mean_ctrl_var[g] = float(np.mean(nbr_var))
            neighbor_mean_ctrl_detect[g] = float(np.mean(nbr_det))

    # ------------------------------------------------
    # assemble rows
    # ------------------------------------------------
    rows = []
    for g in node_list:
        rows.append({
            "gene": g,
            "ctrl_mean": ctrl_mean_map[g],
            "ctrl_var": ctrl_var_map[g],
            "ctrl_detect_frac": ctrl_detect_map[g],
            "string_degree": float(string_deg[g]),
            "coexp_degree": float(coexp_deg[g]),
            "tf_out_degree": float(tf_out_deg[g]),
            "tf_in_degree": float(tf_in_deg[g]),
            "is_TF": float(g in tf_gene_set),

            # new structural features
            "string_weighted_degree": float(string_weighted_degree[g]),
            "string_pagerank": float(string_pagerank[g]),
            "string_clustering": float(string_clustering[g]),

            # new neighborhood summaries
            "neighbor_mean_ctrl_mean": float(neighbor_mean_ctrl_mean[g]),
            "neighbor_mean_ctrl_var": float(neighbor_mean_ctrl_var[g]),
            "neighbor_mean_ctrl_detect_frac": float(neighbor_mean_ctrl_detect[g]),
        })

    bio_feat_df = pd.DataFrame(rows).set_index("gene").loc[node_list]

    # standardize all continuous/binary features together
    scaler = StandardScaler()
    bio_feat_df.loc[:, :] = scaler.fit_transform(bio_feat_df.values)

    print("Biological feature matrix shape:", bio_feat_df.shape)
    print("Biological feature columns:")
    print(list(bio_feat_df.columns))

    return bio_feat_df


# ============================================================
# graph feature cache
# ============================================================
def make_edge_index_and_weight(edge_df, node_list, node2i):
    src = edge_df["gene1"].map(node2i).to_numpy()
    dst = edge_df["gene2"].map(node2i).to_numpy()
    w = edge_df["score"].to_numpy().astype(np.float32)

    if len(w) == 0:
        raise ValueError("edge_df is empty.")

    w = (w - w.min()) / (w.max() - w.min() + 1e-8)

    edge_index = np.vstack([
        np.concatenate([src, dst]),
        np.concatenate([dst, src]),
    ])
    edge_weight = np.concatenate([w, w]).astype(np.float32)

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
    return edge_index, edge_weight


def build_node2vec_features(
    edge_df,
    node_list,
    node2i,
    dim=128,
    walk_length=30,
    num_walks=100,
    window=10,
    p=1.0,
    q=1.0,
    workers=4,
    seed=42,
):
    G = nx.Graph()
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_df[["gene1", "gene2"]].itertuples(index=False, name=None))

    print("Graph nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())

    node2vec = Node2Vec(
        G,
        dimensions=dim,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        p=p,
        q=q,
        quiet=False,
        seed=seed,
    )
    w2v = node2vec.fit(window=window, min_count=1, batch_words=4096)
    emb_dict = {node: w2v.wv[node] for node in G.nodes if node in w2v.wv}

    X_graph = np.zeros((len(node_list), dim), dtype=np.float32)
    miss = 0
    for g, i in node2i.items():
        if g in emb_dict:
            X_graph[i] = emb_dict[g].astype(np.float32)
        else:
            X_graph[i] = 0.0
            miss += 1

    print("Missing node2vec embeddings:", miss)
    return X_graph


def get_or_build_node2vec_cache(
    cache_dir,
    graph_name,
    edge_df,
    node_list,
    node2i,
    dim=128,
    walk_length=30,
    num_walks=100,
    window=10,
    p=1.0,
    q=1.0,
    workers=4,
    seed=42,
):
    ensure_dir(cache_dir)

    cache_name = (
        f"node2vec_{graph_name}"
        f"_dim{dim}"
        f"_wl{walk_length}"
        f"_nw{num_walks}"
        f"_win{window}"
        f"_p{p}"
        f"_q{q}"
        f"_seed{seed}.npy"
    )
    cache_path = os.path.join(cache_dir, cache_name)

    if os.path.exists(cache_path):
        print(f"Loading cached Node2Vec features: {cache_path}")
        return np.load(cache_path)

    X_graph = build_node2vec_features(
        edge_df=edge_df,
        node_list=node_list,
        node2i=node2i,
        dim=dim,
        walk_length=walk_length,
        num_walks=num_walks,
        window=window,
        p=p,
        q=q,
        workers=workers,
        seed=seed,
    )
    np.save(cache_path, X_graph)
    print(f"Saved Node2Vec cache: {cache_path}")
    return X_graph

def build_pathway_embedding_from_gmt(
    pathway_file,
    node_list,
    min_genes_per_pathway=5,
    max_genes_per_pathway=300,
    pathway_svd_dim=32,
    random_state=42,
):
    """
    Build pathway embedding from GMT file.

    GMT format:
      pathway_name <tab> description <tab> gene1 <tab> gene2 ...

    Returns:
      pathway_feat_df: DataFrame indexed by gene, shape [len(node_list), pathway_svd_dim]
    """
    print(f"\nLoading pathway GMT from: {pathway_file}")

    node_set = set(node_list)
    pathway_to_genes = {}

    with open(pathway_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            pathway_name = parts[0]
            genes = [g.strip() for g in parts[2:] if g.strip()]

            # keep only genes in our node_list
            genes = sorted(set(g for g in genes if g in node_set))
            if len(genes) == 0:
                continue

            pathway_to_genes[pathway_name] = genes

    print("Raw pathways with overlap to node_list:", len(pathway_to_genes))

    # filter pathways by size
    pathway_to_genes = {
        p: genes
        for p, genes in pathway_to_genes.items()
        if min_genes_per_pathway <= len(genes) <= max_genes_per_pathway
    }

    print("Pathways after size filtering:", len(pathway_to_genes))

    if len(pathway_to_genes) == 0:
        print("Warning: no pathways left after filtering. Returning zeros.")
        cols = [f"path_emb_{i}" for i in range(pathway_svd_dim)]
        return pd.DataFrame(0.0, index=node_list, columns=cols)

    pathway_names = sorted(pathway_to_genes.keys())
    pathway2j = {p: j for j, p in enumerate(pathway_names)}
    gene2i = {g: i for i, g in enumerate(node_list)}

    M = np.zeros((len(node_list), len(pathway_names)), dtype=np.float32)

    for p, genes in pathway_to_genes.items():
        j = pathway2j[p]
        for g in genes:
            i = gene2i[g]
            M[i, j] = 1.0

    print("Pathway matrix shape:", M.shape)

    eff_dim = min(pathway_svd_dim, max(1, min(M.shape[0] - 1, M.shape[1] - 1)))
    if eff_dim < 1:
        print("Warning: pathway matrix too small. Returning zeros.")
        cols = [f"path_emb_{i}" for i in range(pathway_svd_dim)]
        return pd.DataFrame(0.0, index=node_list, columns=cols)

    svd = TruncatedSVD(n_components=eff_dim, random_state=random_state)
    P = svd.fit_transform(M).astype(np.float32)

    print("Pathway embedding shape before padding:", P.shape)
    print("Pathway explained variance sum:", float(svd.explained_variance_ratio_.sum()))

    if eff_dim < pathway_svd_dim:
        P_pad = np.zeros((len(node_list), pathway_svd_dim), dtype=np.float32)
        P_pad[:, :eff_dim] = P
        P = P_pad

    scaler = StandardScaler()
    P = scaler.fit_transform(P)

    cols = [f"path_emb_{i}" for i in range(pathway_svd_dim)]
    pathway_feat_df = pd.DataFrame(P, index=node_list, columns=cols)

    return pathway_feat_df

def assemble_features(
    X_graph=None,
    bio_feat_df=None,
    go_feat_df=None,
    pathway_feat_df=None,
    node_list=None,
    use_bio_features=False,
    use_go_features=False,
    use_pathway_features=False,
):
    parts = []

    if X_graph is not None:
        parts.append(X_graph.astype(np.float32))

    if use_bio_features:
        if bio_feat_df is None:
            raise ValueError("use_bio_features=True but bio_feat_df is None")
        parts.append(bio_feat_df.loc[node_list].values.astype(np.float32))

    if use_go_features:
        if go_feat_df is None:
            raise ValueError("use_go_features=True but go_feat_df is None")
        parts.append(go_feat_df.loc[node_list].values.astype(np.float32))

    if use_pathway_features:
        if pathway_feat_df is None:
            raise ValueError("use_pathway_features=True but pathway_feat_df is None")
        parts.append(pathway_feat_df.loc[node_list].values.astype(np.float32))

    if len(parts) == 0:
        raise ValueError("No features available to assemble.")

    X = np.concatenate(parts, axis=1)
    print("Final feature matrix shape:", X.shape)
    return X

# ============================================================
# models
# ============================================================
class MLPRegressor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class WeightedGCN_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.15):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

    def forward(self, x, edge_index, edge_weight):
        h = self.conv1(x, edge_index, edge_weight=edge_weight)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        return self.head(h)

def cosine_loss_rows(pred, target, eps=1e-8):
    pred_n = pred / (pred.norm(dim=1, keepdim=True) + eps)
    targ_n = target / (target.norm(dim=1, keepdim=True) + eps)
    cos = (pred_n * targ_n).sum(dim=1)
    return 1.0 - cos.mean()

# ============================================================
# trainers
# ============================================================
def run_mean_baseline(H_all, train_nodes, val_nodes, test_nodes, X_true_val, X_true_test, Vt, seed=42):
    mean_vec = H_all[train_nodes].mean(axis=0, keepdims=True)
    H_pred_val = np.repeat(mean_vec, val_nodes.sum(), axis=0)
    H_pred_test = np.repeat(mean_vec, test_nodes.sum(), axis=0)

    val_metrics = evaluate_predictions(H_all[val_nodes], H_pred_val, X_true_val, Vt, seed=seed)
    test_metrics = evaluate_predictions(H_all[test_nodes], H_pred_test, X_true_test, Vt, seed=seed)
    return val_metrics, test_metrics


def run_sklearn_baseline(model, X, H_all, train_nodes, val_nodes, test_nodes, X_true_val, X_true_test, Vt, seed=42):
    X_train, Y_train = X[train_nodes], H_all[train_nodes]
    X_val, Y_val = X[val_nodes], H_all[val_nodes]
    X_test, Y_test = X[test_nodes], H_all[test_nodes]

    model.fit(X_train, Y_train)

    H_pred_val = model.predict(X_val)
    H_pred_test = model.predict(X_test)

    val_metrics = evaluate_predictions(Y_val, H_pred_val, X_true_val, Vt, seed=seed)
    test_metrics = evaluate_predictions(Y_test, H_pred_test, X_true_test, Vt, seed=seed)
    return val_metrics, test_metrics


def run_mlp_baseline(
    X, H_all, train_nodes, val_nodes, test_nodes, X_true_val, X_true_test, Vt,
    hidden_dim=256, dropout=0.15, lr=1e-3, weight_decay=1e-4,
    max_epochs=300, patience=12, seed=42, device="cpu"
):
    torch.manual_seed(seed)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Y_t = torch.tensor(H_all, dtype=torch.float32, device=device)
    tr = torch.tensor(train_nodes, dtype=torch.bool, device=device)

    model = MLPRegressor(
        in_dim=X.shape[1],
        out_dim=H_all.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_cos = -1e9
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    bad = 0

    def eval_subset(mask_np, X_true_subset):
        model.eval()
        with torch.no_grad():
            pred_all = model(X_t).cpu().numpy()
        H_true = H_all[mask_np]
        H_pred = pred_all[mask_np]
        return evaluate_predictions(H_true, H_pred, X_true_subset, Vt, seed=seed)

    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(X_t)
        loss = F.mse_loss(pred[tr], Y_t[tr])
        loss.backward()
        opt.step()

        if epoch % 20 == 0 or epoch == 1:
            val_metrics = eval_subset(val_nodes, X_true_val)
            print(
                f"Epoch {epoch:03d} | train_mse={loss.item():.6f} | "
                f"VAL Cos_H={val_metrics['Cos_H']:.4f} | "
                f"VAL SpΔ={val_metrics['Spearman_Delta_mean']:.4f}"
            )

            if val_metrics["Cos_H"] > best_val_cos + 1e-4:
                best_val_cos = val_metrics["Cos_H"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print("Early stopping on validation.")
                    break

    model.load_state_dict(best_state)
    val_metrics = eval_subset(val_nodes, X_true_val)
    test_metrics = eval_subset(test_nodes, X_true_test)
    return val_metrics, test_metrics


# def run_gcn_baseline(
#     X, edge_index, edge_weight, H_all, train_nodes, val_nodes, test_nodes,
#     X_true_val, X_true_test, Vt, hidden_dim=256, dropout=0.15,
#     lr=1e-3, weight_decay=1e-4, max_epochs=300, patience=12,
#     seed=42, device="cpu"
# ):
def run_gcn_baseline(
    X,
    edge_index,
    edge_weight,
    H_all,
    X_node,
    Vt,
    train_nodes,
    val_nodes,
    test_nodes,
    X_true_val,
    X_true_test,
    hidden_dim=256,
    dropout=0.15,
    lr=1e-3,
    weight_decay=1e-4,
    max_epochs=300,
    patience=12,
    seed=42,
    device="cpu",
    use_dual_loss=False,
    loss_alpha_latent=1.0,
    loss_beta_gene=1.0,
    loss_gamma_cosine=0.2,
):
    torch.manual_seed(seed)

    x_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(H_all, dtype=torch.float32, device=device)
    x_true_t = torch.tensor(X_node, dtype=torch.float32, device=device)
    vt_t = torch.tensor(Vt, dtype=torch.float32, device=device)

    ei_t = edge_index.to(device)
    ew_t = edge_weight.to(device)
    tr = torch.tensor(train_nodes, dtype=torch.bool, device=device)

    model = WeightedGCN_MLP(
        in_dim=X.shape[1],
        hidden_dim=hidden_dim,
        out_dim=H_all.shape[1],
        dropout=dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_cos = -1e9
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    bad = 0

    def eval_subset(mask_np, X_true_subset):
        model.eval()
        with torch.no_grad():
            pred_all = model(x_t, ei_t, ew_t).cpu().numpy()
        H_true = H_all[mask_np]
        H_pred = pred_all[mask_np]
        return evaluate_predictions(H_true, H_pred, X_true_subset, Vt, seed=seed)

    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(x_t, ei_t, ew_t)

        if use_dual_loss:
            latent_loss = F.mse_loss(pred[tr], y_t[tr])

            x_pred = pred @ vt_t
            gene_loss = F.mse_loss(x_pred[tr], x_true_t[tr])

            cos_loss = cosine_loss_rows(x_pred[tr], x_true_t[tr])

            loss = (
                loss_alpha_latent * latent_loss
                + loss_beta_gene * gene_loss
                + loss_gamma_cosine * cos_loss
            )
        else:
            latent_loss = F.mse_loss(pred[tr], y_t[tr])
            gene_loss = torch.tensor(0.0, device=device)
            cos_loss = torch.tensor(0.0, device=device)
            loss = latent_loss
        loss.backward()
        opt.step()

        if epoch % 20 == 0 or epoch == 1:
            val_metrics = eval_subset(val_nodes, X_true_val)
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={loss.item():.6f} | "
                f"latent={latent_loss.item():.6f} | "
                f"gene={gene_loss.item():.6f} | "
                f"cos={cos_loss.item():.6f} | "
                f"VAL Cos_H={val_metrics['Cos_H']:.4f} | "
                f"VAL SpΔ={val_metrics['Spearman_Delta_mean']:.4f}"
            )

            if val_metrics["Cos_H"] > best_val_cos + 1e-4:
                best_val_cos = val_metrics["Cos_H"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print("Early stopping on validation.")
                    break

    model.load_state_dict(best_state)
    val_metrics = eval_subset(val_nodes, X_true_val)
    test_metrics = eval_subset(test_nodes, X_true_test)
    return val_metrics, test_metrics


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Stable-Shift benchmark with TF graph support")

    parser.add_argument("--h5ad", type=str, required=True)
    parser.add_argument("--string_links", type=str, required=True)
    parser.add_argument("--string_info", type=str, required=True)
    parser.add_argument("--dorothea_file", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="stable_shift_bench_tf_out")

    parser.add_argument("--obs_gene_col", type=str, default="gene")
    parser.add_argument("--var_gene_col", type=str, default="gene_name")
    parser.add_argument("--ctrl_label", type=str, default="non-targeting")
    parser.add_argument("--min_cells_per_pert", type=int, default=50)

    parser.add_argument("--model_type", type=str, default="gcn",
                        choices=["mean", "knn", "ridge", "mlp", "gcn"])

    parser.add_argument("--graph_type", type=str, default="STRING_only",
                        choices=[
                            "none",
                            "STRING_only",
                            "COEXPR_only",
                            "STRING_plus_COEXPR",
                            "TF_only",
                            "STRING_plus_TF",
                        ])

    parser.add_argument("--use_bio_features", action="store_true")

    parser.add_argument("--svd_dim", type=int, default=128)
    parser.add_argument("--node2vec_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=12)

    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--val_frac", type=float, default=0.15)

    parser.add_argument("--string_min_score", type=int, default=700)
    parser.add_argument("--coexpr_top_k", type=int, default=20)
    parser.add_argument("--coexpr_min_abs_corr", type=float, default=0.08)
    parser.add_argument("--max_ctrl_cells", type=int, default=8000)
    parser.add_argument("--positive_only", action="store_true")

    parser.add_argument("--dorothea_mode", type=str, default="curated_clean",
                        choices=["curated_clean", "curated_plus", "all_noisy", "signed_only"])
    parser.add_argument("--dorothea_min_support", type=int, default=1)

    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--go_file", type=str, default=None)
    parser.add_argument("--use_go_features", action="store_true")
    parser.add_argument("--go_gene_col", type=str, default="gene")
    parser.add_argument("--go_term_col", type=str, default="go_id")
    parser.add_argument("--go_sep", type=str, default="\t")
    parser.add_argument("--go_min_genes_per_term", type=int, default=5)
    parser.add_argument("--go_max_genes_per_term", type=int, default=500)
    parser.add_argument("--go_svd_dim", type=int, default=64)
    parser.add_argument("--pathway_file", type=str, default=None)
    parser.add_argument("--use_pathway_features", action="store_true")
    parser.add_argument("--pathway_format", type=str, default="gmt", choices=["gmt"])
    parser.add_argument("--pathway_min_genes_per_term", type=int, default=5)
    parser.add_argument("--pathway_max_genes_per_term", type=int, default=300)
    parser.add_argument("--pathway_svd_dim", type=int, default=32)
    
    parser.add_argument("--n2v_walk_length", type=int, default=30)
    parser.add_argument("--n2v_num_walks", type=int, default=100)
    parser.add_argument("--n2v_window", type=int, default=10)
    parser.add_argument("--n2v_p", type=float, default=1.0)
    parser.add_argument("--n2v_q", type=float, default=1.0)
    parser.add_argument("--n2v_workers", type=int, default=4)
    parser.add_argument("--use_dual_loss", action="store_true")
    parser.add_argument("--loss_alpha_latent", type=float, default=1.0)
    parser.add_argument("--loss_beta_gene", type=float, default=1.0)
    parser.add_argument("--loss_gamma_cosine", type=float, default=0.2)

    args = parser.parse_args()

    ensure_dir(args.outdir)
    cache_dir = os.path.join(args.outdir, "cache")
    ensure_dir(cache_dir)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if args.model_type == "gcn" and args.graph_type == "none":
        raise ValueError("GCN requires a graph_type other than 'none'.")

    if args.model_type in ["knn", "ridge", "mlp"] and args.graph_type == "none" and not args.use_bio_features:
        raise ValueError(
            "For knn/ridge/mlp with graph_type=none, use --use_bio_features or provide a graph."
        )

    if args.graph_type in ["TF_only", "STRING_plus_TF"] and args.dorothea_file is None:
        raise ValueError("TF graph requested but --dorothea_file was not provided.")

    # ------------------------------------------------
    # load data
    # ------------------------------------------------
    print(f"Reading AnnData: {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    edges_gene = load_string_edges_from_raw(args.string_links, args.string_info)

    # ------------------------------------------------
    # perturbation matrix
    # ------------------------------------------------
    X_pert, pert_labels, control_mean = build_pseudobulk_by_perturbation(
        adata=adata,
        obs_gene_col=args.obs_gene_col,
        ctrl_label=args.ctrl_label,
        min_cells_per_pert=args.min_cells_per_pert,
    )

    pert_labels_arr = np.asarray(list(map(str, pert_labels)))
    pert_set = set(pert_labels_arr)

    train_gene_set, val_gene_set, test_gene_set = make_node_splits(
        pert_labels_arr=pert_labels_arr,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
        random_state=args.seed,
    )

    train_row_mask = np.isin(pert_labels_arr, list(train_gene_set))
    val_row_mask = np.isin(pert_labels_arr, list(val_gene_set))
    test_row_mask = np.isin(pert_labels_arr, list(test_gene_set))

    print(
        "Train perts:", int(train_row_mask.sum()),
        "Val perts:", int(val_row_mask.sum()),
        "Test perts:", int(test_row_mask.sum())
    )

    svd, H_all = build_svd_targets_train_only(
        X_pert=X_pert,
        train_row_mask=train_row_mask,
        n_components=args.svd_dim,
        seed=args.seed,
    )

    node_list = sorted(list(pert_set))
    node2i = {g: i for i, g in enumerate(node_list)}
    K = H_all.shape[1]
    print("Pert nodes:", len(node_list))

    pert2row = {g: i for i, g in enumerate(pert_labels_arr)}

    Y = np.zeros((len(node_list), K), dtype=np.float32)
    for g, i in node2i.items():
        if g in pert2row:
            Y[i] = H_all[pert2row[g]]
    
    X_node = np.zeros((len(node_list), X_pert.shape[1]), dtype=np.float32)
    for g, i in node2i.items():
        if g in pert2row:
            X_node[i] = X_pert[pert2row[g]]

    train_nodes = np.array([g in train_gene_set for g in node_list], dtype=bool)
    val_nodes = np.array([g in val_gene_set for g in node_list], dtype=bool)
    test_nodes = np.array([g in test_gene_set for g in node_list], dtype=bool)

    print(
        "Train nodes:", int(train_nodes.sum()),
        "Val nodes:", int(val_nodes.sum()),
        "Test nodes:", int(test_nodes.sum())
    )

    val_gene_list = [g for g, is_v in zip(node_list, val_nodes) if is_v]
    test_gene_list = [g for g, is_t in zip(node_list, test_nodes) if is_t]

    val_rows = [pert2row[g] for g in val_gene_list]
    test_rows = [pert2row[g] for g in test_gene_list]

    X_true_val = np.asarray(X_pert[val_rows])
    X_true_test = np.asarray(X_pert[test_rows])
    Vt = svd.components_

    # ------------------------------------------------
    # graphs
    # ------------------------------------------------
    string_edges = build_string_edges(edges_gene, pert_set, min_score=args.string_min_score)

    coexp_edges = build_coexpression_edges(
        adata=adata,
        pert_genes=pert_labels_arr,
        ctrl_label=args.ctrl_label,
        obs_gene_col=args.obs_gene_col,
        var_gene_col=args.var_gene_col,
        top_k=args.coexpr_top_k,
        min_abs_corr=args.coexpr_min_abs_corr,
        max_ctrl_cells=args.max_ctrl_cells,
        edge_score_scale=(700, 950),
        positive_only=args.positive_only,
    )

    tf_edges = None
    if args.dorothea_file is not None:
        tf_edges = build_dorothea_tf_edges(
            dorothea_file=args.dorothea_file,
            pert_set=pert_set,
            mode=args.dorothea_mode,
            min_support=args.dorothea_min_support,
        )

    both_edges = merge_edge_tables([string_edges, coexp_edges])
    string_tf_edges = None
    if tf_edges is not None:
        string_tf_edges = merge_edge_tables([string_edges, tf_edges])

    print("\nEdge summary")
    print("STRING only:", string_edges.shape)
    print("Coexpression only:", coexp_edges.shape)
    print("STRING + Coexpression:", both_edges.shape)
    if tf_edges is not None:
        print("TF only:", tf_edges.shape)
        print("STRING + TF:", string_tf_edges.shape)

    # ------------------------------------------------
    # bio features
    # ------------------------------------------------
    bio_feat_df = build_biological_features(
        adata=adata,
        node_list=node_list,
        string_edges=string_edges,
        coexp_edges=coexp_edges,
        tf_edges=tf_edges,
        ctrl_label=args.ctrl_label,
        obs_gene_col=args.obs_gene_col,
        var_gene_col=args.var_gene_col,
        max_ctrl_cells=args.max_ctrl_cells,
    )
    bio_feat_df.to_csv(os.path.join(args.outdir, "biological_features_enriched.csv"))
    
    # ------------------------------------------------
    # GO features
    # ------------------------------------------------
    go_feat_df = None
    if args.use_go_features:
        if args.go_file is None:
            raise ValueError("use_go_features=True but --go_file was not provided.")

        go_feat_df = build_go_embedding(
            go_file=args.go_file,
            node_list=node_list,
            gene_col=args.go_gene_col,
            go_col=args.go_term_col,
            sep=args.go_sep,
            min_genes_per_go=args.go_min_genes_per_term,
            max_genes_per_go=args.go_max_genes_per_term,
            go_svd_dim=args.go_svd_dim,
            random_state=args.seed,
        )

        go_feat_df.to_csv(os.path.join(args.outdir, "go_embedding_features.csv"))

    # ------------------------------------------------
    # pathway features
    # ------------------------------------------------
    pathway_feat_df = None
    if args.use_pathway_features:
        if args.pathway_file is None:
            raise ValueError("use_pathway_features=True but --pathway_file was not provided.")

        if args.pathway_format == "gmt":
            pathway_feat_df = build_pathway_embedding_from_gmt(
                pathway_file=args.pathway_file,
                node_list=node_list,
                min_genes_per_pathway=args.pathway_min_genes_per_term,
                max_genes_per_pathway=args.pathway_max_genes_per_term,
                pathway_svd_dim=args.pathway_svd_dim,
                random_state=args.seed,
            )
        else:
            raise ValueError(f"Unsupported pathway_format: {args.pathway_format}")

        pathway_feat_df.to_csv(os.path.join(args.outdir, "pathway_embedding_features.csv"))
    
    # ------------------------------------------------
    # choose graph
    # ------------------------------------------------
    edge_df = None
    if args.graph_type == "none":
        edge_df = None
    elif args.graph_type == "STRING_only":
        edge_df = string_edges
    elif args.graph_type == "COEXPR_only":
        edge_df = coexp_edges
    elif args.graph_type == "STRING_plus_COEXPR":
        edge_df = both_edges
    elif args.graph_type == "TF_only":
        edge_df = tf_edges
    elif args.graph_type == "STRING_plus_TF":
        edge_df = string_tf_edges
    else:
        raise ValueError(f"Unknown graph_type: {args.graph_type}")

    # ------------------------------------------------
    # features
    # ------------------------------------------------
    X_graph = None
    edge_index = None
    edge_weight = None

    if args.model_type in ["gcn", "mlp", "knn", "ridge"] and edge_df is not None:
        X_graph = get_or_build_node2vec_cache(
            cache_dir=cache_dir,
            graph_name=f"{args.graph_type}_{args.dorothea_mode}",
            edge_df=edge_df,
            node_list=node_list,
            node2i=node2i,
            dim=args.node2vec_dim,
            walk_length=args.n2v_walk_length,
            num_walks=args.n2v_num_walks,
            window=args.n2v_window,
            p=args.n2v_p,
            q=args.n2v_q,
            workers=args.n2v_workers,
            seed=args.seed,
        )
        edge_index, edge_weight = make_edge_index_and_weight(edge_df, node_list, node2i)

    X = None
    if args.model_type != "mean":
        X = assemble_features(
            X_graph=X_graph,
            bio_feat_df=bio_feat_df,
            go_feat_df=go_feat_df,
            pathway_feat_df=pathway_feat_df,
            node_list=node_list,
            use_bio_features=args.use_bio_features,
            use_go_features=args.use_go_features,
            use_pathway_features=args.use_pathway_features,
        )

    # ------------------------------------------------
    # run
    # ------------------------------------------------
    run_name = f"{args.model_type}__{args.graph_type}__bio{int(args.use_bio_features)}"
    if args.graph_type in ["TF_only", "STRING_plus_TF"]:
        run_name += f"__{args.dorothea_mode}"
    run_name += f"__go{int(args.use_go_features)}"
    run_name += f"__path{int(args.use_pathway_features)}"
    run_name += f"__n2vd{args.node2vec_dim}"
    run_name += f"__n2vp{args.n2v_p}"
    run_name += f"__n2vq{args.n2v_q}"
    run_name += f"__dual{int(args.use_dual_loss)}"
    run_name += f"__seed{args.seed}"

    print("\n" + "=" * 80)
    print("Running:", run_name)
    print("=" * 80)

    if args.model_type == "mean":
        val_metrics, test_metrics = run_mean_baseline(
            H_all=Y,
            train_nodes=train_nodes,
            val_nodes=val_nodes,
            test_nodes=test_nodes,
            X_true_val=X_true_val,
            X_true_test=X_true_test,
            Vt=Vt,
            seed=args.seed,
        )

    elif args.model_type == "knn":
        model = KNeighborsRegressor(n_neighbors=args.knn_k, weights="distance", metric="euclidean")
        val_metrics, test_metrics = run_sklearn_baseline(
            model=model,
            X=X,
            H_all=Y,
            train_nodes=train_nodes,
            val_nodes=val_nodes,
            test_nodes=test_nodes,
            X_true_val=X_true_val,
            X_true_test=X_true_test,
            Vt=Vt,
            seed=args.seed,
        )

    elif args.model_type == "ridge":
        model = Ridge(alpha=args.ridge_alpha)
        val_metrics, test_metrics = run_sklearn_baseline(
            model=model,
            X=X,
            H_all=Y,
            train_nodes=train_nodes,
            val_nodes=val_nodes,
            test_nodes=test_nodes,
            X_true_val=X_true_val,
            X_true_test=X_true_test,
            Vt=Vt,
            seed=args.seed,
        )

    elif args.model_type == "mlp":
        val_metrics, test_metrics = run_mlp_baseline(
            X=X,
            H_all=Y,
            train_nodes=train_nodes,
            val_nodes=val_nodes,
            test_nodes=test_nodes,
            X_true_val=X_true_val,
            X_true_test=X_true_test,
            Vt=Vt,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
            device=device,
        )

    elif args.model_type == "gcn":
        val_metrics, test_metrics = run_gcn_baseline(
        X=X,
        edge_index=edge_index,
        edge_weight=edge_weight,
        H_all=Y,
        X_node=X_node,
        Vt=Vt,
        train_nodes=train_nodes,
        val_nodes=val_nodes,
        test_nodes=test_nodes,
        X_true_val=X_true_val,
        X_true_test=X_true_test,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        device=device,
        use_dual_loss=args.use_dual_loss,
        loss_alpha_latent=args.loss_alpha_latent,
        loss_beta_gene=args.loss_beta_gene,
        loss_gamma_cosine=args.loss_gamma_cosine,
    )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    print("\nFinal validation metrics")
    for k in sorted(val_metrics.keys()):
        print(f"{k}: {val_metrics[k]:.4f}")

    print("\nFinal TEST metrics")
    for k in sorted(test_metrics.keys()):
        print(f"{k}: {test_metrics[k]:.4f}")

    result = {
        "run_name": run_name,
        "model_type": args.model_type,
        "graph_type": args.graph_type,
        "use_bio_features": bool(args.use_bio_features),
        "dorothea_mode": args.dorothea_mode if args.graph_type in ["TF_only", "STRING_plus_TF"] else None,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(os.path.join(args.outdir, f"{run_name}.json"), "w") as f:
        json.dump(result, f, indent=2)

    row = {
        "run_name": run_name,
        "seed": args.seed,
        "model_type": args.model_type,
        "graph_type": args.graph_type,
        "use_bio_features": int(args.use_bio_features),
        "use_go_features": int(args.use_go_features),
        "use_pathway_features": int(args.use_pathway_features),
        "dorothea_mode": args.dorothea_mode if args.graph_type in ["TF_only", "STRING_plus_TF"] else "",
        "VAL_Cos_H": val_metrics["Cos_H"],
        "VAL_R2_H": val_metrics["R2_H"],
        "VAL_Spearman_Delta_mean": val_metrics["Spearman_Delta_mean"],
        "TEST_Cos_H": test_metrics["Cos_H"],
        "TEST_R2_H": test_metrics["R2_H"],
        "TEST_Spearman_Delta_mean": test_metrics["Spearman_Delta_mean"],
        "TEST_Prec@50_UP": test_metrics["Prec@50_UP"],
        "TEST_Prec@50_DN": test_metrics["Prec@50_DN"],
        "node2vec_dim": args.node2vec_dim,
        "n2v_p": args.n2v_p,
        "n2v_q": args.n2v_q,
        "use_dual_loss": int(args.use_dual_loss),
        "loss_alpha_latent": args.loss_alpha_latent,
        "loss_beta_gene": args.loss_beta_gene,
        "loss_gamma_cosine": args.loss_gamma_cosine,
    }

    csv_path = os.path.join(args.outdir, "benchmark_summary.csv")
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path)
        old = old[old["run_name"] != run_name]
        new = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    else:
        new = pd.DataFrame([row])

    new.to_csv(csv_path, index=False)
    print(f"\nSaved summary row to: {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()