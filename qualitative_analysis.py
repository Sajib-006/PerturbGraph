#!/usr/bin/env python3

import os
import json
import random
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from node2vec import Node2Vec
from torch_geometric.nn import GCNConv
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

# ============================================================
# plotting style
# ============================================================
mpl.rcParams.update({
    "figure.dpi": 220,
    "savefig.dpi": 500,
    "font.size": 17,
    "axes.titlesize": 21,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.titlesize": 22,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# journal-style muted palette
TRUE_COLOR = "#2B6CB0"
PRED_COLOR = "#C53030"
NEUTRAL_COLOR = "#4A5568"
ACCENT_GREEN = "#2F855A"
ACCENT_PURPLE = "#6B46C1"
ACCENT_ORANGE = "#DD6B20"
ACCENT_GOLD = "#B7791F"
LIGHT_BLUE = "#90CDF4"
LIGHT_RED = "#FEB2B2"
LIGHT_GRAY = "#CBD5E0"

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

def safe_spearman(a, b, eps=1e-12):
    if np.std(a) < eps or np.std(b) < eps:
        return 0.0
    rho = spearmanr(a, b).correlation
    return 0.0 if np.isnan(rho) else float(rho)

def row_cosine(a, b, eps=1e-8):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))

def row_pearson(a, b, eps=1e-12):
    if np.std(a) < eps or np.std(b) < eps:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def mean_abs_error(a, b):
    return float(np.mean(np.abs(a - b)))

def signal_norm(x):
    return float(np.linalg.norm(x))

def topk_sets(vec, k=50):
    up = set(np.argsort(vec)[-k:])
    dn = set(np.argsort(vec)[:k])
    return up, dn

def precision_recall_f1_from_sets(true_set, pred_set):
    inter = len(true_set & pred_set)
    p = inter / max(1, len(pred_set))
    r = inter / max(1, len(true_set))
    if p + r == 0:
        f1 = 0.0
    else:
        f1 = 2 * p * r / (p + r)
    return p, r, f1, inter

def signed_overlap_metrics(true_vec, pred_vec, k=50):
    true_up, true_dn = topk_sets(true_vec, k)
    pred_up, pred_dn = topk_sets(pred_vec, k)

    p_up, r_up, f1_up, inter_up = precision_recall_f1_from_sets(true_up, pred_up)
    p_dn, r_dn, f1_dn, inter_dn = precision_recall_f1_from_sets(true_dn, pred_dn)

    return {
        "topk_up_overlap": inter_up,
        "topk_dn_overlap": inter_dn,
        "topk_up_precision": p_up,
        "topk_dn_precision": p_dn,
        "topk_up_recall": r_up,
        "topk_dn_recall": r_dn,
        "topk_up_f1": f1_up,
        "topk_dn_f1": f1_dn,
    }

def savefig_clean(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=500)
    plt.close()

# ============================================================
# data loading
# ============================================================
def load_string_edges_from_raw(string_links_path, string_info_path):
    print(f"Reading STRING links: {string_links_path}")
    print(f"Reading STRING info : {string_info_path}")

    links = pd.read_csv(string_links_path, sep=r"\s+")
    info = pd.read_csv(string_info_path, sep="\t")

    protein_to_gene = dict(zip(info["#string_protein_id"], info["preferred_name"]))

    g1 = links["protein1"].map(protein_to_gene)
    g2 = links["protein2"].map(protein_to_gene)
    score = links["combined_score"].astype(int)

    edges_gene = pd.DataFrame({
        "gene1": g1,
        "gene2": g2,
        "score": score
    }).dropna()

    edges_gene = edges_gene[edges_gene["gene1"] != edges_gene["gene2"]].copy()
    print("Edges after mapping:", edges_gene.shape)
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
        if g == ctrl_label:
            continue
        mask = labels_all == g
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
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    H_train = svd.fit_transform(X_pert[train_row_mask]).astype(np.float32)
    H_all = svd.transform(X_pert).astype(np.float32)
    print("H_train shape:", H_train.shape)
    print("ExplainedVarSum:", float(svd.explained_variance_ratio_.sum()))
    return svd, H_all

# ============================================================
# graph / features
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
    print(f"\nLoading GO annotations from: {go_file}")
    df = pd.read_csv(go_file, sep=sep)
    df[gene_col] = df[gene_col].astype(str)
    df[go_col] = df[go_col].astype(str)

    node_set = set(node_list)
    df = df[df[gene_col].isin(node_set)].copy()

    go_sizes = df.groupby(go_col)[gene_col].nunique()
    keep_go = go_sizes[(go_sizes >= min_genes_per_go) & (go_sizes <= max_genes_per_go)].index
    df = df[df[go_col].isin(keep_go)].copy()

    if df.empty:
        cols = [f"go_emb_{i}" for i in range(go_svd_dim)]
        return pd.DataFrame(0.0, index=node_list, columns=cols), {g: 0 for g in node_list}

    go_count_map = df.groupby(gene_col)[go_col].nunique().to_dict()

    gene_cat = pd.Categorical(df[gene_col], categories=node_list)
    go_terms = sorted(df[go_col].unique())
    go_cat = pd.Categorical(df[go_col], categories=go_terms)

    gene_codes = gene_cat.codes
    go_codes = go_cat.codes
    valid = (gene_codes >= 0) & (go_codes >= 0)

    gene_codes = gene_codes[valid]
    go_codes = go_codes[valid]

    M = np.zeros((len(node_list), len(go_terms)), dtype=np.float32)
    M[gene_codes, go_codes] = 1.0

    eff_dim = min(go_svd_dim, max(1, min(M.shape[0] - 1, M.shape[1] - 1)))
    svd = TruncatedSVD(n_components=eff_dim, random_state=random_state)
    G = svd.fit_transform(M).astype(np.float32)

    if eff_dim < go_svd_dim:
        G_pad = np.zeros((len(node_list), go_svd_dim), dtype=np.float32)
        G_pad[:, :eff_dim] = G
        G = G_pad

    scaler = StandardScaler()
    G = scaler.fit_transform(G)

    cols = [f"go_emb_{i}" for i in range(go_svd_dim)]
    go_feat_df = pd.DataFrame(G, index=node_list, columns=cols)

    go_count_map_full = {g: int(go_count_map.get(g, 0)) for g in node_list}
    return go_feat_df, go_count_map_full

def build_biological_features(
    adata,
    node_list,
    string_edges,
    ctrl_label="non-targeting",
    obs_gene_col="gene",
    var_gene_col="gene_name",
    max_ctrl_cells=8000,
):
    print("\nBuilding biological node features...")

    ctrl_mask = adata.obs[obs_gene_col].astype(str).values == ctrl_label
    ctrl_idx = np.where(ctrl_mask)[0]

    if max_ctrl_cells is not None and len(ctrl_idx) > max_ctrl_cells:
        rng = np.random.default_rng(0)
        ctrl_idx = rng.choice(ctrl_idx, size=max_ctrl_cells, replace=False)

    gene_names = (
        adata.var[var_gene_col].astype(str).values
        if var_gene_col in adata.var.columns
        else adata.var_names.astype(str).values
    )

    sym_to_pos = {}
    for j, sym in enumerate(gene_names):
        if sym not in sym_to_pos:
            sym_to_pos[sym] = j

    present_genes = [g for g in node_list if g in sym_to_pos]

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

    string_deg = {g: 0 for g in node_list}
    string_weighted_degree = {g: 0.0 for g in node_list}
    string_pagerank = {g: 0.0 for g in node_list}
    string_clustering = {g: 0.0 for g in node_list}
    neighbor_mean_ctrl_mean = {g: 0.0 for g in node_list}
    neighbor_mean_ctrl_var = {g: 0.0 for g in node_list}
    neighbor_mean_ctrl_detect = {g: 0.0 for g in node_list}

    Gs = nx.Graph()
    Gs.add_nodes_from(node_list)
    for g1, g2, w in string_edges[["gene1", "gene2", "score"]].itertuples(index=False, name=None):
        Gs.add_edge(g1, g2, weight=float(w))
        string_deg[g1] += 1
        string_deg[g2] += 1

    pr = nx.pagerank(Gs, weight="weight")
    clust = nx.clustering(Gs, weight="weight")

    for g in node_list:
        string_weighted_degree[g] = float(Gs.degree(g, weight="weight"))
        string_pagerank[g] = float(pr.get(g, 0.0))
        string_clustering[g] = float(clust.get(g, 0.0))

        nbrs = list(Gs.neighbors(g))
        if len(nbrs) > 0:
            neighbor_mean_ctrl_mean[g] = float(np.mean([ctrl_mean_map[n] for n in nbrs]))
            neighbor_mean_ctrl_var[g] = float(np.mean([ctrl_var_map[n] for n in nbrs]))
            neighbor_mean_ctrl_detect[g] = float(np.mean([ctrl_detect_map[n] for n in nbrs]))

    rows = []
    for g in node_list:
        rows.append({
            "gene": g,
            "ctrl_mean": ctrl_mean_map[g],
            "ctrl_var": ctrl_var_map[g],
            "ctrl_detect_frac": ctrl_detect_map[g],
            "string_degree": float(string_deg[g]),
            "string_weighted_degree": float(string_weighted_degree[g]),
            "string_pagerank": float(string_pagerank[g]),
            "string_clustering": float(string_clustering[g]),
            "neighbor_mean_ctrl_mean": float(neighbor_mean_ctrl_mean[g]),
            "neighbor_mean_ctrl_var": float(neighbor_mean_ctrl_var[g]),
            "neighbor_mean_ctrl_detect_frac": float(neighbor_mean_ctrl_detect[g]),
        })

    bio_raw_df = pd.DataFrame(rows).set_index("gene").loc[node_list]
    bio_feat_df = bio_raw_df.copy()
    scaler = StandardScaler()
    bio_feat_df.loc[:, :] = scaler.fit_transform(bio_feat_df.values)
    print("Biological feature matrix shape:", bio_feat_df.shape)
    return bio_feat_df, bio_raw_df

def make_edge_index_and_weight(edge_df, node2i):
    src = edge_df["gene1"].map(node2i).to_numpy()
    dst = edge_df["gene2"].map(node2i).to_numpy()
    w = edge_df["score"].to_numpy().astype(np.float32)
    w = (w - w.min()) / (w.max() - w.min() + 1e-8)

    edge_index = np.vstack([
        np.concatenate([src, dst]),
        np.concatenate([dst, src])
    ])
    edge_weight = np.concatenate([w, w]).astype(np.float32)

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
    return edge_index, edge_weight

def build_node2vec_features(
    edge_df,
    node_list,
    node2i,
    dim=256,
    walk_length=40,
    num_walks=150,
    window=10,
    p=0.5,
    q=2.0,
    workers=4,
    seed=42,
):
    G = nx.Graph()
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_df[["gene1", "gene2"]].itertuples(index=False, name=None))

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
    for g, i in node2i.items():
        if g in emb_dict:
            X_graph[i] = emb_dict[g].astype(np.float32)
    return X_graph

def build_string_graph(edge_df, node_list):
    G = nx.Graph()
    G.add_nodes_from(node_list)
    for g1, g2, s in edge_df[["gene1", "gene2", "score"]].itertuples(index=False, name=None):
        G.add_edge(str(g1), str(g2), weight=float(s))
    return G

def compute_train_graph_context_features(G, node_list, train_gene_set):
    train_gene_set = set(map(str, train_gene_set))

    train_neighbor_count = {}
    nearest_train_distance = {}

    for g in node_list:
        if g not in G:
            train_neighbor_count[g] = 0
            nearest_train_distance[g] = np.nan
            continue

        nbrs = list(G.neighbors(g))
        train_neighbor_count[g] = int(sum(1 for n in nbrs if n in train_gene_set))

        if g in train_gene_set:
            nearest_train_distance[g] = 0
            continue

        best_d = np.inf
        for tg in train_gene_set:
            if tg in G:
                try:
                    d = nx.shortest_path_length(G, source=g, target=tg)
                    if d < best_d:
                        best_d = d
                except nx.NetworkXNoPath:
                    pass
        nearest_train_distance[g] = float(best_d) if np.isfinite(best_d) else np.nan

    return train_neighbor_count, nearest_train_distance

# ============================================================
# model
# ============================================================
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

# ============================================================
# better plotting helpers
# ============================================================
def select_case_indices(per_gene_df, signal_quantile=0.30):
    sig_thr = per_gene_df["true_norm"].quantile(signal_quantile)
    sub = per_gene_df[per_gene_df["true_norm"] >= sig_thr].copy()
    if len(sub) < 3:
        sub = per_gene_df.copy()

    sub = sub.sort_values("cosine", ascending=False).reset_index(drop=True)
    best_gene = sub.iloc[0]["perturbed_gene"]
    median_gene = sub.iloc[len(sub) // 2]["perturbed_gene"]
    worst_gene = sub.iloc[-1]["perturbed_gene"]
    return best_gene, median_gene, worst_gene

def boxplot_with_jitter(groups, labels, ylabel, title, outpath, colors=None):
    plt.figure(figsize=(8.6, 6.2))
    ax = plt.gca()
    bp = ax.boxplot(
        groups,
        labels=labels,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(color=NEUTRAL_COLOR, linewidth=1.4),
        capprops=dict(color=NEUTRAL_COLOR, linewidth=1.4),
        boxprops=dict(linewidth=1.4, color=NEUTRAL_COLOR),
    )

    if colors is None:
        colors = [LIGHT_BLUE] * len(groups)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.9)

    rng = np.random.default_rng(42)
    for i, vals in enumerate(groups, start=1):
        vals = np.asarray(vals)
        if len(vals) == 0:
            continue
        x = rng.normal(i, 0.05, size=len(vals))
        plt.scatter(x, vals, s=18, alpha=0.55, color=ACCENT_ORANGE, edgecolors="white", linewidths=0.25)

    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.18)
    savefig_clean(outpath)

def plot_metric_by_distance_bins(df, outpath):
    x = df["nearest_train_distance"].copy()
    cats = []
    for v in x:
        if pd.isna(v):
            cats.append("No path")
        elif v == 1:
            cats.append("1-hop")
        elif v == 2:
            cats.append("2-hop")
        elif v == 3:
            cats.append("3-hop")
        elif v > 3:
            cats.append(">3-hop")
        else:
            cats.append("No path")

    tmp = df.copy()
    tmp["distance_bin"] = cats
    order = ["1-hop", "2-hop", "3-hop", ">3-hop", "No path"]

    groups = [tmp.loc[tmp["distance_bin"] == k, "cosine"].values for k in order]
    boxplot_with_jitter(
        groups=groups,
        labels=order,
        ylabel="Per-gene cosine",
        title="Prediction quality vs graph distance to training genes",
        outpath=outpath,
        colors=[LIGHT_BLUE, "#BEE3F8", "#FBD38D", "#F6AD55", LIGHT_RED],
    )

def plot_metric_by_degree_bins(df, outpath):
    degree = df["string_degree"].values
    bins = [0, 10, 30, 60, 120, np.inf]
    labels = ["0-10", "11-30", "31-60", "61-120", ">120"]
    cats = pd.cut(degree, bins=bins, labels=labels, include_lowest=True)
    groups = [df.loc[cats == lab, "cosine"].values for lab in labels]
    boxplot_with_jitter(
        groups=groups,
        labels=labels,
        ylabel="Per-gene cosine",
        title="Prediction quality vs STRING degree",
        outpath=outpath,
        colors=["#E6FFFA", "#B2F5EA", "#90CDF4", "#63B3ED", "#4299E1"],
    )

def plot_metric_by_neighbor_bins(df, outpath):
    nn = df["train_neighbor_count"].values
    bins = [-0.1, 0.5, 5.5, 20.5, 50.5, np.inf]
    labels = ["0", "1-5", "6-20", "21-50", ">50"]
    cats = pd.cut(nn, bins=bins, labels=labels, include_lowest=True)
    groups = [df.loc[cats == lab, "cosine"].values for lab in labels]
    boxplot_with_jitter(
        groups=groups,
        labels=labels,
        ylabel="Per-gene cosine",
        title="Prediction quality vs local training-neighbor support",
        outpath=outpath,
        colors=["#FEEBC8", "#FBD38D", "#F6AD55", "#ED8936", "#DD6B20"],
    )

def plot_histogram(values, xlabel, title, outpath, bins=25, color=ACCENT_PURPLE):
    plt.figure(figsize=(7.2, 5.4))
    plt.hist(values, bins=bins, color=color, alpha=0.88, edgecolor="white")
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(axis="y", alpha=0.18)
    savefig_clean(outpath)

def plot_true_vs_pred_scatter(true_vec, pred_vec, pert_gene, outpath, annotate_stats=None):
    lim_min = min(float(true_vec.min()), float(pred_vec.min()))
    lim_max = max(float(true_vec.max()), float(pred_vec.max()))
    margin = 0.05 * (lim_max - lim_min + 1e-8)

    plt.figure(figsize=(7.1, 6.3))
    plt.scatter(true_vec, pred_vec, s=14, alpha=0.32, color=ACCENT_PURPLE, edgecolors="none")
    plt.plot([lim_min - margin, lim_max + margin], [lim_min - margin, lim_max + margin],
             linestyle="--", linewidth=1.8, color=NEUTRAL_COLOR)

    plt.xlabel("True expression shift")
    plt.ylabel("Predicted expression shift")
    plt.title(f"{pert_gene}: true vs predicted gene program")
    plt.grid(alpha=0.16)

    if annotate_stats is not None:
        txt = (
            f"Cosine = {annotate_stats['cosine']:.3f}\n"
            f"Pearson = {annotate_stats['pearson']:.3f}\n"
            f"Spearman = {annotate_stats['spearman']:.3f}\n"
            f"Top-{50} overlap U/D = "
            f"{int(annotate_stats['topk_up_overlap'])}/{int(annotate_stats['topk_dn_overlap'])}"
        )
        plt.text(
            0.03, 0.97, txt,
            transform=plt.gca().transAxes,
            va="top", ha="left",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.94, edgecolor="0.8")
        )

    savefig_clean(outpath)

def get_case_union_indices(true_vec, pred_vec, top_n_each=10):
    true_up = list(np.argsort(true_vec)[-top_n_each:])
    true_dn = list(np.argsort(true_vec)[:top_n_each])
    pred_up = list(np.argsort(pred_vec)[-top_n_each:])
    pred_dn = list(np.argsort(pred_vec)[:top_n_each])
    idx = sorted(
        set(true_up + true_dn + pred_up + pred_dn),
        key=lambda j: (max(abs(true_vec[j]), abs(pred_vec[j])), true_vec[j])
    )
    return np.array(idx, dtype=int)

def plot_case_study_lollipop(true_vec, pred_vec, gene_names, pert_gene, outpath, top_n_each=10):
    idx = get_case_union_indices(true_vec, pred_vec, top_n_each=top_n_each)
    labels = [gene_names[j] for j in idx]

    true_vals = true_vec[idx]
    pred_vals = pred_vec[idx]

    order = np.argsort(true_vals)
    true_vals = true_vals[order]
    pred_vals = pred_vals[order]
    labels = [labels[j] for j in order]

    y = np.arange(len(labels))
    plt.figure(figsize=(10.0, max(6.5, 0.36 * len(labels))))

    for yi, tv, pv in zip(y, true_vals, pred_vals):
        plt.plot([tv, pv], [yi, yi], color=LIGHT_GRAY, linewidth=1.2, zorder=1)

    plt.scatter(true_vals, y, s=55, color=TRUE_COLOR, label="True", zorder=3)
    plt.scatter(pred_vals, y, s=55, color=PRED_COLOR, label="Predicted", zorder=3)

    plt.axvline(0, color="black", linewidth=1.0)
    plt.yticks(y, labels)
    plt.xlabel("Expression shift")
    plt.title(f"{pert_gene}: top DE genes (true vs predicted)")
    plt.legend(frameon=False, loc="lower right")
    plt.grid(axis="x", alpha=0.18)
    savefig_clean(outpath)

def plot_topk_overlap_counts(true_vec, pred_vec, pert_gene, outpath, k=50):
    m = signed_overlap_metrics(true_vec, pred_vec, k=k)
    labels = ["Up overlap", "Down overlap"]
    values = [m["topk_up_overlap"], m["topk_dn_overlap"]]
    colors = [TRUE_COLOR, PRED_COLOR]

    ymax = max(values + [1])
    ypad = max(1.0, 0.12 * ymax)
    ytop = ymax + ypad

    plt.figure(figsize=(5.8, 5.0))
    bars = plt.bar(labels, values, color=colors, width=0.48, alpha=0.92)

    for b, v in zip(bars, values):
        plt.text(
            b.get_x() + b.get_width() / 2,
            v + 0.18 * ypad,
            f"{int(v)}",
            ha="center",
            va="bottom",
            fontsize=13
        )

    plt.ylim(0, ytop)
    plt.ylabel(f"Recovered genes in top-{k}")
    plt.title(f"{pert_gene}: top-{k} overlap", pad=16)
    plt.grid(axis="y", alpha=0.18)
    savefig_clean(outpath)

def plot_topk_prf(true_vec, pred_vec, pert_gene, outpath, k=50):
    m = signed_overlap_metrics(true_vec, pred_vec, k=k)

    labels = ["Up precision", "Up recall", "Up F1", "Down precision", "Down recall", "Down F1"]
    values = [
        m["topk_up_precision"], m["topk_up_recall"], m["topk_up_f1"],
        m["topk_dn_precision"], m["topk_dn_recall"], m["topk_dn_f1"]
    ]
    colors = [LIGHT_BLUE, "#63B3ED", TRUE_COLOR, LIGHT_RED, "#FC8181", PRED_COLOR]

    plt.figure(figsize=(8.4, 5.2))
    bars = plt.bar(labels, values, color=colors, width=0.55, alpha=0.92)
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=11)

    plt.ylim(0, 1.08)
    plt.ylabel("Score")
    plt.title(f"{pert_gene}: top-{k} precision / recall / F1")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.18)
    savefig_clean(outpath)

def plot_network_neighborhood_compare(target_gene, true_vec, pred_vec, gene_names, edge_df, outpath, top_neighbors=15):
    G = nx.Graph()
    for g1, g2, w in edge_df[["gene1", "gene2", "score"]].itertuples(index=False, name=None):
        G.add_edge(g1, g2, weight=float(w))

    if target_gene not in G:
        return

    nbrs = sorted(G[target_gene].items(), key=lambda x: x[1].get("weight", 0.0), reverse=True)
    nbrs = [n for n, _ in nbrs[:top_neighbors]]
    sub_nodes = [target_gene] + nbrs
    subG = G.subgraph(sub_nodes).copy()

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    def values_for(v):
        out = {}
        for g in sub_nodes:
            out[g] = float(v[gene_to_idx[g]]) if g in gene_to_idx else 0.0
        return out

    true_map = values_for(true_vec)
    pred_map = values_for(pred_vec)

    all_vals = list(true_map.values()) + list(pred_map.values())
    vmax = max(abs(np.min(all_vals)), abs(np.max(all_vals)), 1e-6)

    pos = nx.spring_layout(subG, seed=42)

    fig = plt.figure(figsize=(14.2, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    axes = [ax1, ax2]
    titles = [f"{target_gene}: true local response", f"{target_gene}: predicted local response"]
    value_maps = [true_map, pred_map]

    nodes_artist = None
    for ax, title, value_map in zip(axes, titles, value_maps):
        ax.set_title(title, fontsize=16, pad=14)
        nx.draw_networkx_edges(subG, pos, alpha=0.35, width=1.3, ax=ax)

        node_vals = [1.15 * vmax if g == target_gene else value_map.get(g, 0.0) for g in subG.nodes()]
        nodes_artist = nx.draw_networkx_nodes(
            subG,
            pos,
            node_color=node_vals,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            node_size=[1200 if g == target_gene else 780 for g in subG.nodes()],
            edgecolors="black",
            linewidths=0.9,
            ax=ax,
        )
        nx.draw_networkx_labels(subG, pos, font_size=9, ax=ax)
        ax.axis("off")

    cbar = fig.colorbar(nodes_artist, cax=cax)
    cbar.set_label("Expression shift", rotation=90, labelpad=14, fontsize=15)
    cbar.ax.tick_params(labelsize=12)

    plt.savefig(outpath, bbox_inches="tight", dpi=500)
    plt.close()

def plot_pathway_overlap_bar(shared, true_only, pred_only, pert_gene, outpath):
    labels = ["Shared", "True only", "Pred only"]
    vals = [shared, true_only, pred_only]
    colors = [ACCENT_GREEN, TRUE_COLOR, PRED_COLOR]

    ymax = max(vals + [1])
    ytop = ymax + max(0.8, 0.18 * ymax)

    plt.figure(figsize=(5.8, 4.9))
    bars = plt.bar(labels, vals, color=colors, width=0.48, alpha=0.92)

    for b, v in zip(bars, vals):
        plt.text(
            b.get_x() + b.get_width()/2,
            v + max(0.08, 0.04 * ymax),
            str(int(v)),
            ha="center",
            va="bottom",
            fontsize=12
        )

    plt.ylim(0, ytop)
    plt.ylabel("Top pathway count")
    plt.title(f"{pert_gene}: pathway overlap", pad=14)
    plt.grid(axis="y", alpha=0.18)
    savefig_clean(outpath)

def plot_pathway_comparison_heatmap(true_df, pred_df, pert_gene, outpath, top_n=10):
    true_df = true_df.head(top_n).copy()
    pred_df = pred_df.head(top_n).copy()

    all_paths = list(dict.fromkeys(true_df["pathway"].tolist() + pred_df["pathway"].tolist()))
    all_paths = all_paths[:top_n]

    true_map = {r["pathway"]: r["jaccard"] for _, r in true_df.iterrows()}
    pred_map = {r["pathway"]: r["jaccard"] for _, r in pred_df.iterrows()}

    mat = np.array([[true_map.get(p, 0.0), pred_map.get(p, 0.0)] for p in all_paths], dtype=float)
    vmax = max(0.2, float(mat.max()) if mat.size > 0 else 0.2)

    fig, ax = plt.subplots(figsize=(8.8, max(5.2, 0.46 * len(all_paths))))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0.0, vmax=vmax)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["True top\npathways", "Predicted top\npathways"],
        rotation=0,
        ha="center"
    )
    ax.tick_params(axis="x", pad=10)

    ax.set_yticks(np.arange(len(all_paths)))
    ax.set_yticklabels(all_paths)

    ax.set_title(f"{pert_gene}: pathway recovery", pad=20, fontsize=20)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.05)
    cbar.set_label("Pathway Jaccard score", fontsize=16, labelpad=12)
    cbar.ax.tick_params(labelsize=12)

    plt.savefig(outpath, bbox_inches="tight", dpi=500)
    plt.close()
# ============================================================
# simple pathway enrichment from GMT
# ============================================================
def load_gmt(gmt_file, gene_universe=None, min_size=5, max_size=500):
    pathways = {}
    gene_universe = set(gene_universe) if gene_universe is not None else None

    with open(gmt_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            pname = parts[0]
            genes = set(g.strip() for g in parts[2:] if g.strip())
            if gene_universe is not None:
                genes = genes & gene_universe
            if min_size <= len(genes) <= max_size:
                pathways[pname] = genes
    return pathways

def top_overlap_pathways(selected_genes, pathways, top_n=10):
    selected_genes = set(selected_genes)
    rows = []
    for pname, pgenes in pathways.items():
        inter = len(selected_genes & pgenes)
        if inter > 0:
            jacc = inter / len(selected_genes | pgenes)
            rows.append((pname, inter, jacc, len(pgenes)))
    rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return rows[:top_n]

def pathway_overlap_summary(true_genes, pred_genes, pathways, top_n=10):
    true_top = top_overlap_pathways(true_genes, pathways, top_n=top_n)
    pred_top = top_overlap_pathways(pred_genes, pathways, top_n=top_n)

    true_names = [x[0] for x in true_top]
    pred_names = [x[0] for x in pred_top]

    shared = len(set(true_names) & set(pred_names))
    true_only = len(set(true_names) - set(pred_names))
    pred_only = len(set(pred_names) - set(true_names))

    return true_top, pred_top, shared, true_only, pred_only

# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", type=str, required=True)
    parser.add_argument("--string_links", type=str, required=True)
    parser.add_argument("--string_info", type=str, required=True)
    parser.add_argument("--go_file", type=str, required=True)
    parser.add_argument("--gmt_file", type=str, default=None)
    parser.add_argument("--outdir", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_cells_per_pert", type=int, default=50)
    parser.add_argument("--svd_dim", type=int, default=128)
    parser.add_argument("--node2vec_dim", type=int, default=256)
    parser.add_argument("--go_svd_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--string_min_score", type=int, default=700)

    parser.add_argument("--obs_gene_col", type=str, default="gene")
    parser.add_argument("--var_gene_col", type=str, default="gene_name")
    parser.add_argument("--ctrl_label", type=str, default="non-targeting")

    parser.add_argument("--go_gene_col", type=str, default="gene")
    parser.add_argument("--go_term_col", type=str, default="go_id")
    parser.add_argument("--go_sep", type=str, default="\t")
    parser.add_argument("--go_min_genes_per_term", type=int, default=5)
    parser.add_argument("--go_max_genes_per_term", type=int, default=500)

    parser.add_argument("--n2v_walk_length", type=int, default=40)
    parser.add_argument("--n2v_num_walks", type=int, default=150)
    parser.add_argument("--n2v_window", type=int, default=10)
    parser.add_argument("--n2v_p", type=float, default=0.5)
    parser.add_argument("--n2v_q", type=float, default=2.0)
    parser.add_argument("--n2v_workers", type=int, default=4)

    parser.add_argument("--topk_eval", type=int, default=50)
    parser.add_argument("--case_top_n_each", type=int, default=10)
    parser.add_argument("--network_top_neighbors", type=int, default=15)
    parser.add_argument("--signal_quantile_for_cases", type=float, default=0.30)

    args = parser.parse_args()

    ensure_dir(args.outdir)
    ensure_dir(os.path.join(args.outdir, "plots"))
    ensure_dir(os.path.join(args.outdir, "arrays"))
    ensure_dir(os.path.join(args.outdir, "tables"))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------------------------------------
    # load data
    # ------------------------------------------------
    print(f"Reading AnnData: {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    edges_gene = load_string_edges_from_raw(args.string_links, args.string_info)

    X_pert, pert_labels, _ = build_pseudobulk_by_perturbation(
        adata=adata,
        obs_gene_col=args.obs_gene_col,
        ctrl_label=args.ctrl_label,
        min_cells_per_pert=args.min_cells_per_pert,
    )

    gene_names = (
        adata.var[args.var_gene_col].astype(str).tolist()
        if args.var_gene_col in adata.var.columns
        else adata.var_names.astype(str).tolist()
    )

    pert_labels_arr = np.asarray(list(map(str, pert_labels)))
    pert_set = set(pert_labels_arr)

    train_gene_set, val_gene_set, test_gene_set = make_node_splits(
        pert_labels_arr=pert_labels_arr,
        test_frac=0.2,
        val_frac=0.15,
        random_state=args.seed,
    )

    train_row_mask = np.isin(pert_labels_arr, list(train_gene_set))

    svd, H_all = build_svd_targets_train_only(
        X_pert=X_pert,
        train_row_mask=train_row_mask,
        n_components=args.svd_dim,
        seed=args.seed,
    )

    node_list = sorted(list(pert_set))
    node2i = {g: i for i, g in enumerate(node_list)}
    K = H_all.shape[1]

    pert2row = {g: i for i, g in enumerate(pert_labels_arr)}

    Y = np.zeros((len(node_list), K), dtype=np.float32)
    X_node = np.zeros((len(node_list), X_pert.shape[1]), dtype=np.float32)
    for g, i in node2i.items():
        if g in pert2row:
            Y[i] = H_all[pert2row[g]]
            X_node[i] = X_pert[pert2row[g]]

    train_nodes = np.array([g in train_gene_set for g in node_list], dtype=bool)
    val_nodes = np.array([g in val_gene_set for g in node_list], dtype=bool)
    test_nodes = np.array([g in test_gene_set for g in node_list], dtype=bool)

    test_gene_list = [g for g, is_t in zip(node_list, test_nodes) if is_t]
    test_rows = [pert2row[g] for g in test_gene_list]
    X_true_test = np.asarray(X_pert[test_rows])
    Vt = svd.components_

    val_gene_list = [g for g, is_v in zip(node_list, val_nodes) if is_v]
    val_rows = [pert2row[g] for g in val_gene_list]
    X_true_val = np.asarray(X_pert[val_rows])

    # ------------------------------------------------
    # graph / features
    # ------------------------------------------------
    string_edges = build_string_edges(edges_gene, pert_set, min_score=args.string_min_score)
    string_edges.to_csv(os.path.join(args.outdir, "tables", "string_edges_used.csv"), index=False)

    bio_feat_df, bio_raw_df = build_biological_features(
        adata=adata,
        node_list=node_list,
        string_edges=string_edges,
        ctrl_label=args.ctrl_label,
        obs_gene_col=args.obs_gene_col,
        var_gene_col=args.var_gene_col,
        max_ctrl_cells=8000,
    )

    go_feat_df, go_count_map = build_go_embedding(
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

    X_graph = build_node2vec_features(
        edge_df=string_edges,
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

    X = np.concatenate([
        X_graph.astype(np.float32),
        bio_feat_df.loc[node_list].values.astype(np.float32),
        go_feat_df.loc[node_list].values.astype(np.float32),
    ], axis=1)

    edge_index, edge_weight = make_edge_index_and_weight(string_edges, node2i)

    G_string = build_string_graph(string_edges, node_list)
    train_neighbor_count, nearest_train_distance = compute_train_graph_context_features(
        G_string, node_list, train_gene_set
    )

    # ------------------------------------------------
    # train best model
    # ------------------------------------------------
    x_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    ei_t = edge_index.to(device)
    ew_t = edge_weight.to(device)
    tr = torch.tensor(train_nodes, dtype=torch.bool, device=device)

    model = WeightedGCN_MLP(
        in_dim=X.shape[1],
        hidden_dim=args.hidden_dim,
        out_dim=K,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_cos = -1e9
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    bad = 0

    def eval_cos(mask_np, X_true_subset):
        model.eval()
        with torch.no_grad():
            pred_all_local = model(x_t, ei_t, ew_t).cpu().numpy()
        H_pred_local = pred_all_local[mask_np]
        X_pred_local = H_pred_local @ Vt
        vals = [row_cosine(X_true_subset[i], X_pred_local[i]) for i in range(len(X_true_subset))]
        return float(np.mean(vals))

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(x_t, ei_t, ew_t)
        loss = F.mse_loss(pred[tr], y_t[tr])
        loss.backward()
        opt.step()

        if epoch % 20 == 0 or epoch == 1:
            val_cos = eval_cos(val_nodes, X_true_val)
            print(f"Epoch {epoch:03d} | train_mse={loss.item():.6f} | VAL Cos_Delta={val_cos:.4f}")

            if val_cos > best_val_cos + 1e-4:
                best_val_cos = val_cos
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    print("Early stopping.")
                    break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_all = model(x_t, ei_t, ew_t).cpu().numpy()

    H_pred_test = pred_all[test_nodes]
    X_pred_test = H_pred_test @ Vt

    # ------------------------------------------------
    # save arrays
    # ------------------------------------------------
    np.save(os.path.join(args.outdir, "arrays", "true_delta_test.npy"), X_true_test)
    np.save(os.path.join(args.outdir, "arrays", "pred_delta_test.npy"), X_pred_test)
    np.save(os.path.join(args.outdir, "arrays", "pred_latent_test.npy"), H_pred_test)
    np.save(os.path.join(args.outdir, "arrays", "all_node_embeddings.npy"), pred_all)

    pd.Series(gene_names).to_csv(os.path.join(args.outdir, "arrays", "gene_names.txt"), index=False, header=False)
    pd.Series(test_gene_list).to_csv(os.path.join(args.outdir, "arrays", "test_genes.txt"), index=False, header=False)

    # ------------------------------------------------
    # per-gene metrics + context
    # ------------------------------------------------
    rows = []
    for i, pert_gene in enumerate(test_gene_list):
        tv = X_true_test[i]
        pv = X_pred_test[i]
        overlap_metrics = signed_overlap_metrics(tv, pv, k=args.topk_eval)

        rows.append({
            "perturbed_gene": pert_gene,
            "cosine": row_cosine(tv, pv),
            "pearson": row_pearson(tv, pv),
            "spearman": safe_spearman(tv, pv),
            "mse": float(np.mean((tv - pv) ** 2)),
            "mae": mean_abs_error(tv, pv),
            "true_norm": signal_norm(tv),
            "pred_norm": signal_norm(pv),

            "topk_up_overlap": overlap_metrics["topk_up_overlap"],
            "topk_dn_overlap": overlap_metrics["topk_dn_overlap"],
            "topk_up_precision": overlap_metrics["topk_up_precision"],
            "topk_dn_precision": overlap_metrics["topk_dn_precision"],
            "topk_up_recall": overlap_metrics["topk_up_recall"],
            "topk_dn_recall": overlap_metrics["topk_dn_recall"],
            "topk_up_f1": overlap_metrics["topk_up_f1"],
            "topk_dn_f1": overlap_metrics["topk_dn_f1"],

            "string_degree": float(bio_raw_df.loc[pert_gene, "string_degree"]) if pert_gene in bio_raw_df.index else np.nan,
            "string_weighted_degree": float(bio_raw_df.loc[pert_gene, "string_weighted_degree"]) if pert_gene in bio_raw_df.index else np.nan,
            "string_pagerank": float(bio_raw_df.loc[pert_gene, "string_pagerank"]) if pert_gene in bio_raw_df.index else np.nan,
            "string_clustering": float(bio_raw_df.loc[pert_gene, "string_clustering"]) if pert_gene in bio_raw_df.index else np.nan,
            "ctrl_mean": float(bio_raw_df.loc[pert_gene, "ctrl_mean"]) if pert_gene in bio_raw_df.index else np.nan,
            "ctrl_var": float(bio_raw_df.loc[pert_gene, "ctrl_var"]) if pert_gene in bio_raw_df.index else np.nan,
            "ctrl_detect_frac": float(bio_raw_df.loc[pert_gene, "ctrl_detect_frac"]) if pert_gene in bio_raw_df.index else np.nan,

            "n_go_terms": int(go_count_map.get(pert_gene, 0)),
            "train_neighbor_count": int(train_neighbor_count.get(pert_gene, 0)),
            "nearest_train_distance": float(nearest_train_distance.get(pert_gene, np.nan)),
        })

    per_gene_df = pd.DataFrame(rows).sort_values("cosine", ascending=False).reset_index(drop=True)
    per_gene_df.to_csv(os.path.join(args.outdir, "per_gene_test_metrics.csv"), index=False)

    # ------------------------------------------------
    # choose qualitative cases
    # ------------------------------------------------
    best_gene, median_gene, worst_gene = select_case_indices(
        per_gene_df, signal_quantile=args.signal_quantile_for_cases
    )
    chosen = [("best", best_gene), ("median", median_gene), ("worst", worst_gene)]

    # ------------------------------------------------
    # global plots
    # ------------------------------------------------
    plot_histogram(
        per_gene_df["cosine"].values,
        xlabel="Per-gene cosine similarity",
        title="Distribution of PerturbGraph test-gene accuracy",
        outpath=os.path.join(args.outdir, "plots", "hist_per_gene_cosine.png"),
        bins=24,
        color=ACCENT_PURPLE,
    )

    plot_metric_by_degree_bins(
        per_gene_df,
        outpath=os.path.join(args.outdir, "plots", "box_cosine_vs_string_degree_bins.png"),
    )

    plot_metric_by_neighbor_bins(
        per_gene_df,
        outpath=os.path.join(args.outdir, "plots", "box_cosine_vs_train_neighbor_bins.png"),
    )

    plot_metric_by_distance_bins(
        per_gene_df,
        outpath=os.path.join(args.outdir, "plots", "box_cosine_vs_graph_distance_bins.png"),
    )

    # ------------------------------------------------
    # per-case qualitative plots and tables
    # ------------------------------------------------
    pathways = None
    if args.gmt_file is not None:
        pathways = load_gmt(args.gmt_file, gene_universe=set(gene_names), min_size=5, max_size=500)

    for tag, pert_gene in chosen:
        i = test_gene_list.index(pert_gene)
        true_vec = X_true_test[i]
        pred_vec = X_pred_test[i]

        row = per_gene_df[per_gene_df["perturbed_gene"] == pert_gene].iloc[0].to_dict()

        plot_true_vs_pred_scatter(
            true_vec=true_vec,
            pred_vec=pred_vec,
            pert_gene=pert_gene,
            outpath=os.path.join(args.outdir, "plots", f"scatter_{tag}_{pert_gene}.png"),
            annotate_stats=row,
        )

        plot_case_study_lollipop(
            true_vec=true_vec,
            pred_vec=pred_vec,
            gene_names=gene_names,
            pert_gene=pert_gene,
            outpath=os.path.join(args.outdir, "plots", f"lollipop_{tag}_{pert_gene}.png"),
            top_n_each=args.case_top_n_each,
        )

        plot_topk_overlap_counts(
            true_vec=true_vec,
            pred_vec=pred_vec,
            pert_gene=pert_gene,
            outpath=os.path.join(args.outdir, "plots", f"topk_counts_{tag}_{pert_gene}.png"),
            k=args.topk_eval,
        )

        plot_topk_prf(
            true_vec=true_vec,
            pred_vec=pred_vec,
            pert_gene=pert_gene,
            outpath=os.path.join(args.outdir, "plots", f"topk_scores_{tag}_{pert_gene}.png"),
            k=args.topk_eval,
        )

        plot_network_neighborhood_compare(
            target_gene=pert_gene,
            true_vec=true_vec,
            pred_vec=pred_vec,
            gene_names=gene_names,
            edge_df=string_edges,
            outpath=os.path.join(args.outdir, "plots", f"network_compare_{tag}_{pert_gene}.png"),
            top_neighbors=args.network_top_neighbors,
        )

        # gene-level case table
        union_idx = get_case_union_indices(true_vec, pred_vec, top_n_each=args.case_top_n_each)
        case_tbl = pd.DataFrame({
            "gene": [gene_names[j] for j in union_idx],
            "true_delta": true_vec[union_idx],
            "pred_delta": pred_vec[union_idx],
        })
        case_tbl["abs_true"] = case_tbl["true_delta"].abs()
        case_tbl["abs_pred"] = case_tbl["pred_delta"].abs()
        case_tbl["same_direction"] = np.sign(case_tbl["true_delta"]) == np.sign(case_tbl["pred_delta"])

        true_up, true_dn = topk_sets(true_vec, k=args.topk_eval)
        pred_up, pred_dn = topk_sets(pred_vec, k=args.topk_eval)

        case_tbl["in_true_topk_up"] = [j in true_up for j in union_idx]
        case_tbl["in_true_topk_dn"] = [j in true_dn for j in union_idx]
        case_tbl["in_pred_topk_up"] = [j in pred_up for j in union_idx]
        case_tbl["in_pred_topk_dn"] = [j in pred_dn for j in union_idx]
        case_tbl = case_tbl.sort_values(["abs_true", "abs_pred"], ascending=False)
        case_tbl.to_csv(os.path.join(args.outdir, "tables", f"case_gene_table_{tag}_{pert_gene}.csv"), index=False)

        # pathway validation
        if pathways is not None:
            true_gene_sel = [gene_names[j] for j in sorted(set(list(true_up) + list(true_dn)))]
            pred_gene_sel = [gene_names[j] for j in sorted(set(list(pred_up) + list(pred_dn)))]

            true_top, pred_top, shared, true_only, pred_only = pathway_overlap_summary(
                true_gene_sel, pred_gene_sel, pathways, top_n=10
            )

            true_path_df = pd.DataFrame(true_top, columns=["pathway", "overlap", "jaccard", "pathway_size"])
            pred_path_df = pd.DataFrame(pred_top, columns=["pathway", "overlap", "jaccard", "pathway_size"])

            true_path_df.to_csv(os.path.join(args.outdir, "tables", f"pathways_true_{tag}_{pert_gene}.csv"), index=False)
            pred_path_df.to_csv(os.path.join(args.outdir, "tables", f"pathways_pred_{tag}_{pert_gene}.csv"), index=False)

            plot_pathway_overlap_bar(
                shared=shared,
                true_only=true_only,
                pred_only=pred_only,
                pert_gene=pert_gene,
                outpath=os.path.join(args.outdir, "plots", f"pathway_overlap_{tag}_{pert_gene}.png"),
            )

            if len(true_path_df) > 0 or len(pred_path_df) > 0:
                plot_pathway_comparison_heatmap(
                    true_path_df,
                    pred_path_df,
                    pert_gene=pert_gene,
                    outpath=os.path.join(args.outdir, "plots", f"pathway_heatmap_{tag}_{pert_gene}.png"),
                    top_n=10,
                )

    # ------------------------------------------------
    # summary json
    # ------------------------------------------------
    summary = {
        "method_name": "PerturbGraph",
        "seed": args.seed,
        "num_test_genes": len(test_gene_list),
        "best_gene": best_gene,
        "best_cosine": float(per_gene_df.set_index("perturbed_gene").loc[best_gene, "cosine"]),
        "median_gene": median_gene,
        "median_cosine": float(per_gene_df.set_index("perturbed_gene").loc[median_gene, "cosine"]),
        "worst_gene": worst_gene,
        "worst_cosine": float(per_gene_df.set_index("perturbed_gene").loc[worst_gene, "cosine"]),
        "mean_cosine": float(per_gene_df["cosine"].mean()),
        "mean_spearman": float(per_gene_df["spearman"].mean()),
        "mean_pearson": float(per_gene_df["pearson"].mean()),
        "mean_topk_up_overlap": float(per_gene_df["topk_up_overlap"].mean()),
        "mean_topk_dn_overlap": float(per_gene_df["topk_dn_overlap"].mean()),
        "mean_train_neighbor_count": float(per_gene_df["train_neighbor_count"].mean()),
    }

    with open(os.path.join(args.outdir, "qualitative_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print("Outputs written to:", args.outdir)
    print("Important files:")
    print("  per_gene_test_metrics.csv")
    print("  qualitative_summary.json")
    print("  plots/box_cosine_vs_*.png")
    print("  plots/scatter_*.png")
    print("  plots/lollipop_*.png")
    print("  plots/topk_counts_*.png")
    print("  plots/topk_scores_*.png")
    print("  plots/network_compare_*.png")
    print("  plots/pathway_overlap_*.png")
    print("  plots/pathway_heatmap_*.png")
    print("  tables/case_gene_table_*.csv")
    print("  tables/pathways_*.csv")

if __name__ == "__main__":
    main()