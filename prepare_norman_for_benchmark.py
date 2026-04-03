#!/usr/bin/env python3

import scanpy as sc
import pandas as pd

INFILE = "/home/sajibacharjeedip/shared_disk/Perturb/data_norman/norman_2019_adata.h5ad"
OUTFILE = "/home/sajibacharjeedip/shared_disk/Perturb/data_norman/norman_single_gene_perturbseq.h5ad"

MIN_CELLS_PER_PERT = 30


def convert_label(x: str):
    x = str(x).strip()

    # pure control
    if x == "ctrl":
        return "control"

    parts = [p.strip() for p in x.split("+")]

    # single gene perturbation, e.g. "KLF1"
    if len(parts) == 1 and parts[0] != "ctrl":
        return parts[0]

    # gene + ctrl or ctrl + gene
    if len(parts) == 2 and "ctrl" in parts:
        non_ctrl = [p for p in parts if p != "ctrl"]
        if len(non_ctrl) == 1:
            return non_ctrl[0]

    # everything else is combinatorial perturbation
    return None


def main():
    print(f"Loading dataset: {INFILE}")
    adata = sc.read_h5ad(INFILE)
    print(adata)

    # Use gene symbols as var names for downstream STRING / GO matching
    if "gene_name" in adata.var.columns:
        adata.var_names = adata.var["gene_name"].astype(str).values

    # Check required column
    if "guide_merged" not in adata.obs.columns:
        raise ValueError("Expected obs column 'guide_merged' not found.")

    # Parse perturbation labels
    adata.obs["gene"] = adata.obs["guide_merged"].astype(str).map(convert_label)

    # Keep only single perturbations + control
    adata = adata[adata.obs["gene"].notna()].copy()
    print("\nAfter removing combinatorial perturbations:")
    print("  cells:", adata.n_obs)
    print("  genes:", adata.n_vars)
    print("  perturbations:", adata.obs['gene'].nunique())
    print(adata.obs["gene"].value_counts().head(20))

    # Keep labels with enough cells
    vc = adata.obs["gene"].value_counts()
    keep = vc[vc >= MIN_CELLS_PER_PERT].index
    adata = adata[adata.obs["gene"].isin(keep)].copy()

    print(f"\nAfter filtering perturbations with < {MIN_CELLS_PER_PERT} cells:")
    print("  cells:", adata.n_obs)
    print("  genes:", adata.n_vars)
    print("  perturbations:", adata.obs['gene'].nunique())
    print(adata.obs["gene"].value_counts().head(30))

    # Save
    adata.obs["gene"] = adata.obs["gene"].astype(str)
    adata.write(OUTFILE)
    print(f"\nSaved processed dataset to: {OUTFILE}")


if __name__ == "__main__":
    main()