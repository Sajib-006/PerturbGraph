#!/usr/bin/env bash
set -uo pipefail

SCRIPT="stable_shift_bench_extended.py"

H5AD="/home/sajibacharjeedip/shared_disk/Perturb/data_norman/norman_single_gene_perturbseq.h5ad"
STRING_LINKS="/home/sajibacharjeedip/shared_disk/Perturb/9606.protein.links.v11.5.txt"
STRING_INFO="/home/sajibacharjeedip/shared_disk/Perturb/9606.protein.info.v11.5.txt"
GO_FILE="/home/sajibacharjeedip/shared_disk/Perturb/go_annotations.tsv"

OUTDIR="eccb/bench_norman_all_methods"
LOGDIR="${OUTDIR}/logs"

mkdir -p "${OUTDIR}" "${LOGDIR}"

COMMON_ARGS=(
  --h5ad "${H5AD}"
  --string_links "${STRING_LINKS}"
  --string_info "${STRING_INFO}"
  --go_file "${GO_FILE}"
  --outdir "${OUTDIR}"
  --ctrl_label control
  --graph_type STRING_only
  --use_bio_features
  --use_go_features
  --go_gene_col gene
  --go_term_col go_id
  --go_sep $'\t'
  --go_min_genes_per_term 5
  --go_max_genes_per_term 500
  --go_svd_dim 64
  --node2vec_dim 256
  --n2v_walk_length 40
  --n2v_num_walks 150
  --n2v_window 10
  --n2v_p 0.5
  --n2v_q 2.0
  --seed 42
)

run_method () {
  local METHOD="$1"
  shift
  echo "Running ${METHOD} ..."
  python "${SCRIPT}" \
    "${COMMON_ARGS[@]}" \
    --model_type "${METHOD}" \
    "$@" \
    > "${LOGDIR}/${METHOD}.log" 2>&1 || echo "FAILED: ${METHOD}" | tee -a "${LOGDIR}/failures.log"
}

# 9 paper models
run_method lasso      --lasso_alpha 0.001
run_method elasticnet --elastic_alpha 0.01 --elastic_l1_ratio 0.5
run_method ridge      --ridge_alpha 1.0
run_method knn        --knn_k 10
run_method rf         --rf_estimators 300 --rf_max_depth 20
run_method mlp        --hidden_dim 256 --dropout 0.15 --lr 1e-3 --weight_decay 1e-4 --epochs 300 --patience 12
run_method sage       --hidden_dim 256 --dropout 0.15 --lr 1e-3 --weight_decay 1e-4 --epochs 300 --patience 12
run_method gat        --hidden_dim 256 --dropout 0.15 --gat_heads 4 --lr 1e-3 --weight_decay 1e-4 --epochs 300 --patience 12
run_method gcn        --hidden_dim 256 --dropout 0.15 --lr 1e-3 --weight_decay 1e-4 --epochs 300 --patience 12

echo "Done."
echo "Summary CSV: ${OUTDIR}/benchmark_summary.csv"