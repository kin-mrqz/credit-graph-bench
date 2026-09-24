# Gate A: graph signal for credit default, commercial summary

_Generated 2026-09-25 06:49 from 189/190 finished units._

## What was tested

Four public credit datasets (Taiwan card default, Home Credit, Lending Club, DGraph-Fin). Tier 0 XGBoost is the reference. Challengers: hand-built graph features (Tier 1), GNNs (Tier 2), GNN-embedding + XGBoost hybrid (Tier 3), and a tabular student distilled from a graph teacher (Tier 4, LUPI). Every number is on a frozen test split, five seeds, paired bootstrap 95% CI against XGBoost.

## Headline numbers (test AUC, mean over seeds)

**dgraphfin** — XGBoost 0.7246; best challenger 1 graph features / native / xgb_graphfeat at 0.8225.
**home_credit** — XGBoost 0.7806; best challenger 4 lupi student / random_control / student_self_xgb at 0.7808.
**lending_club** — XGBoost 0.7178; best challenger 4 lupi student / shared_attribute / student_self_xgb at 0.7180.
**taiwan** — XGBoost 0.7906; best challenger 4 lupi student / knn / student_knn_k10 at 0.7916.

## Lift over XGBoost (paired 95% CI; 'sig' = seeds where CI excludes zero)

- dgraphfin: 04 native/xgb_graphfeat: ΔAUC +0.0979 [+0.0906, +0.1047], sig 5/5
- dgraphfin: 06 native/hybrid_sage: ΔAUC +0.0606 [+0.0537, +0.0675], sig 5/5
- dgraphfin: 05 native/sage: ΔAUC +0.0559 [+0.0490, +0.0628], sig 5/5
- home_credit: 04 knn/xgb_graphfeat: ΔAUC -0.0019 [-0.0036, -0.0002], sig 3/5
- home_credit: 04 random_control/xgb_graphfeat: ΔAUC -0.0025 [-0.0043, -0.0006], sig 2/5
- home_credit: 06 relational/hybrid_rgcn: ΔAUC -0.0149 [-0.0186, -0.0110], sig 5/5
- lending_club: 04 shared_attribute/xgb_graphfeat: ΔAUC -0.0001 [-0.0008, +0.0006], sig 0/5
- lending_club: 06 shared_attribute/hybrid_gat: ΔAUC -0.0008 [-0.0013, -0.0002], sig 5/5
- lending_club: 06 shared_attribute/hybrid_sage: ΔAUC -0.0010 [-0.0018, -0.0002], sig 4/5
- taiwan: 04 knn/xgb_graphfeat: ΔAUC -0.0020 [-0.0065, +0.0026], sig 0/5
- taiwan: 04 random_control/xgb_graphfeat: ΔAUC -0.0032 [-0.0080, +0.0014], sig 0/5
- taiwan: 06 knn/hybrid_gat: ΔAUC -0.0050 [-0.0110, +0.0009], sig 2/5

## Business translation (stage 08, mean over seeds)

- dgraphfin: 04:native:xgb_graphfeat approves +13.27 pp more applicants at XGBoost's bad rate (0.911% at 80% approval); capture@10% 0.436
- dgraphfin: 06:native:hybrid_sage approves +11.07 pp more applicants at XGBoost's bad rate (0.911% at 80% approval); capture@10% 0.373
- home_credit: 07:random_control:student_random_control_k20 approves +0.14 pp more applicants at XGBoost's bad rate (4.461% at 80% approval); capture@10% 0.363
- home_credit: 07:knn:student_self_xgb approves +0.12 pp more applicants at XGBoost's bad rate (4.461% at 80% approval); capture@10% 0.367
- lending_club: 04:shared_attribute:xgb_graphfeat approves +0.08 pp more applicants at XGBoost's bad rate (16.262% at 80% approval); capture@10% 0.221
- lending_club: 07:random_control:student_random_control_k5 approves +0.05 pp more applicants at XGBoost's bad rate (16.262% at 80% approval); capture@10% 0.221
- taiwan: 07:knn:student_self_xgb approves +0.08 pp more applicants at XGBoost's bad rate (13.300% at 80% approval); capture@10% 0.328
- taiwan: 07:random_control:student_self_xgb approves +0.08 pp more applicants at XGBoost's bad rate (13.300% at 80% approval); capture@10% 0.328

## Learning using privileged information

- dgraphfin: structured teacher native: student ΔAUC +0.0010 (λ=0.25, T=2.0, 5 seeds)
- dgraphfin: self_distill teacher self_xgb: student ΔAUC +0.0005 (λ=0.75, T=4.0, 5 seeds)
- home_credit: self_distill teacher self_xgb: student ΔAUC +0.0002 (λ=0.5, T=2.0, 5 seeds)
- home_credit: structured teacher relational: student ΔAUC -0.0006 (λ=0.25, T=2.0, 5 seeds)
- home_credit: structured teacher knn_k5: student ΔAUC -0.0007 (λ=0.25, T=2.0, 5 seeds)
- home_credit: random_control teacher random_control_k20: student ΔAUC -0.0008 (λ=0.25, T=2.0, 5 seeds)
- home_credit: structured teacher knn_k20: student ΔAUC -0.0009 (λ=0.25, T=2.0, 5 seeds)
- home_credit: structured teacher knn_k10: student ΔAUC -0.0011 (λ=0.25, T=2.0, 5 seeds)
- home_credit: random_control teacher random_control_k10: student ΔAUC -0.0016 (λ=0.25, T=4.0, 5 seeds)
- home_credit: random_control teacher random_control_k5: student ΔAUC -0.0018 (λ=0.25, T=2.0, 5 seeds)
- lending_club: self_distill teacher self_xgb: student ΔAUC +0.0002 (λ=0.75, T=1.0, 5 seeds)
- lending_club: structured teacher knn_k5: student ΔAUC +0.0001 (λ=0.25, T=1.0, 5 seeds)
- lending_club: random_control teacher random_control_k5: student ΔAUC +0.0001 (λ=0.25, T=1.0, 5 seeds)
- lending_club: structured teacher knn_k20: student ΔAUC +0.0001 (λ=0.25, T=1.0, 5 seeds)
- lending_club: structured teacher knn_k10: student ΔAUC -0.0001 (λ=0.25, T=1.0, 5 seeds)
- lending_club: random_control teacher random_control_k20: student ΔAUC -0.0002 (λ=0.25, T=1.0, 5 seeds)
- lending_club: structured teacher shared_attribute: student ΔAUC -0.0003 (λ=0.25, T=1.0, 5 seeds)
- lending_club: random_control teacher random_control_k10: student ΔAUC -0.0004 (λ=0.25, T=1.0, 5 seeds)
- taiwan: structured teacher knn_k10: student ΔAUC +0.0010 (λ=0.5, T=2.0, 5 seeds)
- taiwan: structured teacher knn_k5: student ΔAUC +0.0010 (λ=0.5, T=1.0, 5 seeds)
- taiwan: structured teacher knn_k20: student ΔAUC +0.0008 (λ=0.25, T=2.0, 5 seeds)
- taiwan: random_control teacher random_control_k5: student ΔAUC +0.0007 (λ=0.25, T=2.0, 5 seeds)
- taiwan: random_control teacher random_control_k20: student ΔAUC +0.0001 (λ=0.25, T=2.0, 5 seeds)
- taiwan: random_control teacher random_control_k10: student ΔAUC -0.0000 (λ=0.25, T=1.0, 5 seeds)
- taiwan: self_distill teacher self_xgb: student ΔAUC -0.0000 (λ=0.25, T=1.0, 5 seeds)

## Pre-registered criteria

- **FAIL** — Tier 1 graph features = lift. datasets with lift: ['dgraphfin (native/xgb_graphfeat)']
- **FAIL** — Tier 2 pure GNN = lift (not expected). datasets with lift: ['dgraphfin (native/sage)']
- **FAIL** — Tier 3 hybrid = lift (primary candidate). datasets with lift: ['dgraphfin (native/hybrid_sage)']
- **FAIL** — LUPI student received graph knowledge (beats both controls, CI > 0). no dataset where structured teacher beats both controls
- **PASS** — DGraph-Fin reproduces published AUC within 0.02. [{'model': 'mlp', 'auc': 0.7242, 'paper_auc': 0.723}, {'model': 'sage', 'auc': 0.7805, 'paper_auc': 0.778}]

## What this does and does not show

- All labels are public proxies for default; no number here is a calibrated PD for a real portfolio.
- On three datasets the graph is constructed by us and is framed as a structural regulariser, not as evidence of real borrower relationships. DGraph-Fin is the only dataset with real edges.
- The hybrid and the distilled student are the only shapes that could go on a serving path; pure GNNs are evidence, not candidates.
- Figures: `report/figures/`. Tables: `report/*.csv`. Full detail: the executed notebooks under `runs/`.