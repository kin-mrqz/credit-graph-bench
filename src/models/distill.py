"""Learning using privileged information via generalised distillation (Lopez-Paz et al. 2016).

Teacher: a model that sees the graph (privileged information) at training time.
Student: XGBoost on tabular features only. Its training target is

    target = (1 - lambda) * y  +  lambda * sigmoid(teacher_logit / T)

lambda = 0 is the plain XGBoost baseline. T > 1 softens the teacher's probabilities so the
student learns the teacher's ranking, not just its hard calls.

Controls (both must be run for a claim of "graph knowledge transferred"):
    self-distillation:  teacher = XGBoost on the same tabular features, no graph
    random control:     teacher = GNN trained on a degree-preserving rewired graph
"""

from __future__ import annotations

import numpy as np

from .xgb import fit_xgb, predict_xgb


def soft_labels(teacher_logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(teacher_logits, dtype=np.float64) / float(T)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def logits_from_probs(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p)).astype(np.float32)


def mixed_targets(y: np.ndarray, soft: np.ndarray, lam: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    soft = np.asarray(soft, dtype=np.float32)
    return ((1.0 - lam) * y + lam * soft).astype(np.float32)


def train_student(X_tr, y_tr, teacher_logits_tr, lam: float, T: float, X_val, y_val, xgb_params: dict, seed: int = 0):
    """XGBoost student on mixed targets. Validation uses hard labels (selection is on real outcomes)."""
    target = mixed_targets(y_tr, soft_labels(teacher_logits_tr, T), lam) if lam > 0 else np.asarray(y_tr, np.float32)
    booster = fit_xgb(X_tr, target, X_val, y_val, xgb_params, seed=seed)
    return booster


def student_predict(booster, X) -> np.ndarray:
    return predict_xgb(booster, X)


def teacher_agreement(teacher_logits, y) -> dict:
    """How good is the teacher on the rows it labels for the student? (Sanity check, not a result.)"""
    from ..eval.metrics import auc

    p = soft_labels(teacher_logits, 1.0)
    return {"teacher_auc_on_train": auc(y, p), "teacher_mean_p": float(p.mean()), "n": int(len(p))}
