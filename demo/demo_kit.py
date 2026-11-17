"""Plotting and widget helpers for graph_demo.ipynb.

The pipeline itself (graph builders, random control, leakage check, graph features, XGBoost,
bootstrap, hybrid) is imported from src/. This file only draws things.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

# joblib tries to count physical cores via a subprocess that fails on some Windows setups
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))
warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support.*")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
warnings.filterwarnings("ignore", category=FutureWarning)

# On my Windows machine, importing optree inside a Jupyter kernel crashes the kernel. torch.optim
# imports it; blocking it makes torch fall back to its pure-Python pytree. Has to happen before
# torch is imported, so import this module first.
if "torch" not in sys.modules:
    sys.modules.setdefault("optree", None)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ipywidgets as W  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402
import xgboost as xgb  # noqa: E402
from IPython.display import HTML, display  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from src import config, paths  # noqa: E402
from src.data import registry, splits  # noqa: E402
from src.data.base import Bundle  # noqa: E402
from src.device import xgb_device  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.eval.bootstrap import paired_bootstrap  # noqa: E402
from src.models import xgb as XGB  # noqa: E402

GREEN, GOLD, BLUE, RED, GREY = "#76b900", "#d29a2a", "#4691e6", "#e05a5a", "#6b737b"
PURPLE, LIGHT, INK, MUTED = "#b4489a", "#c9ced3", "#1d2125", "#5b636b"
FONT = "Segoe UI, Helvetica, Arial, sans-serif"
KIND_COLOR = {"native": GREEN, "knn": GOLD, "shared_attribute": GOLD, "random_control": GREY, "relational": BLUE}
KIND_LABEL = {"native": "real edges", "knn": "kNN", "shared_attribute": "shared zip/employer",
              "random_control": "random control", "relational": "relational"}
DS_LABEL = {"dgraphfin": "DGraph-Fin", "home_credit": "Home Credit", "lending_club": "Lending Club", "taiwan": "Taiwan"}
REPORT = ROOT / "report"
CACHE = Path(__file__).resolve().parent / ".cache"
WIDTH = 960   # FigureWidgets don't autosize


def style(font_size: int = 18) -> None:
    t = go.layout.Template(pio.templates["plotly_white"])
    t.layout.update(font=dict(family=FONT, size=font_size, color=INK), width=WIDTH,
                    title=dict(font=dict(size=font_size + 4), x=0, xanchor="left", y=0.985, yref="container", yanchor="top"),
                    hoverlabel=dict(font=dict(size=font_size - 1, family=FONT)),
                    margin=dict(l=20, r=30, t=110, b=50), paper_bgcolor="white", plot_bgcolor="white",
                    legend=dict(orientation="h", y=1.01, yanchor="bottom", x=0, font=dict(size=font_size - 2)))
    t.layout.xaxis.update(gridcolor="#eceef0", zerolinecolor="#d5d9dd", linecolor="#d5d9dd", ticks="")
    t.layout.yaxis.update(gridcolor="#eceef0", zerolinecolor="#d5d9dd", linecolor="#d5d9dd", ticks="")
    pio.templates["demo"] = t
    pio.templates.default = "demo"
    # larger text for screen sharing
    display(HTML(f"""<style>
      .jp-Notebook {{ --jp-content-font-size1: {font_size}px; --jp-code-font-size: {font_size - 1}px; }}
      .widget-label, .widget-toggle-button, .widget-dropdown select, .widget-button {{ font-size: {font_size - 2}px !important; }}
    </style>"""))


@contextmanager
def step(what: str):
    t = time.time()
    print(f"{what}...", end=" ", flush=True)
    yield
    print(f"done ({time.time() - t:.1f}s)", flush=True)


# ---------------------------------------------------------------- data
def load(dataset: str = "taiwan") -> SimpleNamespace:
    """Same loading block as notebooks 03-06. Builds the processed data on first run."""
    cfg = config.load_config(dataset)
    pdir = paths.processed_dir(dataset)
    if not (pdir / "meta.json").exists():
        with step(f"building {dataset} data (first run only)"):
            registry.get(dataset).build(cfg).save(pdir)
    bundle = Bundle.load(pdir)
    split = splits.load_split(dataset)
    X, y = bundle.X, bundle.y.to_numpy()
    tr, va, te = split["train"], split["val"], split["test"]
    m = splits.masks(split, bundle.n)
    D = SimpleNamespace(dataset=dataset, cfg=cfg, X=X, y=y, n=bundle.n, tr=tr, va=va, te=te,
                        allowed=np.sort(np.concatenate([tr, va])), train_mask=m["train"],
                        test_mask=m["test"], Xnp=X.to_numpy(np.float32))
    D.part = np.full(D.n, "val", dtype=object)
    D.part[tr], D.part[te] = "train", "test"
    D.map2d = feature_map(D)
    print(f"{D.n:,} rows, {X.shape[1]} features, {np.mean(y):.1%} default rate, "
          f"train/val/test {len(tr):,}/{len(va):,}/{len(te):,}")
    return D


def feature_map(D) -> np.ndarray:
    """t-SNE of the 16-d PCA space the kNN graph is built in. Only used for drawing.
    Takes ~2 min the first time, then loads from demo/.cache/."""
    path = CACHE / f"map_{D.dataset}.npy"
    if path.exists():
        return np.load(path)
    from sklearn.manifold import TSNE

    from src.graphs import knn as KNN

    with step("computing 2-D map for plots (t-SNE, cached after this)"):
        Z = KNN.KnnIndex(16).fit(D.Xnp, D.allowed, 1)._embed(D.Xnp, fit=False)
        P = TSNE(2, perplexity=40, init="pca", random_state=0, n_jobs=-1).fit_transform(Z)
    CACHE.mkdir(exist_ok=True)
    np.save(path, P)
    return P


# ---------------------------------------------------------------- 03: graphs
def _source_html(fn) -> str:
    import inspect

    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import PythonLexer

    name = f"{fn.__module__.replace('.', '/')}.py: {fn.__qualname__}"
    code = highlight(inspect.getsource(fn), PythonLexer(), HtmlFormatter(noclasses=True, style="friendly", nobackground=True))
    return (f'<div style="font:15px {FONT};color:{MUTED};margin:6px 0 2px">{name}</div>'
            f'<div style="font-size:15px;line-height:1.45;background:#f7f8f9;border:1px solid #e2e5e8;'
            f'padding:4px 14px;overflow-x:auto">{code}</div>')


def graph_build_walkthrough(D, k: int = 3, seed: int = 1) -> W.Widget:
    """kNN graph construction on 36 rows, step by step, with the source for each step.
    Uses pca_dims=2 (16 in the real run) so the plot is the space the search runs in."""
    from src.graphs import EdgeList, assert_no_test_nodes
    from src.graphs import knn as KNN
    from src.graphs import random_control as RC

    rng = np.random.default_rng(seed)
    rows = np.r_[rng.choice(D.tr, 24, replace=False), rng.choice(D.va, 6, replace=False), rng.choice(D.te, 6, replace=False)]
    Xs, ys = D.Xnp[rows], D.y[rows]
    allowed, test = np.arange(30), np.arange(30, 36)
    is_tr, is_va, is_te = np.arange(36) < 24, (np.arange(36) >= 24) & (np.arange(36) < 30), np.arange(36) >= 30

    idx = KNN.KnnIndex(pca_dims=2, seed=0).fit(Xs, allowed, k)
    clean = idx.clean_edges([k])[k]
    query = idx.query_edges(Xs, test, [k])[k]
    rnd = RC.rewire(clean, seed=0)
    Z = idx._embed(Xs, fit=False)
    full = EdgeList.concat(clean, query).symmetrized()
    full_r = EdgeList.concat(rnd, RC.rewire(query, seed=1)).symmetrized()
    assert_no_test_nodes(clean, test, "walkthrough")
    mutual = 1 - clean.symmetrized().n_edges / 2 / clean.n_edges

    def arrows(e, color, opacity=0.9):
        return [dict(x=Z[d, 0], y=Z[d, 1], ax=Z[s, 0], ay=Z[s, 1], xref="x", yref="y", axref="x", ayref="y", text="",
                     showarrow=True, arrowhead=3, arrowsize=1.1, arrowwidth=2.2, arrowcolor=color, opacity=opacity,
                     standoff=9, startstandoff=7) for s, d in zip(e.src, e.dst)]

    def lines(e, color):
        keep = e.src < e.dst
        return [dict(x=Z[d, 0], y=Z[d, 1], ax=Z[s, 0], ay=Z[s, 1], xref="x", yref="y", axref="x", ayref="y", text="",
                     showarrow=True, arrowhead=0, arrowwidth=2.2, arrowcolor=color, standoff=7, startstandoff=7)
                for s, d in zip(e.src[keep], e.dst[keep])]

    deg = lambda e, nodes: np.bincount(e.src, minlength=36)[nodes].mean()
    STEPS = {
        "1. Rows": (KNN.KnnIndex._embed, [],
                    f"24 train, 6 validation, 6 test rows. Features are imputed, standardised and reduced with PCA "
                    f"(fit on train + val only). The axes are the two PCA components."),
        "2. kNN edges": (KNN.KnnIndex.clean_edges, arrows(clean, GOLD),
                         f"Each train/val row gets an edge to its {k} nearest train/val rows. Test rows are not in the index."),
        "3. Query edges": (KNN.KnnIndex.query_edges, arrows(clean, "#e3c58a", 0.5) + arrows(query, BLUE),
                           f"Each test row is matched to its {k} nearest train/val rows. Edges go one way only; "
                           f"test rows never connect to each other."),
        "4. Leak check": (assert_no_test_nodes, arrows(clean, GOLD),
                          "The training graph has 0 edges touching a test row, so assert_no_test_nodes passes. "
                          "Fitting the index on all rows would fail it."),
        "5. Random control": (RC.rewire, arrows(rnd, GREY),
                              "Same source for every edge, destinations shuffled. Out-degree is unchanged; "
                              "the edges no longer mean anything."),
        "6. Symmetrised": (EdgeList.symmetrized, lines(full, GOLD),
                           f"GNNs use the graph undirected. {mutual:.0%} of kNN pairs here are mutual and collapse into one edge, "
                           f"so rows end up with {deg(full, allowed):.1f} neighbours on average vs {deg(full_r, allowed):.1f} "
                           f"for the random control."),
    }

    fig = go.FigureWidget()
    pad = 0.6
    fig.update_layout(height=620, transition=dict(duration=400), title=f"kNN graph construction (36 rows, k={k})",
                      xaxis=dict(range=[Z[:, 0].min() - pad, Z[:, 0].max() + pad], showticklabels=False, title="PC 1", zeroline=False),
                      yaxis=dict(range=[Z[:, 1].min() - pad, Z[:, 1].max() + pad], showticklabels=False, title="PC 2", zeroline=False,
                                 scaleanchor="x", scaleratio=1))
    tr_bad, tr_good = is_tr & (ys == 1), is_tr & (ys == 0)
    for m, name, mk in ((tr_good, "train, repaid", dict(color=GREY)), (tr_bad, "train, defaulted", dict(color=RED)),
                        (is_va, "validation", dict(color="white", line=dict(width=2.5, color=GREY))),
                        (is_te, "test", dict(symbol="diamond", size=19, color=BLUE, line=dict(width=2, color="white")))):
        fig.add_scatter(x=Z[m, 0], y=Z[m, 1], mode="markers", name=name, hovertemplate="%{text}<extra></extra>",
                        text=[f"row {rows[i]} ({name})" for i in np.flatnonzero(m)],
                        marker={"size": 16, "line": dict(width=2, color="white"), **mk})
    stepper = W.ToggleButtons(options=list(STEPS), style={"button_width": "150px"})
    caption, code = W.HTML(), W.HTML()

    def draw(*_):
        fn, ann, text = STEPS[stepper.value]
        with fig.batch_update():
            fig.layout.annotations = ann
            fig.data[3].marker.line = dict(width=4, color=GREEN) if stepper.value.startswith("4") else dict(width=2, color="white")
        caption.value = f'<div style="font:19px {FONT};line-height:1.5;margin:6px 0">{text}</div>'
        code.value = _source_html(fn)

    stepper.observe(draw, "value")
    draw()
    return W.VBox([stepper, caption, fig, code])


def neighbourhood_explorer(D, graphs: dict) -> W.Widget:
    """graphs: {name: {k: EdgeList}}. Shows the neighbours of one row on the 2-D map."""
    adj = {(g, k): e.to_scipy(symmetric=True) for g, ks in graphs.items() for k, e in ks.items()}
    g_names, k_vals = list(graphs), sorted(next(iter(graphs.values())))
    P, y = D.map2d, D.y
    bg = np.random.default_rng(0).choice(D.tr, 12000, replace=False)
    full = [[P[:, 0].min() - 2, P[:, 0].max() + 2], [P[:, 1].min() - 2, P[:, 1].max() + 2]]
    bad_bg, good_bg = bg[y[bg] == 1], bg[y[bg] == 0]

    pick = np.random.default_rng(7)
    opts = [(f"test row {i}", int(i)) for i in pick.choice(D.te, 2, replace=False)]
    opts += [(f"train row {i} (defaulted)", int(i)) for i in pick.choice(D.tr[y[D.tr] == 1], 2, replace=False)]
    opts += [(f"train row {i} (repaid)", int(i)) for i in pick.choice(D.tr[y[D.tr] == 0], 2, replace=False)]

    def hov(idx):
        X = D.X.iloc[idx]
        return [f"row {i} ({D.part[i]})<br>limit {lim:,.0f}, age {a:.0f}, PAY_0 {p:.0f}"
                for i, lim, a, p in zip(idx, X["LIMIT_BAL"], X["AGE"], X["PAY_0"])]

    fig = go.FigureWidget()
    fig.add_scatter(x=P[good_bg, 0], y=P[good_bg, 1], mode="markers", name="repaid",
                    marker=dict(size=5, color=LIGHT, opacity=0.5), hoverinfo="skip")
    fig.add_scatter(x=P[bad_bg, 0], y=P[bad_bg, 1], mode="markers", name="defaulted",
                    marker=dict(size=5, color=RED, opacity=0.35), hoverinfo="skip")
    fig.add_scatter(x=[], y=[], mode="lines", line=dict(width=2.5, color=GOLD), name="edge", hoverinfo="skip")
    fig.add_scatter(x=[], y=[], mode="markers", name="neighbour, repaid", hovertemplate="%{text}<extra></extra>",
                    marker=dict(size=14, color=GREY, line=dict(width=2, color="white")))
    fig.add_scatter(x=[], y=[], mode="markers", name="neighbour, defaulted", hovertemplate="%{text}<extra></extra>",
                    marker=dict(size=14, color=RED, line=dict(width=2, color="white")))
    fig.add_scatter(x=[], y=[], mode="markers", name="neighbour, val/test (label unused)", hovertemplate="%{text}<extra></extra>",
                    marker=dict(size=14, color="white", line=dict(width=2.5, color=GREY)))
    fig.add_scatter(x=[], y=[], mode="markers", name="selected row", hovertemplate="%{text}<extra></extra>",
                    marker=dict(size=26, symbol="star", color=INK, line=dict(width=2, color="white")))
    fig.update_layout(height=640, transition=dict(duration=600, easing="cubic-in-out"),
                      xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=None),
                      yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=None),
                      legend=dict(font=dict(size=15)), margin=dict(l=10, r=10, t=120, b=10))

    who = W.Dropdown(options=opts, value=opts[0][1], description="Row", layout=W.Layout(width="360px"))
    rand_btn = W.Button(description="Random row", layout=W.Layout(width="130px"))
    graph = W.ToggleButtons(options=g_names, value=g_names[0], description="Graph")
    k = W.ToggleButtons(options=k_vals, value=k_vals[len(k_vals) // 2], description="k", style={"button_width": "60px"})
    info = W.HTML()

    def draw(*_):
        i, g = who.value, graph.value
        nb = adj[(g, k.value)][i].indices
        ex, ey = [], []
        for j in nb:
            ex += [P[i, 0], P[j, 0], None]
            ey += [P[i, 1], P[j, 1], None]
        lab = D.train_mask[nb]
        good, bad, hid = nb[lab & (y[nb] == 0)], nb[lab & (y[nb] == 1)], nb[~lab]
        off = np.abs(P[nb] - P[i]) if len(nb) else np.zeros((1, 2))
        half = np.maximum(np.quantile(off, 0.8, axis=0) * 1.35, 4.0)
        zoomed = bool((half * 2 < np.diff(full, axis=1).ravel() * 0.6).all())
        with fig.batch_update():
            fig.data[2].update(x=ex, y=ey, line=dict(color=GOLD if g == g_names[0] else GREY), opacity=0.9)
            for tr_, idx in zip(fig.data[3:6], (good, bad, hid)):
                tr_.update(x=P[idx, 0], y=P[idx, 1], text=hov(idx))
            fig.data[6].update(x=[P[i, 0]], y=[P[i, 1]], text=hov([i]))
            fig.layout.xaxis.range = [P[i, 0] - half[0], P[i, 0] + half[0]] if zoomed else full[0]
            fig.layout.yaxis.range = [P[i, 1] - half[1], P[i, 1] + half[1]] if zoomed else full[1]
            fig.layout.title.text = f"{g}, k={k.value}: neighbours of row {i}" + (" (zoomed in)" if zoomed else "")
        n_lab = int(lab.sum())
        rate = f"{len(bad) / n_lab:.0%}" if n_lab else "n/a"
        note = ("test row: its label is never used and it only has query edges into train/val"
                if D.part[i] == "test" else f"{D.part[i]} row, {'defaulted' if y[i] == 1 else 'repaid'}")
        info.value = (f'<div style="font:18px {FONT};line-height:1.6;padding:6px 2px">{note}<br>'
                      f'{len(nb)} neighbours, {n_lab} with a training label, {len(bad)} of those defaulted ({rate}). '
                      f'Base rate {np.mean(y[D.tr]):.0%}.</div>')

    def pick_random(_):
        j = int(np.random.default_rng().choice(np.concatenate([D.tr, D.te])))
        label = f"{D.part[j]} row {j}"
        if (label, j) not in who.options:
            who.options = list(who.options) + [(label, j)]
        who.value = j

    for w in (who, graph, k):
        w.observe(draw, "value")
    rand_btn.on_click(pick_random)
    draw()
    return W.VBox([W.HBox([who, rand_btn]), graph, k, fig, info])


def degree_comparison(D, knn_clean: dict, rand_clean: dict) -> W.Widget:
    """Rewiring keeps the directed degree, but not the degree after symmetrising."""
    from src.graphs.base import label_homophily

    def stats(e):
        und = e.degrees(symmetric=True)
        h = label_homophily(e.symmetrized(), D.y, D.train_mask)
        return dict(directed=e.n_edges / len(D.allowed), und=und[D.allowed], und_mean=und[D.allowed].mean(),
                    mutual=1 - e.symmetrized().n_edges / 2 / e.n_edges, homo=h["p_bad_neighbor_given_bad"] / h["base_rate"])

    S = {k: (stats(knn_clean[k]), stats(rand_clean[k])) for k in sorted(knn_clean)}
    fig = go.FigureWidget()
    for name, col in (("kNN", GOLD), ("random control", GREY)):
        fig.add_histogram(x=[], name=name, marker=dict(color=col, line=dict(width=1, color="white")), opacity=0.75,
                          xbins=dict(size=1), hovertemplate="%{y:,} rows with %{x} neighbours<extra>" + name + "</extra>")
    fig.update_layout(barmode="overlay", height=460, xaxis_title="neighbours after symmetrising",
                      yaxis_title="rows", title="Degree after symmetrising (dashed = mean)")
    k = W.ToggleButtons(options=list(S), value=10 if 10 in S else list(S)[0], description="k", style={"button_width": "60px"})
    table = W.HTML()

    def draw(*_):
        a, b = S[k.value]
        with fig.batch_update():
            fig.data[0].x, fig.data[1].x = a["und"], b["und"]
            fig.layout.shapes = [dict(type="line", x0=s["und_mean"], x1=s["und_mean"], y0=0, y1=1, yref="paper",
                                      line=dict(color=c, width=3, dash="dash")) for s, c in ((a, GOLD), (b, GREY))]
            fig.layout.xaxis.range = [0, max(np.percentile(b["und"], 99.5), np.percentile(a["und"], 99.5)) + 2]
        rows = [("out-edges per row (directed)", f"{a['directed']:.1f}", f"{b['directed']:.1f}"),
                ("neighbours after symmetrising", f"{a['und_mean']:.1f}", f"{b['und_mean']:.1f}"),
                ("mutual pairs", f"{a['mutual']:.0%}", f"{b['mutual']:.1%}"),
                ("P(bad neighbour | bad) / base rate", f"{a['homo']:.2f}", f"{b['homo']:.2f}")]
        td = 'style="padding:3px 16px 3px 0"'
        body = "".join(f"<tr><td {td}>{r[0]}</td><td {td}>{r[1]}</td><td {td}>{r[2]}</td></tr>" for r in rows)
        table.value = (f'<table style="font:18px {FONT};border-collapse:collapse;margin:4px 0 8px">'
                       f'<tr><th {td}></th><th {td}>kNN</th><th {td}>random</th></tr>{body}</table>')

    k.observe(draw, "value")
    draw()
    return W.VBox([k, table, fig])


# ---------------------------------------------------------------- 04 / 06: XGBoost
class Race:
    """Plots validation AUC per boosting round while XGBoost trains.
    Same params and early stopping as src.models.xgb.fit_xgb, plus a callback."""

    def __init__(self, D, title="Validation AUC by boosting round", y_range=None):
        self.D = D
        self.fig = go.FigureWidget()
        self.fig.update_layout(height=430, title=title, xaxis_title="boosting round", yaxis_title="validation AUC",
                               yaxis=dict(range=y_range, tickformat=".3f"), hovermode="x unified")
        self.status = W.HTML()
        self.show()

    def show(self):
        display(W.VBox([self.status, self.fig]))

    def _status(self, text):
        self.status.value = f'<div style="font:17px {FONT};color:{MUTED}">{text}</div>'

    def train(self, name: str, XD: pd.DataFrame, color: str, dash: str = "solid", seed: int = 0):
        D = self.D
        p = {**XGB.DEFAULTS, **D.cfg["models"]["xgb"]}
        rounds, early = int(p.pop("num_boost_round")), int(p.pop("early_stopping_rounds"))
        p.update(seed=seed, device=xgb_device())
        dtr = XGB._dmatrix(XD.iloc[D.tr], D.y[D.tr].astype(np.float32))
        dva = XGB._dmatrix(XD.iloc[D.va], D.y[D.va].astype(np.float32))
        self.fig.add_scatter(x=[], y=[], mode="lines", name=name, line=dict(color=color, width=3, dash=dash),
                             hovertemplate="%{y:.4f}")
        line, xs, ys, fig, race = self.fig.data[-1], [], [], self.fig, self
        t0 = time.time()

        class Live(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                xs.append(epoch)
                ys.append(evals_log["val"]["auc"][-1])
                if epoch % 10 == 0:
                    with fig.batch_update():
                        line.x, line.y = xs, ys
                    race._status(f"{name}: round {epoch}, val AUC {ys[-1]:.4f}")
                return False

        booster = xgb.train(p, dtr, num_boost_round=rounds, evals=[(dva, "val")], early_stopping_rounds=early,
                            verbose_eval=False, callbacks=[Live()])
        b = booster.best_iteration
        with fig.batch_update():
            line.x, line.y = xs, ys
            fig.add_scatter(x=[b], y=[ys[b]], mode="markers", showlegend=False, hovertemplate=f"{name}: best round %{{x}}<extra></extra>",
                            marker=dict(size=13, color=color, line=dict(width=2, color="white")))
        self._status(f"{name}: best val AUC {ys[b]:.4f} at round {b} ({time.time() - t0:.1f}s)")
        return booster, XGB.predict_xgb(booster, XD.iloc[D.va]), XGB.predict_xgb(booster, XD.iloc[D.te])


class Scoreboard:
    """Test AUC minus the baseline, with a 95% paired-bootstrap CI. Rows are added as models finish."""

    def __init__(self, D, base_name: str, p_base_te: np.ndarray, n_boot: int = 400):
        self.D, self.base, self.n_boot, self.rows = D, p_base_te, n_boot, []
        base_auc = metrics.auc(D.y[D.te], p_base_te)
        self.fig = go.FigureWidget()
        self.fig.update_layout(height=200, title=f"Test AUC minus {base_name} ({base_auc:.3f}), 95% CI",
                               xaxis=dict(title="difference in test AUC", zeroline=True, zerolinewidth=3, zerolinecolor=INK, tickformat="+.3f"),
                               yaxis=dict(autorange="reversed"), showlegend=False, margin=dict(l=20, r=260, t=70, b=60))
        self.show()

    def show(self):
        display(self.fig)

    def add(self, name: str, p_te: np.ndarray, color: str):
        r = paired_bootstrap(self.D.y[self.D.te], self.base, p_te, "auc", n_boot=self.n_boot, seed=0)
        verdict = "better" if r["ci_low"] > 0 else "worse" if r["ci_high"] < 0 else "no significant difference"
        self.rows.append(dict(name=name, color=color, verdict=verdict, **r))
        df = pd.DataFrame(self.rows)
        with self.fig.batch_update():
            self.fig.data = []
            self.fig.add_scatter(x=df["diff"], y=df["name"], mode="markers",
                                 error_x=dict(type="data", symmetric=False, array=df.ci_high - df["diff"],
                                              arrayminus=df["diff"] - df.ci_low, thickness=3, width=8, color=GREY),
                                 marker=dict(size=16, color=df.color, line=dict(width=2, color="white")),
                                 customdata=np.c_[df["new"], df.ci_low, df.ci_high],
                                 hovertemplate="%{y}<br>test AUC %{customdata[0]:.4f}<br>diff %{x:+.4f} "
                                               "[%{customdata[1]:+.4f}, %{customdata[2]:+.4f}]<extra></extra>")
            lim = max(0.01, float(np.abs(df[["ci_low", "ci_high"]].to_numpy()).max()) * 1.6)
            self.fig.layout.xaxis.range = [-lim, lim]
            self.fig.layout.height = 170 + 62 * len(df)
            self.fig.layout.annotations = [dict(x=1.01, xref="paper", y=row.name, yref="y", xanchor="left", showarrow=False,
                                                text=f"{row.verdict} ({row.new:.3f})", font=dict(size=16, color=MUTED))
                                           for row in df.itertuples()]
        return r


# ---------------------------------------------------------------- results from the full runs (report/)
def results_explorer(highlight: tuple[str, str] | None = ("taiwan", "knn")) -> W.Widget:
    """Lift over XGBoost for every dataset and graph, mean of 5 seeds (report/lifts_vs_xgb.csv)."""
    L = pd.read_csv(REPORT / "lifts_vs_xgb.csv")
    L["kind"], L["ds"] = L.graph_builder.map(KIND_LABEL), L.dataset.map(DS_LABEL)
    tiers = {"Tier 1: graph features + XGBoost": "04_graph_features_xgb", "Tier 2: GNN": "05_gnn",
             "Tier 3: GNN embedding + XGBoost": "06_hybrid"}
    mets = {"AUC": "auc", "capture@10%": "capture_10"}
    tier = W.ToggleButtons(options=list(tiers), style={"button_width": "290px"})
    met = W.ToggleButtons(options=list(mets), style={"button_width": "140px"})
    fig = go.FigureWidget()
    fig.update_layout(height=560, showlegend=False, margin=dict(l=20, r=60, t=80, b=60),
                      yaxis=dict(autorange="reversed", automargin=True))
    order = ["dgraphfin", "home_credit", "lending_club", "taiwan"]

    def draw(*_):
        d = L[(L.stage == tiers[tier.value]) & (L.metric == mets[met.value])].copy()
        d["o"] = d.dataset.map({k: i for i, k in enumerate(order)})
        d = d.sort_values(["o", "graph_builder", "model"])
        model = d.model.str.replace("hybrid_", "").str.replace("xgb_graphfeat", "").str.upper()
        d["label"] = d.ds + ", " + d.kind + np.where(model != "", " (" + model + ")", "")
        is_hl = (d.dataset == highlight[0]) & (d.graph_builder == highlight[1]) if highlight else np.zeros(len(d), bool)
        with fig.batch_update():
            fig.data = []
            fig.add_scatter(x=d["diff"], y=d.label, mode="markers",
                            error_x=dict(type="data", symmetric=False, array=d.ci_high - d["diff"], arrayminus=d["diff"] - d.ci_low,
                                         thickness=3, width=7, color="#9aa1a8"),
                            marker=dict(size=17, color=d.graph_builder.map(KIND_COLOR).tolist(), line=dict(width=2, color="white")),
                            customdata=np.c_[d.ci_low, d.ci_high, d.sig_seeds],
                            hovertemplate="%{y}<br>diff %{x:+.4f} [%{customdata[0]:+.4f}, %{customdata[1]:+.4f}]"
                                          "<br>significant in %{customdata[2]} seeds<extra></extra>")
            fig.add_scatter(x=d["diff"][is_hl], y=d.label[is_hl], mode="markers", hoverinfo="skip",
                            marker=dict(size=30, color="rgba(0,0,0,0)", line=dict(width=3, color=INK)))
            fig.layout.title = f"{tier.value}, {met.value} vs XGBoost (5 seeds)"
            fig.layout.xaxis = dict(title="difference vs XGBoost", zeroline=True, zerolinewidth=3, zerolinecolor=INK,
                                    tickformat="+.0%" if mets[met.value] == "capture_10" else "+.2f")
            fig.layout.height = 160 + 44 * len(d)

    tier.observe(draw, "value")
    met.observe(draw, "value")
    draw()
    key = ", ".join(f'<span style="color:{c}">&#9679;</span> {lbl}' for lbl, c in
                    (("real edges", GREEN), ("kNN / shared attribute", GOLD), ("relational", BLUE), ("random control", GREY)))
    note = "; circled = the Taiwan kNN run above" if highlight else ""
    return W.VBox([tier, met, W.HTML(f'<div style="font:16px {FONT};color:{MUTED}">{key}{note}</div>'), fig])


def dgraphfin_summary() -> None:
    """DGraph-Fin test AUC by tier (mean of 5 seeds) and the business numbers from report/."""
    R = pd.read_csv(REPORT / "results_test_mean.csv")
    B = pd.read_csv(REPORT / "business.csv")
    dd = R[R.dataset == "dgraphfin"].set_index("model")
    d = dd.auc
    rows = [("XGBoost (tabular only)", d["xgb"], GREY), ("GAT", d["gat"], BLUE), ("GraphSAGE", d["sage"], BLUE),
            ("SAGE embedding + XGBoost", d["hybrid_sage"], PURPLE), ("graph features + XGBoost", d["xgb_graphfeat"], GREEN)]
    fig = go.Figure()
    for lbl, v, c in rows:
        fig.add_scatter(x=[d["xgb"], v], y=[lbl, lbl], mode="lines", line=dict(color=LIGHT, width=6), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=[v], y=[lbl], mode="markers+text", text=[f"  {v:.3f}"], textposition="middle right",
                        marker=dict(size=20, color=c, line=dict(width=2, color="white")), showlegend=False,
                        hovertemplate=f"{lbl}<br>test AUC %{{x:.4f}}<extra></extra>", textfont=dict(size=18))
    fig.update_layout(height=420, title="DGraph-Fin (3.7M nodes, real edges): test AUC, mean of 5 seeds",
                      xaxis=dict(range=[0.70, 0.86], title="test AUC"), yaxis=dict(autorange="reversed"))
    fig.show()
    t1 = B[(B.dataset == "dgraphfin") & B.model.str.contains("xgb_graphfeat")].iloc[0]
    print(f"graph features + XGBoost vs XGBoost:\n"
          f"  approval rate at the same bad rate ({t1.target_bad_rate:.2%}): +{t1.approval_delta_pp:.1f} pp\n"
          f"  share of defaulters in the riskiest 10%: {dd.capture_10['xgb']:.0%} -> {t1.capture_10:.0%}")


# ---------------------------------------------------------------- 05: GNN
def adjacency(edges, n: int):
    """Row-normalised adjacency as a torch CSR tensor, so A @ x is the neighbour mean."""
    import torch

    A = edges.to_scipy(symmetric=True).tocoo()
    deg = np.asarray(A.sum(1)).ravel()
    w = 1.0 / np.maximum(deg[A.row], 1)
    coo = torch.sparse_coo_tensor(np.vstack([A.row, A.col]), w.astype(np.float32), (n, n)).coalesce()
    return coo.to_sparse_csr()   # much faster than COO for A @ x on CPU


class GnnMonitor:
    def __init__(self, title="GraphSAGE training"):
        self.fig = go.FigureWidget(make_subplots(1, 2, subplot_titles=("training loss", "validation AUC"), horizontal_spacing=0.12))
        for col in (1, 2):
            self.fig.add_scatter(x=[], y=[], mode="lines+markers", line=dict(color=BLUE, width=3), marker=dict(size=6),
                                 row=1, col=col, hovertemplate="epoch %{x}: %{y:.4f}<extra></extra>")
        self.fig.update_layout(height=400, showlegend=False, title=title)
        self.fig.update_xaxes(title_text="epoch")
        self.status = W.HTML()
        display(W.VBox([self.status, self.fig]))

    def update(self, ep, loss, auc, best, extra=""):
        a, b = self.fig.data
        with self.fig.batch_update():
            a.x, a.y = list(a.x) + [ep], list(a.y) + [loss]
            b.x, b.y = list(b.x) + [ep], list(b.y) + [auc]
        self.status.value = (f'<div style="font:17px {FONT};color:{MUTED}">epoch {ep}, loss {loss:.4f}, '
                             f'val AUC {auc:.4f} (best {best:.4f}){extra}</div>')


def train_gnn(model, x, A, D, epochs=100, lr=1e-2, patience=10, snap_at=(0, 1, 2, 4, 8, 16, 32), seed=0, pace=0.0):
    """Full-batch training. Loss on train rows only, early stopping on val AUC.
    Saves validation embeddings at a few epochs for embedding_animation()."""
    import copy

    import torch

    torch.manual_seed(seed)
    y = torch.tensor(np.nan_to_num(D.y), dtype=torch.float32)
    tr, va = torch.tensor(D.tr), torch.tensor(D.va)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    mon = GnnMonitor()
    snaps, best, best_state, wait = {}, -1.0, None, 0
    samp = np.random.default_rng(0).choice(len(D.va), 3000, replace=False)

    def snapshot(key):
        model.eval()
        with torch.no_grad():
            _, emb = model(x, A)
        snaps[key] = emb[va].numpy()[samp]

    snapshot(0)
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logit, _ = model(x, A)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit[tr], y[tr])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            logit, _ = model(x, A)
        auc = metrics.auc(D.y[D.va], torch.sigmoid(logit[va]).numpy())
        if auc > best:
            best, best_state, wait = auc, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if ep in snap_at:
            snapshot(ep)
        mon.update(ep, float(loss), auc, best)
        if pace:
            time.sleep(pace)
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    snapshot("best")
    mon.update(ep, float(loss), auc, best, extra=f", stopped after {patience} epochs without improvement")
    model.eval()
    with torch.no_grad():
        logit, emb = model(x, A)
    return torch.sigmoid(logit).numpy(), emb.numpy(), snaps, D.y[D.va][samp]


def embedding_animation(snaps: dict, y_s: np.ndarray) -> None:
    """PCA of the 32-d validation embeddings at each saved epoch."""
    frames, keys = [], list(snaps)
    for key in keys:
        pc = PCA(2, random_state=0).fit_transform(snaps[key])
        pc = (pc - pc.mean(0)) / (pc.std(0) + 1e-9)
        pc[:, 0] *= np.sign(np.corrcoef(pc[:, 0], y_s)[0, 1] or 1)   # keep defaulters on the same side across frames
        frames.append(go.Frame(name=str(key), data=[
            go.Scattergl(x=pc[y_s == 0, 0], y=pc[y_s == 0, 1]), go.Scattergl(x=pc[y_s == 1, 0], y=pc[y_s == 1, 1])]))
    f0 = frames[0].data
    fig = go.Figure(data=[go.Scattergl(x=f0[0].x, y=f0[0].y, mode="markers", name="repaid", marker=dict(size=6, color=LIGHT, opacity=0.6)),
                          go.Scattergl(x=f0[1].x, y=f0[1].y, mode="markers", name="defaulted", marker=dict(size=7, color=RED, opacity=0.7))],
                    frames=frames)
    lab = lambda k: "untrained" if k == 0 else "best" if k == "best" else f"epoch {k}"
    fig.update_layout(
        height=640, title="GraphSAGE embeddings of validation rows (PCA)",
        xaxis=dict(range=[-4, 4], showticklabels=False, title=None, zeroline=False),
        yaxis=dict(range=[-4, 4], showticklabels=False, title=None, zeroline=False),
        updatemenus=[dict(type="buttons", x=1, y=1.01, xanchor="right", yanchor="bottom", direction="left",
                          buttons=[dict(label="Play", method="animate",
                                        args=[None, dict(frame=dict(duration=900, redraw=True), transition=dict(duration=400), fromcurrent=True)])])],
        sliders=[dict(x=0, y=0, len=1, pad=dict(t=20), currentvalue=dict(prefix="", font=dict(size=17)),
                      steps=[dict(label=lab(k), method="animate", args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                             for k in keys])])
    fig.show()


def attribution_share(shares: dict) -> None:
    """Share of the hybrid's SHAP attribution on embedding vs tabular columns."""
    fig = go.Figure()
    for name, s in shares.items():
        fig.add_bar(y=[name], x=[s["emb_share"]], orientation="h", marker_color=PURPLE, marker_line=dict(width=2, color="white"),
                    text=[f"embedding {s['emb_share']:.0%}"], textposition="inside", insidetextanchor="start")
        fig.add_bar(y=[name], x=[s["tab_share"]], orientation="h", marker_color=LIGHT, marker_line=dict(width=2, color="white"),
                    text=[f"tabular {s['tab_share']:.0%}"], textposition="inside", insidetextanchor="end")
    fig.update_layout(barmode="stack", height=230, showlegend=False, title="SHAP attribution: embedding vs tabular columns",
                      xaxis=dict(range=[0, 1], tickformat=".0%", title=None))
    fig.update_traces(textfont_size=18)
    fig.show()
