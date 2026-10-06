"""Separability gain (Wang et al., ICML 2024, Theorem 2 and Definition 2) on Minesweeper and
the five Bodnar et al. benchmarks.

gain(k, t) = (1/sqrt 2) * || sqrt(d_k) * m_k - sqrt(d_t) * m_t ||
  m_k[i] : fraction of class-i nodes among the neighbours of class-k nodes
  d_k    : mean degree of class-k nodes
Good heterophily: min over class pairs > threshold; bad: max < threshold; otherwise mixed.
In theory the threshold is about 1; for real-world data the paper suggests 0.2 (Sec. 5.2).

Graphs are symmetrised with self-loops and duplicate edges removed. Data sources are the
ones PyTorch Geometric uses (HeterophilousGraphDataset, WebKB, WikipediaNetwork geom-gcn).
Usage: python heterophily_gain.py
"""
import io
import itertools
import urllib.request

import numpy as np

GEOM_GCN = "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/{commit}/new_data/{name}/{file}"
WEBKB_COMMIT = "1c4c04f93fa6ada91976cda8d7577eec0e3e5cce"
WIKI_COMMIT = "f1fc0d14b3b019c562737240d06ec83b07d16a8f"
MINESWEEPER = "https://github.com/yandex-research/heterophilous-graphs/raw/main/data/minesweeper.npz"
THRESHOLD = 0.2


def _get(url):
    return urllib.request.urlopen(url).read()


def load_geom_gcn(name, commit):
    def lines(file):
        text = _get(GEOM_GCN.format(commit=commit, name=name, file=file)).decode()
        return [r for r in text.split("\n")[1:] if r.strip()]

    labels = {int(r.split("\t")[0]): int(r.split("\t")[2]) for r in lines("out1_node_feature_label.txt")}
    y = np.array([labels[i] for i in range(max(labels) + 1)])
    edges = np.array([list(map(int, r.split("\t"))) for r in lines("out1_graph_edges.txt")])
    return y, edges


def load_minesweeper():
    d = np.load(io.BytesIO(_get(MINESWEEPER)))
    return d["node_labels"], d["edges"]


def heterophily_gain(y, edges):
    edges = edges[edges[:, 0] != edges[:, 1]]
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    n, C = len(y), y.max() + 1
    deg = np.bincount(src, minlength=n)
    m, d = np.zeros((C, C)), np.zeros(C)
    for k in range(C):
        nbr = y[dst[y[src] == k]]
        m[k] = np.bincount(nbr, minlength=C) / len(nbr)
        d[k] = deg[y == k].mean()
    gains = [np.linalg.norm(np.sqrt(d[a]) * m[a] - np.sqrt(d[b]) * m[b]) / np.sqrt(2)
             for a, b in itertools.combinations(range(C), 2)]
    edge_homophily = np.mean(y[src] == y[dst])
    return min(gains), max(gains), edge_homophily


def verdict(gmin, gmax):
    return "good" if gmin > THRESHOLD else "bad" if gmax < THRESHOLD else "mixed"


if __name__ == "__main__":
    datasets = {name: (lambda n=name: load_geom_gcn(n, WEBKB_COMMIT)) for name in ["texas", "wisconsin", "cornell"]}
    datasets.update({name: (lambda n=name: load_geom_gcn(n, WIKI_COMMIT)) for name in ["chameleon", "squirrel"]})
    datasets["minesweeper"] = load_minesweeper
    print(f"{'dataset':<12} {'min gain':>9} {'max gain':>9} {'edge hom.':>10}  verdict")
    for name, loader in datasets.items():
        gmin, gmax, h = heterophily_gain(*loader())
        print(f"{name:<12} {gmin:>9.3f} {gmax:>9.3f} {h:>10.3f}  {verdict(gmin, gmax)}")
