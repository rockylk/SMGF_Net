"""IBGC graph equations, reimplemented for the frozen [direction,size,IAT] cache."""
import numpy as np

REPRESENTATION = "IBGC-V17-v1-13features-fullRaw400-intSize-noFilter-noPad-noIAT"
FEATURE_NAMES = (
    "signed_size", "slice_count", "previous_count", "next_count",
    "previous_count_ratio", "next_count_ratio", "slice_bytes",
    "previous_bytes", "next_bytes", "packet_byte_fraction", "slice_mean_size",
    "bytes_over_previous", "bytes_over_next",
)


def validate_raw(raw):
    a = np.asarray(raw)
    if a.ndim != 2 or a.shape[1] != 3 or not 1 <= len(a) <= 400:
        raise ValueError("Raw sequence must have shape [1..400,3]; no dropping/padding")
    if not np.isfinite(a).all():
        raise ValueError("Non-finite raw input")
    if not np.isin(a[:, 0], [0, 1]).all():
        raise ValueError("Direction must already be the frozen 0/1 encoding")
    if (a[:, 1] < 1).any() or (a[:, 2] < 0).any():
        raise ValueError("Size must be >=1; IAT must be >=0")
    return a


def build_graph(raw):
    """One packet/node; chains within slices; complete bipartite adjacent slices.

    Direction 0 has negative size, direction 1 positive (official convention).
    Truncation toward zero preserves the official int-size feature equations,
    including for fractional dummy sizes emitted by the frozen attack module.
    IAT is validated but deliberately not exposed to the classifier.
    """
    a = validate_raw(raw)
    d, size = a[:, 0], np.trunc(a[:, 1]).astype(np.float64)
    bounds = np.r_[0, np.flatnonzero(d[1:] != d[:-1]) + 1, len(d)]
    counts = np.diff(bounds)
    totals = np.add.reduceat(size, bounds[:-1])
    x = np.empty((len(a), 13), dtype=np.float32)
    edges = []
    for k, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        n, total = counts[k], totals[k]
        pn = counts[k - 1] if k else 0
        nn = counts[k + 1] if k + 1 < len(counts) else 0
        pb = totals[k - 1] if k else 0
        nb = totals[k + 1] if k + 1 < len(totals) else 0
        x[lo:hi, 0] = size[lo:hi] * (2 * d[lo:hi] - 1)
        x[lo:hi, 1:9] = [n, pn, nn, pn / n, nn / n, total, pb, nb]
        x[lo:hi, 9] = size[lo:hi] / total
        x[lo:hi, 10:] = [total / n, total / pb if pb else 0,
                         total / nb if nb else 0]
        # Explicit directed pairs represent an undirected graph, as upstream.
        if hi - lo > 1:
            u = np.arange(lo, hi - 1)
            edges.extend([np.stack([u, u + 1]), np.stack([u + 1, u])])
        if k:
            u = np.repeat(np.arange(bounds[k - 1], lo), hi - lo)
            v = np.tile(np.arange(lo, hi), lo - bounds[k - 1])
            edges.extend([np.stack([u, v]), np.stack([v, u])])
    edge = np.concatenate(edges, axis=1) if edges else np.empty((2, 0), np.int64)
    if edge.shape[1]:
        edge = edge[:, np.lexsort((edge[1], edge[0]))]
    if not np.isfinite(x).all():
        raise ValueError("Non-finite graph features")
    return x, edge.astype(np.int64, copy=False)
