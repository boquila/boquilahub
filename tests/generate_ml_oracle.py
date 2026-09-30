# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.5.3", "scipy==1.18.1", "scikit-learn==1.9.1"]
# ///
"""Regenerate with: uv run --no-project tests/generate_ml_oracle.py"""

import json
from pathlib import Path
import warnings

import numpy as np
import scipy
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


def scale(data):
    return float(np.max(np.abs(data))) or 1.0


def farthest_first(data, count):
    """Match our explicit seed policy, including choosing the last exact tie."""
    centers = [data[0].copy()]
    distances = np.full(len(data), np.inf)
    while len(centers) < min(count, len(data)):
        distances = np.minimum(distances, np.sum((data - centers[-1]) ** 2, axis=1))
        if distances.max() == 0.0:
            break
        index = np.flatnonzero(distances == distances.max())[-1]
        centers.append(data[index].copy())
    return np.asarray(centers)


def kmeans_case(name, data, count):
    data = np.asarray(data, dtype=np.float64)
    factor = scale(data)
    normalized = data / factor
    initial = farthest_first(normalized, count)
    fit = KMeans(
        n_clusters=len(initial), init=initial, n_init=1,
        algorithm="lloyd", tol=0.0, max_iter=1000, random_state=0,
    ).fit(normalized)
    assert fit.n_iter_ < 1000, name
    return dict(
        name=name, data=data.tolist(), count=count,
        initial_centers=(initial * factor).tolist(),
        centers=(fit.cluster_centers_ * factor).tolist(),
        labels=fit.labels_.tolist(), inertia=float(fit.inertia_), iterations=fit.n_iter_,
    )


def pca_case(name, data, count):
    data = np.asarray(data, dtype=np.float64)
    factor = scale(data)
    normalized = data / factor
    # Full LAPACK SVD is the reference, without whitening or per-feature scaling.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fit = PCA(n_components=min(count, *data.shape), svd_solver="full").fit(normalized)
    components = np.zeros((min(count, data.shape[1]), data.shape[1]))
    variance = np.zeros(len(components))
    tolerance = fit.singular_values_[0] * np.finfo(np.float64).eps * max(data.shape)
    # Null-space bases are arbitrary. Our contract uses zero directions instead.
    if not np.all(data == data[0]):
        for i, singular in enumerate(fit.singular_values_):
            if singular > tolerance:
                components[i] = fit.components_[i]
                variance[i] = fit.explained_variance_[i]
    scores = (normalized - fit.mean_) @ components.T
    return dict(
        name=name, data=data.tolist(), count=count,
        mean=(fit.mean_ * factor).tolist(), components=components.tolist(),
        variance=variance.tolist(), scores=(scores * factor).tolist(),
    )


def main():
    rng = np.random.default_rng(20261001)
    minor_first = [[0, 1], [0, -1], [4, 0], [-4, 0]]
    correlated = rng.normal(size=(80, 5)) @ rng.normal(size=(5, 5)) + [4, -2, 10, 1, 0]
    embeddings = rng.normal(size=(32, 64))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    pca_inputs = [
        ("minor_axis_first", minor_first, 2),
        ("translated_correlated", correlated, 3),
        ("equal_eigenvalues", np.vstack([np.eye(3), -np.eye(3)]), 3),
        ("nearly_equal_eigenvalues", np.vstack([np.diag([1 + 1e-8, 1, 0.1]), -np.diag([1 + 1e-8, 1, 0.1])]), 2),
        ("rank_one", [[i, 2 * i, 3 * i] for i in range(5)], 3),
        ("constant", [[1.1, 2.3, 5]] * 17, 3),
        ("singleton", [[2, 3, 4]], 3),
        ("wide_matrix", rng.normal(size=(12, 32)), 3),
        ("normalized_embeddings", embeddings, 3),
        ("thin_rank_deficient", rng.normal(size=(4, 8)), 3),
        ("small_variance_direction", [[-1, 0], [1, 0], [0, -1e-8], [0, 1e-8]], 2),
        ("tiny_scale", np.asarray(minor_first) * 1e-100, 2),
        ("huge_scale", np.asarray(minor_first) * 1e100, 2),
        ("stride_aliasing", [[0 if i % 2 == 0 else 2, 0] for i in range(1024)], 2),
    ]
    xy = [[-0.1, 0], [0.1, 0], [-0.1, 0], [0.1, 0]]
    xyz = [[-0.1, 0, -1], [0.1, 0, -1], [-0.1, 0, 1], [0.1, 0, 1]]
    chain = np.asarray([[x] for x in [0, 0.4, 0.45, 0.49, 0.51, 0.56, 1]])
    kmeans_inputs = [
        ("visible_2d_coordinates", xy, 2),
        ("three_dimensional_coordinates", xyz, 2),
        ("separated_2d", np.vstack([rng.normal([x, y], 0.1, size=(30, 2)) for x, y in [(-2, -2), (0, 2), (2, -2)]]), 3),
        ("separated_3d", np.vstack([rng.normal([x, y, z], 0.1, size=(20, 3)) for x, y, z in [(-2, -2, -2), (0, 2, -2), (2, -2, -2), (0, 0, 2)]]), 4),
        ("stride_aliasing", [[i % 2, 0] for i in range(4096)], 2),
        ("small_scale_convergence", chain * 1e-6, 2),
        ("tiny_scale", chain * 1e-100, 2),
        ("huge_scale", chain * 1e100, 2),
        ("duplicates_excess_k", [[0], [1], [1], [2]], 20),
        ("one_cluster", correlated, 1),
        ("singleton", [[2, 3]], 4),
        ("constant", [[2, 3]] * 17, 4),
        ("normalized_embeddings", embeddings, 5),
    ]
    for i in range(10):
        kmeans_inputs.append((f"dense_random_{i}", rng.normal(size=(150 + i * 10, 2 + i % 2)), 3 + i))
    kmeans = [kmeans_case(*case) for case in kmeans_inputs]
    # Include a deterministic full-data case where the former 20-step cap fails.
    for trial in range(100):
        case = kmeans_case("converges_after_twenty_steps", rng.uniform(-1, 1, size=(1029, 2)), 7)
        if case["iterations"] > 20:
            kmeans.append(case)
            break
    else:
        raise AssertionError("did not find a long-convergence regression case")
    oracle = dict(
        versions=dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
        description="PCA: full SVD. K-means: explicit farthest-first seeds, n_init=1, tol=0, max_iter=1000. Inertia and variance use a single global input scale to avoid overflow. Numerical libraries run with one thread for reproducible reductions.",
        pca=[pca_case(*case) for case in pca_inputs],
        kmeans=kmeans,
    )
    path = Path(__file__).with_name("oracles") / "ml.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(oracle, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {path}: {len(oracle['pca'])} PCA and {len(oracle['kmeans'])} k-means cases")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
