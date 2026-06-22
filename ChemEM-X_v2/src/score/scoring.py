#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-contained ChemEM scoring engine (reproduced in the GUI).

This module is GUI-agnostic: every function takes plain numpy arrays / a
``MapLike`` density object and returns floats. The map metrics (CCC, MI, SCI,
QScore) are pure numpy/scipy ports of the ChemEM backend so the GUI produces the
same numbers as the pipeline without spawning a backend job. MMGBSA reuses
OpenMM/ParmEd/mdtraj (imported lazily, only when requested).

Ported from:
  - ChemEM/protocols/core/density.py        (blur + mutual_information_score)
  - ChemEM/protocols/core/sci_score.py      (truncated_cc, amp-eq, derivatives, SCI)
  - ChemEM/tools/map_q_score.py             (Q-score)
  - ChemEM/protocols/_docking/mmgbsa_score.py (MMGBSA energy decomposition)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Lightweight EMMap-like container (the interface the ported math expects)
# ---------------------------------------------------------------------------

class MapLike:
    """Minimal stand-in for the backend ``EMMap``.

    Attributes mirror what the ported scoring functions read:
      - origin: (x, y, z) in Å of voxel (0, 0, 0)
      - apix:   (ax, ay, az) in Å/voxel
      - density_map: ndarray in (z, y, x) order
    """

    def __init__(self, origin, apix, density_map, resolution=0.0):
        self.origin = tuple(float(o) for o in origin)
        self.apix = tuple(float(a) for a in apix)
        self.density_map = np.asarray(density_map)
        self.resolution = float(resolution)

    def copy(self):
        return MapLike(self.origin, self.apix,
                       np.array(self.density_map, copy=True), self.resolution)


def _as_xyz(v) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.size == 1:
        return np.repeat(a, 3)
    if a.size != 3:
        raise ValueError(f"Expected scalar or length-3, got shape {a.shape}")
    return a


def _zscore_array(x, eps: float = 1e-12):
    x = np.asarray(x, dtype=np.float64)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < eps:
        return x - mu
    return (x - mu) / sd


# ---------------------------------------------------------------------------
# Ligand -> density (rasterise + Gaussian blur over the support subgrid)
# ---------------------------------------------------------------------------

def simulate_ligand_density_subgrid(
    coords_xyz_A,
    atom_masses,
    map_origin_xyz_A,
    map_apix_xyz_A,
    map_shape_zyx,
    *,
    resolution_A: float,
    sigma_coeff: float = 0.356,
    normalise: bool = True,
    truncate: float = 4.0,
):
    """Rasterise ligand masses and blur only over the Gaussian support box.

    Returns ``(sim_subgrid, lo_zyx, hi_zyx)`` where the bounding box is half-open
    ``[lo, hi)`` in (z, y, x). The full simulated map is exactly zero outside this
    box for the same ``truncate`` value.
    """
    coords = np.asarray(coords_xyz_A, dtype=np.float64)
    masses = np.asarray(atom_masses, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords_xyz_A must be (N,3), got {coords.shape}")
    if masses.shape[0] != coords.shape[0]:
        raise ValueError("atom_masses length does not match coordinates")

    origin = _as_xyz(map_origin_xyz_A)
    apix = _as_xyz(map_apix_xyz_A)
    nz, ny, nx = [int(i) for i in map_shape_zyx]

    sigma_A = float(sigma_coeff) * float(resolution_A)
    sigma_zyx = np.array([
        sigma_A / max(apix[2], 1e-12),
        sigma_A / max(apix[1], 1e-12),
        sigma_A / max(apix[0], 1e-12),
    ], dtype=np.float64)
    radius_zyx = np.asarray(
        [int(float(truncate) * float(s) + 0.5) for s in sigma_zyx], dtype=int)

    voxel_indices = []
    voxel_masses = []
    for xyz, mass in zip(coords, masses):
        ix = int(np.rint((float(xyz[0]) - origin[0]) / apix[0]))
        iy = int(np.rint((float(xyz[1]) - origin[1]) / apix[1]))
        iz = int(np.rint((float(xyz[2]) - origin[2]) / apix[2]))
        if ix < 0 or iy < 0 or iz < 0:
            continue
        if ix >= nx or iy >= ny or iz >= nz:
            continue
        voxel_indices.append((iz, iy, ix))
        voxel_masses.append(float(mass))

    if not voxel_indices:
        empty = np.zeros((0, 0, 0), dtype=np.float64)
        zero = np.zeros(3, dtype=int)
        return empty, zero, zero

    idx_zyx = np.asarray(voxel_indices, dtype=int)
    shape_zyx = np.asarray([nz, ny, nx], dtype=int)
    lo_zyx = np.maximum(np.min(idx_zyx, axis=0) - radius_zyx, 0)
    hi_zyx = np.minimum(np.max(idx_zyx, axis=0) + radius_zyx + 1, shape_zyx)

    grid = np.zeros(tuple((hi_zyx - lo_zyx).tolist()), dtype=np.float64)
    for (iz, iy, ix), mass in zip(idx_zyx, voxel_masses):
        grid[int(iz - lo_zyx[0]), int(iy - lo_zyx[1]), int(ix - lo_zyx[2])] += mass

    sim = gaussian_filter(grid, sigma=sigma_zyx, mode="constant", cval=0.0)

    if normalise:
        vmax = float(np.max(sim))
        if vmax > 0.0:
            sim = sim / vmax

    return np.asarray(sim, dtype=np.float64), lo_zyx.astype(int), hi_zyx.astype(int)


# ---------------------------------------------------------------------------
# CCC / SCI helpers
# ---------------------------------------------------------------------------

def _safe_mask(mask, a, b):
    if mask is None:
        m = np.isfinite(a) & np.isfinite(b) & ((a != 0.0) | (b != 0.0))
        if int(np.count_nonzero(m)) < 64:
            m = np.isfinite(a) & np.isfinite(b)
        return m
    m = np.asarray(mask, dtype=bool)
    if m.shape != a.shape:
        raise ValueError(f"mask shape {m.shape} does not match map shape {a.shape}")
    return m & np.isfinite(a) & np.isfinite(b)


def truncated_cc(a, b, mask=None, eps: float = 1e-12) -> float:
    """Pearson cross-correlation on the masked overlap (clamped to >= 0)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Map shape mismatch: {a.shape} vs {b.shape}")

    m = _safe_mask(mask, a, b)
    av = a[m].ravel()
    bv = b[m].ravel()
    if av.size < 4:
        return 0.0

    av = av - float(np.mean(av))
    bv = bv - float(np.mean(bv))

    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom < eps:
        return 0.0

    cc = float(np.dot(av, bv) / denom)
    if not np.isfinite(cc):
        return 0.0
    return max(0.0, cc)


def amplitude_equalize_pair(map_a, map_b):
    """Keep each map's phase, replace both amplitudes with their average."""
    a = np.asarray(map_a, dtype=np.float64)
    b = np.asarray(map_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Map shape mismatch: {a.shape} vs {b.shape}")

    fa = np.fft.fftn(a)
    fb = np.fft.fftn(b)
    amp = 0.5 * (np.abs(fa) + np.abs(fb))
    a_eq = np.fft.ifftn(amp * np.exp(1j * np.angle(fa))).real
    b_eq = np.fft.ifftn(amp * np.exp(1j * np.angle(fb))).real
    return np.asarray(a_eq, dtype=np.float64), np.asarray(b_eq, dtype=np.float64)


def derivative_channels_3d(volume, sigma: float = 1.0):
    """Base + directional 1st/2nd derivatives for a (z, y, x) volume."""
    v = np.asarray(volume, dtype=np.float64)
    s = max(float(sigma), 0.0)
    base = gaussian_filter(v, sigma=s, mode="nearest") if s > 0.0 else v
    return {
        "base": np.asarray(base, dtype=np.float64),
        "dx": gaussian_filter(v, sigma=s, order=(0, 0, 1), mode="nearest"),
        "dy": gaussian_filter(v, sigma=s, order=(0, 1, 0), mode="nearest"),
        "dz": gaussian_filter(v, sigma=s, order=(1, 0, 0), mode="nearest"),
        "dxx": gaussian_filter(v, sigma=s, order=(0, 0, 2), mode="nearest"),
        "dyy": gaussian_filter(v, sigma=s, order=(0, 2, 0), mode="nearest"),
        "dzz": gaussian_filter(v, sigma=s, order=(2, 0, 0), mode="nearest"),
    }


def sci_score_3d(exp_map, sim_map, *, use_amp_eq: bool = True, sigma: float = 1.0,
                 w0: float = 1.0, w1: float = 1.0, w2: float = 1.0,
                 eps: float = 1e-8, mask=None):
    """3D SCI score: log-domain fusion of base/1st/2nd derivative channel CCs."""
    e = np.asarray(exp_map, dtype=np.float64)
    r = np.asarray(sim_map, dtype=np.float64)
    if e.shape != r.shape:
        raise ValueError(f"Map shape mismatch: {e.shape} vs {r.shape}")

    if use_amp_eq:
        e, r = amplitude_equalize_pair(e, r)

    m = _safe_mask(mask, e, r)
    e_ch = derivative_channels_3d(e, sigma=sigma)
    r_ch = derivative_channels_3d(r, sigma=sigma)

    cc0 = truncated_cc(e_ch["base"], r_ch["base"], m)
    ccx = truncated_cc(e_ch["dx"], r_ch["dx"], m)
    ccy = truncated_cc(e_ch["dy"], r_ch["dy"], m)
    ccz = truncated_cc(e_ch["dz"], r_ch["dz"], m)
    ccxx = truncated_cc(e_ch["dxx"], r_ch["dxx"], m)
    ccyy = truncated_cc(e_ch["dyy"], r_ch["dyy"], m)
    cczz = truncated_cc(e_ch["dzz"], r_ch["dzz"], m)

    first_log_mean = float(np.mean([np.log(eps + ccx), np.log(eps + ccy), np.log(eps + ccz)]))
    second_log_mean = float(np.mean([np.log(eps + ccxx), np.log(eps + ccyy), np.log(eps + cczz)]))

    sci_log = (float(w0) * np.log(eps + cc0)) + (float(w1) * first_log_mean) + (float(w2) * second_log_mean)
    sci = float(np.exp(sci_log))
    if not np.isfinite(sci):
        sci = 0.0

    details = {"cc0": float(cc0), "ccx": float(ccx), "ccy": float(ccy), "ccz": float(ccz),
               "ccxx": float(ccxx), "ccyy": float(ccyy), "cczz": float(cczz), "sci": float(sci)}
    return sci, details


# ---------------------------------------------------------------------------
# MI
# ---------------------------------------------------------------------------

def mutual_information_score(map_a, map_b, n_bins: int = 64, mask=None,
                             nonzero_union: bool = True, zscore: bool = False,
                             normalized: bool = False, eps: float = 1e-12) -> float:
    """Histogram-based mutual information between two same-grid maps."""
    a = np.asarray(map_a.density_map if hasattr(map_a, "density_map") else map_a, dtype=np.float64)
    b = np.asarray(map_b.density_map if hasattr(map_b, "density_map") else map_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Map shape mismatch: {a.shape} vs {b.shape}")

    if mask is None:
        use_mask = np.ones(a.shape, dtype=bool)
        if nonzero_union:
            use_mask &= ((a != 0.0) | (b != 0.0))
    else:
        use_mask = np.asarray(mask, dtype=bool)
        if use_mask.shape != a.shape:
            raise ValueError(f"Mask shape mismatch: {use_mask.shape} vs {a.shape}")

    av = a[use_mask].ravel()
    bv = b[use_mask].ravel()
    if av.size == 0:
        return 0.0

    if zscore:
        av = _zscore_array(av, eps=eps)
        bv = _zscore_array(bv, eps=eps)

    joint_hist, _, _ = np.histogram2d(av, bv, bins=int(n_bins))
    total = float(joint_hist.sum())
    if total <= 0.0:
        return 0.0

    pxy = joint_hist / total
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]

    nz = pxy > 0.0
    mi = float(np.sum(pxy[nz] * np.log((pxy[nz] + eps) / (px_py[nz] + eps))))

    if not normalized:
        return mi

    px_nz = px > 0.0
    py_nz = py > 0.0
    hx = -float(np.sum(px[px_nz] * np.log(px[px_nz] + eps)))
    hy = -float(np.sum(py[py_nz] * np.log(py[py_nz] + eps)))
    denom = np.sqrt(max(hx * hy, eps))
    return float(mi / denom)


# ---------------------------------------------------------------------------
# Q-score
# ---------------------------------------------------------------------------

class MapGrid:
    """Trilinear-sampling adapter around a (z, y, x) density grid."""

    def __init__(self, data, origin_xyz, apix_xyz):
        self.data = np.asarray(data, dtype=np.float32)
        self.origin_xyz = np.asarray(origin_xyz, dtype=float).reshape(3)
        self.apix_xyz = np.asarray(apix_xyz, dtype=float).reshape(3)

    @staticmethod
    def from_emmap(emmap) -> "MapGrid":
        data = np.asarray(emmap.density_map, dtype=np.float32)
        if data.ndim != 3:
            raise ValueError(f"density_map must be 3D, got shape {data.shape}")
        return MapGrid(data, _as_xyz(emmap.origin), _as_xyz(emmap.apix))

    def stats(self):
        return float(np.mean(self.data)), float(np.std(self.data))

    def sample_trilinear(self, xyz):
        xyz2 = np.atleast_2d(np.asarray(xyz, dtype=float))
        if xyz2.shape[1] != 3:
            raise ValueError(f"xyz must be (...,3), got {xyz2.shape}")

        orig = self.origin_xyz.reshape(3)
        apix = self.apix_xyz.reshape(3)
        f = (xyz2 - orig[None, :]) / apix[None, :]
        x, y, z = f[:, 0], f[:, 1], f[:, 2]

        nz, ny, nx = self.data.shape
        x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int); z0 = np.floor(z).astype(int)
        x1 = x0 + 1; y1 = y0 + 1; z1 = z0 + 1

        oob = ((x0 < 0) | (y0 < 0) | (z0 < 0) | (x1 >= nx) | (y1 >= ny) | (z1 >= nz))

        # Clamp so the gather never indexes out of bounds; OOB points are NaN-ed below.
        x0c = np.clip(x0, 0, nx - 1); x1c = np.clip(x1, 0, nx - 1)
        y0c = np.clip(y0, 0, ny - 1); y1c = np.clip(y1, 0, ny - 1)
        z0c = np.clip(z0, 0, nz - 1); z1c = np.clip(z1, 0, nz - 1)

        xd = (x - x0).astype(float); yd = (y - y0).astype(float); zd = (z - z0).astype(float)

        c000 = self.data[z0c, y0c, x0c]; c100 = self.data[z0c, y0c, x1c]
        c010 = self.data[z0c, y1c, x0c]; c110 = self.data[z0c, y1c, x1c]
        c001 = self.data[z1c, y0c, x0c]; c101 = self.data[z1c, y0c, x1c]
        c011 = self.data[z1c, y1c, x0c]; c111 = self.data[z1c, y1c, x1c]

        c00 = c000 * (1 - xd) + c100 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c11 = c011 * (1 - xd) + c111 * xd
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        val = (c0 * (1 - zd) + c1 * zd).astype(np.float32)

        if np.any(oob):
            val = val.copy()
            val[oob] = np.nan
        return val


def fibonacci_sphere(n: int):
    i = np.arange(n, dtype=float)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    theta = 2.0 * np.pi * i / phi
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1).astype(np.float32)


def greedy_farthest_point(dirs, k: int):
    dirs = np.asarray(dirs, dtype=float)
    if dirs.shape[0] == 0:
        return np.empty((0, 3), dtype=float)
    k = min(k, dirs.shape[0])
    chosen = [0]
    min_d = 1.0 - np.clip(dirs @ dirs[0], -1.0, 1.0)
    for _ in range(1, k):
        idx = int(np.argmax(min_d))
        chosen.append(idx)
        d_new = 1.0 - np.clip(dirs @ dirs[idx], -1.0, 1.0)
        min_d = np.minimum(min_d, d_new)
    return dirs[np.array(chosen, dtype=int)]


def reference_gaussian_values(radii, *, map_mean: float, map_std: float,
                              sigma_ref: float = 0.6, mu: float = 0.0):
    A = map_mean + 10.0 * map_std
    B = map_mean - 1.0 * map_std
    x = np.asarray(radii, dtype=float)
    return (A * np.exp(-0.5 * ((x - mu) / sigma_ref) ** 2) + B).astype(np.float32)


def qscore_from_uv(u, v) -> float:
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    mask = np.isfinite(u) & np.isfinite(v)
    u = u[mask]; v = v[mask]
    if u.size < 2:
        return float("nan")
    u0 = u - np.mean(u); v0 = v - np.mean(v)
    nu = np.linalg.norm(u0); nv = np.linalg.norm(v0)
    if nu == 0.0 or nv == 0.0:
        return float("nan")
    return float(np.dot(u0, v0) / (nu * nv))


def compute_atom_qscore(atom_index, atom_xyz, *, kdtree, mapgrid, radii, dirs_unit,
                        n_points_per_shell: int = 8, sigma_ref: float = 0.6,
                        map_mean: float, map_std: float) -> float:
    atom_xyz = np.asarray(atom_xyz, dtype=float).reshape(3)
    v_r = reference_gaussian_values(radii, map_mean=map_mean, map_std=map_std, sigma_ref=sigma_ref)

    u_vals = []
    v_vals = []
    for i, r in enumerate(radii):
        if np.isclose(r, 0.0):
            u0 = float(mapgrid.sample_trilinear(atom_xyz)[0])
            u_vals.extend([u0] * n_points_per_shell)
            v_vals.extend([float(v_r[i])] * n_points_per_shell)
            continue

        cand_pts = atom_xyz[None, :] + r * dirs_unit
        nn_idx = kdtree.query(cand_pts, k=1)[1]
        allowed = dirs_unit[nn_idx == atom_index]
        if allowed.shape[0] == 0:
            allowed = dirs_unit
        chosen_dirs = greedy_farthest_point(allowed, n_points_per_shell)
        if chosen_dirs.shape[0] < n_points_per_shell:
            reps = n_points_per_shell - chosen_dirs.shape[0]
            chosen_dirs = np.vstack([chosen_dirs, np.tile(chosen_dirs[:1], (reps, 1))])

        pts = atom_xyz[None, :] + r * chosen_dirs
        u_shell = mapgrid.sample_trilinear(pts)
        u_vals.extend([float(x) for x in u_shell])
        v_vals.extend([float(v_r[i])] * n_points_per_shell)

    return qscore_from_uv(np.array(u_vals), np.array(v_vals))


def compute_qscores_from_emmap(*, atoms_xyz, emmap, sigma_ref: float = 0.6, radii=None,
                               n_points_per_shell: int = 8, candidate_dirs: int = 256,
                               score_indices=None):
    """Per-atom Q-scores. The neighbour tree is built from all ``atoms_xyz`` so
    that nearby (context) atoms occlude shells, but only ``score_indices`` are
    scored and returned."""
    if radii is None:
        radii = np.round(np.arange(0.0, 2.0 + 1e-6, 0.1), 3)

    atoms_xyz = np.asarray(atoms_xyz, dtype=float)
    if atoms_xyz.ndim != 2 or atoms_xyz.shape[1] != 3:
        raise ValueError(f"atoms_xyz must be (N,3), got {atoms_xyz.shape}")

    mapgrid = MapGrid.from_emmap(emmap)
    map_mean, map_std = mapgrid.stats()
    kdtree = cKDTree(atoms_xyz)
    dirs_unit = fibonacci_sphere(candidate_dirs)

    if score_indices is None:
        score_idx = np.arange(atoms_xyz.shape[0], dtype=int)
    else:
        score_idx = np.asarray(score_indices, dtype=int).reshape(-1)
        if np.any(score_idx < 0) or np.any(score_idx >= atoms_xyz.shape[0]):
            raise ValueError("score_indices contains out-of-range atom indices")

    out = np.empty((score_idx.shape[0],), dtype=np.float32)
    for row, i in enumerate(score_idx.tolist()):
        out[row] = compute_atom_qscore(
            int(i), atoms_xyz[int(i)], kdtree=kdtree, mapgrid=mapgrid, radii=radii,
            dirs_unit=dirs_unit, n_points_per_shell=n_points_per_shell,
            sigma_ref=sigma_ref, map_mean=map_mean, map_std=map_std)
    return out


# ---------------------------------------------------------------------------
# High-level drivers (the GUI calls these)
# ---------------------------------------------------------------------------

DENSITY_METRICS = ("ccc", "mi", "sci")


def score_map_metrics(coords_A, masses, emmap, resolution, which=DENSITY_METRICS,
                      *, sigma_coeff: float = 0.356, normalise: bool = True,
                      n_bins: int = 64, sci_sigma: float = 1.0,
                      use_amp_eq: bool = True):
    """Compute CCC / MI / SCI for one pose against ``emmap``.

    Mirrors the backend ``score_density_metrics``: blur the ligand onto the map
    grid over its support box, crop the experimental map to the same box, then
    correlate. Returns a dict with any of ``ccc``/``mi``/``sci`` in ``which``.
    """
    which = tuple(w.lower() for w in which)
    requested = [w for w in which if w in DENSITY_METRICS]
    if not requested:
        return {}

    density = np.asarray(emmap.density_map, dtype=np.float64)
    sim_sub, lo, hi = simulate_ligand_density_subgrid(
        coords_A, masses, emmap.origin, emmap.apix, density.shape,
        resolution_A=resolution, sigma_coeff=sigma_coeff, normalise=normalise)

    if sim_sub.size == 0:
        return {w: 0.0 for w in requested}

    z0, y0, x0 = [int(i) for i in lo]
    z1, y1, x1 = [int(i) for i in hi]
    exp_sub = np.asarray(density[z0:z1, y0:y1, x0:x1], dtype=np.float64)
    if exp_sub.shape != sim_sub.shape:
        return {w: 0.0 for w in requested}

    results = {}
    if "ccc" in requested:
        results["ccc"] = float(truncated_cc(exp_sub, sim_sub, None))
    if "mi" in requested:
        results["mi"] = float(mutual_information_score(
            exp_sub, sim_sub, n_bins=n_bins, nonzero_union=True, normalized=False))
    if "sci" in requested:
        sci, _ = sci_score_3d(exp_sub, sim_sub, use_amp_eq=use_amp_eq, sigma=sci_sigma)
        results["sci"] = float(sci)
    return results


def score_qscore(coords_A, emmap, *, context_coords=None, sigma_ref: float = 0.6,
                 candidate_dirs: int = 128):
    """Per-atom Q-score over ``coords_A`` (mean / min reported).

    ``context_coords`` (e.g. nearby protein heavy atoms) are added to the
    neighbour tree so buried shells are correctly occluded, but are not scored.
    """
    coords_A = np.asarray(coords_A, dtype=float).reshape(-1, 3)
    if coords_A.shape[0] == 0:
        return {"qscore_mean": 0.0, "qscore_min": 0.0, "qscore_per_atom": []}

    if context_coords is not None and len(context_coords):
        all_coords = np.vstack([coords_A, np.asarray(context_coords, dtype=float).reshape(-1, 3)])
    else:
        all_coords = coords_A
    score_indices = np.arange(coords_A.shape[0], dtype=int)

    q = compute_qscores_from_emmap(
        atoms_xyz=all_coords, emmap=emmap, sigma_ref=sigma_ref,
        candidate_dirs=candidate_dirs, score_indices=score_indices)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    finite = q[np.isfinite(q)]
    return {
        "qscore_mean": float(np.mean(finite)) if finite.size else 0.0,
        "qscore_min": float(np.min(finite)) if finite.size else 0.0,
        "qscore_per_atom": [float(v) for v in q],
    }


# ---------------------------------------------------------------------------
# MMGBSA (heavy; OpenMM/ParmEd/mdtraj imported lazily)
# ---------------------------------------------------------------------------

def _split_nonbonded_terms(system):
    """Split the NonbondedForce into Coulomb (group 1) and LJ (group 2)."""
    from openmm import NonbondedForce, CustomNonbondedForce, unit

    nb = nb_i = None
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if isinstance(f, NonbondedForce):
            nb, nb_i = f, i
            break
    if nb is None:
        raise RuntimeError("System has no NonbondedForce")

    coul_force = NonbondedForce()
    coul_force.setCutoffDistance(nb.getCutoffDistance())
    coul_force.setNonbondedMethod(nb.getNonbondedMethod())
    coul_force.setReactionFieldDielectric(nb.getReactionFieldDielectric())
    coul_force.setEwaldErrorTolerance(nb.getEwaldErrorTolerance())
    coul_force.setUseDispersionCorrection(False)
    zero_eps = 0.0 * unit.kilojoule_per_mole

    lj_force = CustomNonbondedForce("4*epsilon*((sigma/r)^12-(sigma/r)^6);"
                                    "sigma = 0.5*(sigma1+sigma2);"
                                    "epsilon = sqrt(epsilon1*epsilon2)")
    lj_force.addPerParticleParameter("sigma")
    lj_force.addPerParticleParameter("epsilon")
    lj_force.setCutoffDistance(nb.getCutoffDistance())
    lj_force.setNonbondedMethod(nb.getNonbondedMethod())

    for idx in range(nb.getNumParticles()):
        charge, sigma, eps = nb.getParticleParameters(idx)
        coul_force.addParticle(charge, sigma, zero_eps)
        lj_force.addParticle([sigma, eps])

    for i in range(nb.getNumExceptions()):
        i1, i2, q, sig, eps = nb.getExceptionParameters(i)
        coul_force.addException(i1, i2, q, sig, zero_eps)
        lj_force.addExclusion(i1, i2)

    system.removeForce(nb_i)
    coul_force.setForceGroup(1)
    lj_force.setForceGroup(2)
    system.addForce(coul_force)
    system.addForce(lj_force)
    return system


def _modify_system(system):
    from openmm import CustomGBForce, GBSAOBCForce
    for f in system.getForces():
        if isinstance(f, (CustomGBForce, GBSAOBCForce)):
            f.setForceGroup(3)
    return system


def _make_system_triplet(complex_struct):
    from openmm.app import HBonds, OBC2
    from openmm import unit

    rec_idx, lig_idx = [], []
    for atom in complex_struct.atoms:
        if atom.residue.name.startswith("LIG"):
            lig_idx.append(atom.idx)
        else:
            rec_idx.append(atom.idx)

    receptor_struct = complex_struct[rec_idx]
    ligand_struct = complex_struct[lig_idx]

    common = dict(nonbondedCutoff=1.2 * unit.nanometers, constraints=HBonds,
                  removeCMMotion=False, implicitSolvent=OBC2)
    rec_sys = _modify_system(_split_nonbonded_terms(receptor_struct.createSystem(**common)))
    lig_sys = _modify_system(_split_nonbonded_terms(ligand_struct.createSystem(**common)))
    cmp_sys = _modify_system(_split_nonbonded_terms(complex_struct.createSystem(**common)))
    return rec_sys, lig_sys, cmp_sys, rec_idx, lig_idx


def _compute_frame_energies(system, positions, frame, gamma, beta, platform_name=None):
    import mdtraj as md
    from openmm import VerletIntegrator, unit, Platform, Context

    intg = VerletIntegrator(0.002 * unit.picoseconds)
    if platform_name is not None:
        try:
            ctx = Context(system, intg, Platform.getPlatformByName(str(platform_name)))
        except Exception:
            ctx = Context(system, intg)
    else:
        ctx = Context(system, intg)

    try:
        ctx.setPositions(positions)
        EEL = ctx.getState(getEnergy=True, groups={1}).getPotentialEnergy()
        VDW = ctx.getState(getEnergy=True, groups={2}).getPotentialEnergy()
        EGB = ctx.getState(getEnergy=True, groups={3}).getPotentialEnergy()
        sasa = md.shrake_rupley(frame)[0].sum() * 100.0  # nm^2 -> Å^2
        ECAV = gamma * sasa + beta
        conv = unit.kilocalories_per_mole
        return (EEL.value_in_unit(conv), VDW.value_in_unit(conv),
                EGB.value_in_unit(conv), ECAV)
    finally:
        del ctx


def _parmed_to_single_frame_traj(struct):
    import mdtraj as md
    from openmm import unit

    pos = getattr(struct, "positions", None)
    if pos is None:
        raise ValueError("ParmEd Structure has no positions set.")
    xyz_nm = np.asarray(pos.value_in_unit(unit.nanometer), dtype=float)
    return md.Trajectory(xyz=xyz_nm[None, :, :],
                         topology=md.Topology.from_openmm(struct.topology))


def compute_mmgbsa(complex_struct, *, platform_name=None, gamma: float = 0.005,
                   beta: float = 0.0):
    """ΔG (MMGBSA, OBC2 implicit solvent) for a parametrised protein+ligand
    ParmEd ``complex_struct`` whose ligand residues are named ``LIG*``.

    Returns ``(components, deltaG)`` with components
    ``{"EEL", "VDW", "EGB", "ECAV"}`` in kcal/mol and ``deltaG`` their sum.
    """
    from openmm import unit

    traj = _parmed_to_single_frame_traj(complex_struct)
    rec_sys, lig_sys, cmp_sys, rec_idx, lig_idx = _make_system_triplet(complex_struct)
    if not lig_idx:
        raise RuntimeError("No ligand (LIG*) residue found in the complex structure.")

    frame = traj[0]
    pos_nm = frame.xyz[0] * unit.nanometer
    rec_pos = pos_nm[rec_idx]
    lig_pos = pos_nm[lig_idx]

    # NOTE: pass the SAME full-complex `frame` to all three energy calls. The
    # per-subsystem OpenMM energies use the sliced positions (rec_pos/lig_pos),
    # but the SASA term reads `frame` exactly as the ChemEM backend does, so the
    # GUI's ECAV/deltaG match the pipeline's MMGBSA numbers (parity over a
    # "corrected" per-subsystem SASA).
    eel_c, vdw_c, egb_c, ecav_c = _compute_frame_energies(cmp_sys, pos_nm, frame, gamma, beta, platform_name)
    eel_r, vdw_r, egb_r, ecav_r = _compute_frame_energies(rec_sys, rec_pos, frame, gamma, beta, platform_name)
    eel_l, vdw_l, egb_l, ecav_l = _compute_frame_energies(lig_sys, lig_pos, frame, gamma, beta, platform_name)

    components = {
        "EEL": float(eel_c - (eel_r + eel_l)),
        "VDW": float(vdw_c - (vdw_r + vdw_l)),
        "EGB": float(egb_c - (egb_r + egb_l)),
        "ECAV": float(ecav_c - (ecav_r + ecav_l)),
    }
    deltaG = float(sum(components.values()))
    return components, deltaG
