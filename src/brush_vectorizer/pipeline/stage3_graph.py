"""Stage 3: Graph Construction & Momentum Junction Untangling."""

from __future__ import annotations

from dataclasses import dataclass
import math
import networkx as nx
import numpy as np
from scipy import ndimage

from brush_vectorizer.config import PhysicalCalibrationConfig
from brush_vectorizer.models import PixelPath


@dataclass
class Stage3Result:
    """Outputs of Stage 3 Graph Construction & Momentum Junction Untangling."""

    raw_graph: nx.MultiGraph
    collapsed_graph: nx.MultiGraph
    collapsed_bridge_count: int
    untangled_paths: list[PixelPath]


def _compute_8_degree(skel: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)
    counts = ndimage.convolve(skel.astype(np.int32), kernel, mode="constant", cval=0)
    return np.where(skel, counts, 0)


def _polyline_length(coords_xy: np.ndarray) -> float:
    if len(coords_xy) < 2:
        return 0.0
    diffs = np.diff(coords_xy, axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))


def build_skeleton_graph(
    skeleton: np.ndarray,
    edt_radius_px: np.ndarray,
) -> nx.MultiGraph:
    """Convert 1-pixel binary skeleton K into an undirected MultiGraph G = (V, E).

    Nodes V are endpoints (deg == 1), junction clusters (deg >= 3), or loop anchors.
    Edges E store ordered arrays of pixel coordinates (x_i, y_i) = (col, row).
    """
    G = nx.MultiGraph()
    if not np.any(skeleton):
        return G

    h, w = skeleton.shape
    deg_map = _compute_8_degree(skeleton)

    # Identify node pixels: deg == 1 (endpoints) or deg >= 3 (junctions)
    node_mask = skeleton & ((deg_map == 1) | (deg_map >= 3))

    # Label 8-connected components of node_mask so adjacent junction pixels form a single node
    struct_8 = np.ones((3, 3), dtype=bool)
    node_labels, num_nodes = ndimage.label(node_mask, structure=struct_8)

    # Also check for pure closed loops (connected skeleton components with NO node_mask pixels)
    skel_labels, num_comp = ndimage.label(skeleton, structure=struct_8)
    if num_comp > 0:
        flat_skel = skel_labels.ravel()
        node_counts_per_comp = np.bincount(
            flat_skel, weights=node_mask.ravel().astype(np.float64), minlength=num_comp + 1
        )
        loop_comp_ids = np.where(node_counts_per_comp[1 : num_comp + 1] == 0)[0] + 1
        if len(loop_comp_ids) > 0:
            # Find first pixel of each loop component in one pass
            unique_comps, first_flat_indices = np.unique(flat_skel, return_index=True)
            comp_to_first_idx = dict(zip(unique_comps.tolist(), first_flat_indices.tolist()))
            for comp_id in loop_comp_ids:
                flat_idx = comp_to_first_idx.get(int(comp_id))
                if flat_idx is not None:
                    r0, c0 = divmod(int(flat_idx), w)
                    num_nodes += 1
                    node_labels[r0, c0] = num_nodes
                    node_mask[r0, c0] = True

    # Extract coordinates for all nodes in a single pass instead of scanning HxW per node
    node_rs, node_cs = np.nonzero(node_labels)
    if len(node_rs) > 0:
        labels_at_nz = node_labels[node_rs, node_cs]
        order = np.argsort(labels_at_nz, kind="mergesort")
        sorted_rs = node_rs[order]
        sorted_cs = node_cs[order]
        sorted_lbls = labels_at_nz[order]
        split_pts = np.searchsorted(sorted_lbls, np.arange(1, num_nodes + 2))

        for nid in range(1, num_nodes + 1):
            i0, i1 = int(split_pts[nid - 1]), int(split_pts[nid])
            if i0 >= i1:
                continue
            rs = sorted_rs[i0:i1]
            cs = sorted_cs[i0:i1]
            mean_r = float(np.mean(rs))
            mean_c = float(np.mean(cs))
            max_rad = float(np.max(edt_radius_px[rs, cs]))
            is_junction = bool(np.any(deg_map[rs, cs] >= 3)) or (i1 - i0) > 1
            G.add_node(
                nid,
                x_px=mean_c,
                y_px=mean_r,
                r_px=max_rad,
                is_junction=is_junction,
                pixels=[(int(c), int(r)) for r, c in zip(rs, cs)],
            )
    # Trace edges along degree-2 chains between node clusters
    visited_chain_pixels = np.zeros_like(skeleton, dtype=bool)
    visited_direct_pairs: set[tuple[int, int, int, int]] = set()

    for nr, nc in zip(node_rs, node_cs):
        u_id = int(node_labels[nr, nc])
        u_xy = (float(G.nodes[u_id]["x_px"]), float(G.nodes[u_id]["y_px"]))

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(nr + dr), int(nc + dc)
                if not (0 <= rr < h and 0 <= cc < w and skeleton[rr, cc]):
                    continue

                nb_label = int(node_labels[rr, cc])
                if nb_label > 0:
                    # Direct adjacency between two different node clusters
                    if nb_label != u_id:
                        pair_key = (
                            min(u_id, nb_label),
                            max(u_id, nb_label),
                            min(int(nr) * w + int(nc), rr * w + cc),
                            max(int(nr) * w + int(nc), rr * w + cc),
                        )
                        if pair_key not in visited_direct_pairs:
                            visited_direct_pairs.add(pair_key)
                            v_xy = (
                                float(G.nodes[nb_label]["x_px"]),
                                float(G.nodes[nb_label]["y_px"]),
                            )
                            coords = np.array([u_xy, v_xy], dtype=np.float64)
                            G.add_edge(
                                u_id,
                                nb_label,
                                coords_xy=coords,
                                length_px=_polyline_length(coords),
                            )
                    continue

                if visited_chain_pixels[rr, cc]:
                    continue

                # Trace along degree-2 pixel chain until reaching another node cluster
                chain_xy: list[tuple[float, float]] = [u_xy, (float(cc), float(rr))]
                visited_chain_pixels[rr, cc] = True
                prev_rc = (int(nr), int(nc))
                curr_rc = (rr, cc)
                end_node_id: int | None = None

                while True:
                    cr, cc_curr = curr_rc
                    next_candidates: list[tuple[int, int]] = []
                    adj_node_candidates: list[tuple[int, int, int]] = []

                    for ddr in (-1, 0, 1):
                        for ddc in (-1, 0, 1):
                            if ddr == 0 and ddc == 0:
                                continue
                            tr, tc = cr + ddr, cc_curr + ddc
                            if not (0 <= tr < h and 0 <= tc < w and skeleton[tr, tc]):
                                continue
                            if (tr, tc) == prev_rc:
                                continue
                            lbl = int(node_labels[tr, tc])
                            if lbl > 0:
                                adj_node_candidates.append((lbl, tr, tc))
                            elif not visited_chain_pixels[tr, tc]:
                                next_candidates.append((tr, tc))

                    if adj_node_candidates:
                        # Prefer reaching a node different from u_id if available
                        other_nodes = [c for c in adj_node_candidates if c[0] != u_id]
                        chosen = other_nodes[0] if other_nodes else adj_node_candidates[0]
                        end_node_id = chosen[0]
                        v_xy = (
                            float(G.nodes[end_node_id]["x_px"]),
                            float(G.nodes[end_node_id]["y_px"]),
                        )
                        chain_xy.append(v_xy)
                        break

                    if not next_candidates:
                        break

                    nxt_rc = next_candidates[0]
                    visited_chain_pixels[nxt_rc[0], nxt_rc[1]] = True
                    chain_xy.append((float(nxt_rc[1]), float(nxt_rc[0])))
                    prev_rc = curr_rc
                    curr_rc = nxt_rc

                if end_node_id is not None:
                    coords = np.array(chain_xy, dtype=np.float64)
                    G.add_edge(
                        u_id,
                        end_node_id,
                        coords_xy=coords,
                        length_px=_polyline_length(coords),
                    )

    return G


def collapse_short_bridges(
    graph: nx.MultiGraph,
    bridge_collapse_factor: float = 2.0,
) -> tuple[nx.MultiGraph, int]:
    """Collapse any edge connecting two junction nodes if length < 2 * R(x_J, y_J)."""
    G = graph.copy()
    collapsed_count = 0

    while True:
        bridge_to_collapse: tuple[int, int, int, float] | None = None

        for u, v, k, data in G.edges(keys=True, data=True):
            if u == v:
                continue
            deg_u = G.degree(u)
            deg_v = G.degree(v)
            if deg_u < 3 or deg_v < 3:
                continue

            r_u = float(G.nodes[u]["r_px"])
            r_v = float(G.nodes[v]["r_px"])
            # Cap junction radius for bridge collapse so broad wash blobs do not collapse into a starburst
            r_j = min(max(r_u, r_v), 10.0)
            length = float(data.get("length_px", 0.0))
            # Effective bridge span accounting for the junction radius zone
            effective_span = max(0.0, length - 0.5 * min(r_u, r_v, 10.0))

            if effective_span < bridge_collapse_factor * r_j:
                if bridge_to_collapse is None or length < bridge_to_collapse[3]:
                    bridge_to_collapse = (u, v, k, length)

        if bridge_to_collapse is None:
            break

        u, v, k, _len = bridge_to_collapse
        # Merge v into u at their average position
        ux, uy = float(G.nodes[u]["x_px"]), float(G.nodes[u]["y_px"])
        vx, vy = float(G.nodes[v]["x_px"]), float(G.nodes[v]["y_px"])
        mx, my = 0.5 * (ux + vx), 0.5 * (uy + vy)
        mr = max(float(G.nodes[u]["r_px"]), float(G.nodes[v]["r_px"]))

        G.nodes[u]["x_px"] = mx
        G.nodes[u]["y_px"] = my
        G.nodes[u]["r_px"] = mr
        G.nodes[u]["is_junction"] = True

        # Remove the bridge edge (and any parallel short edges between u and v)
        G.remove_edge(u, v, key=k)

        # Rewire remaining edges incident to u or v so their endpoint sits at (mx, my)
        for node_id in (u, v):
            incident = list(G.edges(node_id, keys=True, data=True))
            for n1, n2, ek, edata in incident:
                other = n2 if n1 == node_id else n1
                coords = edata["coords_xy"].copy()
                # Check which end of coords corresponds to node_id
                orig_x = ux if node_id == u else vx
                orig_y = uy if node_id == u else vy
                d0 = math.hypot(coords[0, 0] - orig_x, coords[0, 1] - orig_y)
                d1 = math.hypot(coords[-1, 0] - orig_x, coords[-1, 1] - orig_y)
                if d0 <= d1:
                    coords[0] = [mx, my]
                else:
                    coords[-1] = [mx, my]

                new_len = _polyline_length(coords)
                if node_id == v:
                    G.remove_edge(n1, n2, key=ek)
                    if other != u:
                        G.add_edge(u, other, coords_xy=coords, length_px=new_len)
                else:
                    G[n1][n2][ek]["coords_xy"] = coords
                    G[n1][n2][ek]["length_px"] = new_len

        G.remove_node(v)
        collapsed_count += 1

    return G, collapsed_count


def _orient_edge_from_node(
    coords_xy: np.ndarray, node_x: float, node_y: float
) -> np.ndarray:
    """Return coordinate array oriented so index 0 is at (node_x, node_y) and moves outward."""
    d_start = math.hypot(coords_xy[0, 0] - node_x, coords_xy[0, 1] - node_y)
    d_end = math.hypot(coords_xy[-1, 0] - node_x, coords_xy[-1, 1] - node_y)
    if d_start <= d_end:
        return coords_xy.copy()
    return coords_xy[::-1].copy()


def _sample_field_bilinear(field: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Sample 2D array (H, W) at floating-point pixel coordinates (x=col, y=row)."""
    h, w = field.shape
    cols = np.clip(np.round(xs).astype(int), 0, w - 1)
    rows = np.clip(np.round(ys).astype(int), 0, h - 1)
    return field[rows, cols].astype(np.float64)


def compute_edge_junction_features(
    oriented_coords_xy: np.ndarray,
    r_junction: float,
    edt_radius_px: np.ndarray,
    density_map: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Compute unit outward tangent vector T_a, local radius r_a, and optical density D_a.

    Per Stage 3.3: Fit a linear regression line over the pixel coordinates sitting
    just outside the junction's radius zone ([R_J, 3 * R_J] pixels back from v_J).
    The tangent vector is oriented outward from v_J along the edge.
    """
    n_pts = len(oriented_coords_xy)
    if n_pts < 2:
        return np.array([1.0, 0.0], dtype=np.float64), max(r_junction, 1.0), 0.5

    diffs = np.diff(oriented_coords_xy, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = float(cum_dist[-1])

    # Select points in the [R_J, 3 * R_J] zone from the junction
    r_min = min(r_junction, 0.35 * total_len)
    r_max = max(3.0 * r_junction, r_min + 2.0)
    zone_mask = (cum_dist >= r_min) & (cum_dist <= r_max)

    if np.count_nonzero(zone_mask) >= 2:
        pts = oriented_coords_xy[zone_mask]
        dists = cum_dist[zone_mask]
    else:
        pts = oriented_coords_xy
        dists = cum_dist

    # Linear regression X(s) = x0 + vx * s, Y(s) = y0 + vy * s wrt arc-length s
    s_mean = float(np.mean(dists))
    ds = dists - s_mean
    denom = float(np.sum(ds * ds))
    if denom > 1e-8:
        vx = float(np.sum(ds * (pts[:, 0] - np.mean(pts[:, 0])))) / denom
        vy = float(np.sum(ds * (pts[:, 1] - np.mean(pts[:, 1])))) / denom
    else:
        vx = float(pts[-1, 0] - pts[0, 0])
        vy = float(pts[-1, 1] - pts[0, 1])

    norm = math.hypot(vx, vy)
    if norm < 1e-8:
        tangent = np.array([1.0, 0.0], dtype=np.float64)
    else:
        tangent = np.array([vx / norm, vy / norm], dtype=np.float64)

    # Sample local radius r_a and optical density D_a in the [R_J, 3*R_J] zone
    radii = _sample_field_bilinear(edt_radius_px, pts[:, 0], pts[:, 1])
    densities = _sample_field_bilinear(density_map, pts[:, 0], pts[:, 1])
    r_a = float(np.mean(radii)) if len(radii) > 0 else max(r_junction, 1.0)
    d_a = float(np.mean(densities)) if len(densities) > 0 else 0.5

    return tangent, max(r_a, 0.5), d_a


def compute_continuity_cost(
    t_a: np.ndarray,
    r_a: float,
    d_a: float,
    t_b: np.ndarray,
    r_b: float,
    d_b: float,
    config: PhysicalCalibrationConfig,
) -> tuple[float, float]:
    """Compute pairing cost C(a, b) and deflection angle (in degrees) for edges meeting at v_J.

    T_a and T_b are unit vectors pointing outward from v_J along edges a and b.
    For a perfectly straight continuation across v_J, T_a and T_b point in opposite
    directions (T_a . T_b = -1.0, deflection = 0 deg, |T_a . T_b| = 1.0).
    """
    dot = float(np.clip(np.dot(t_a, t_b), -1.0, 1.0))
    # Deflection angle from a straight through-line (where T_a = -T_b)
    cos_deflect = float(np.clip(-dot, -1.0, 1.0))
    deflection_deg = math.degrees(math.acos(cos_deflect))

    # Stage 3.4 formula:
    # C(a, b) = w_theta * (1 - |T_a . T_b|) + w_r * |r_a - r_b| / max(r_a, r_b) + w_d * |D_a - D_b|
    angular_term = 1.0 - abs(dot)
    radius_term = abs(r_a - r_b) / max(r_a, r_b, 1e-6)
    density_term = abs(d_a - d_b)

    cost = (
        config.weight_theta * angular_term
        + config.weight_radius * radius_term
        + config.weight_density * density_term
    )
    return float(cost), float(deflection_deg)


def untangle_graph_paths(
    graph: nx.MultiGraph,
    edt_radius_px: np.ndarray,
    density_map: np.ndarray,
    config: PhysicalCalibrationConfig,
) -> list[PixelPath]:
    """Resolve junctions via continuity cost matrix and greedy path merging into polylines."""
    # Assign each edge a unique integer ID
    edge_records: dict[int, tuple[int, int, int, np.ndarray]] = {}
    node_incident_edges: dict[int, list[int]] = {n: [] for n in graph.nodes()}

    for eid, (u, v, k, data) in enumerate(graph.edges(keys=True, data=True)):
        coords = data["coords_xy"]
        edge_records[eid] = (u, v, k, coords)
        node_incident_edges[u].append(eid)
        if v != u:
            node_incident_edges[v].append(eid)

    # At every node with degree >= 2, decide which incident edges pair together
    # Maps (node_id, edge_id) -> paired_edge_id
    node_pairings: dict[tuple[int, int], int] = {}

    for nid, eids in node_incident_edges.items():
        if len(eids) < 2:
            continue

        nx_px = float(graph.nodes[nid]["x_px"])
        ny_px = float(graph.nodes[nid]["y_px"])
        r_j = float(graph.nodes[nid]["r_px"])

        if len(eids) == 2 and eids[0] != eids[1]:
            # Degree-2 node: pair if deflection < max_deflection_deg
            e1, e2 = eids[0], eids[1]
            c1 = _orient_edge_from_node(edge_records[e1][3], nx_px, ny_px)
            c2 = _orient_edge_from_node(edge_records[e2][3], nx_px, ny_px)
            t1, r1, d1 = compute_edge_junction_features(c1, r_j, edt_radius_px, density_map)
            t2, r2, d2 = compute_edge_junction_features(c2, r_j, edt_radius_px, density_map)
            _cost, deflect = compute_continuity_cost(t1, r1, d1, t2, r2, d2, config)
            if deflect < config.max_deflection_deg:
                node_pairings[(nid, e1)] = e2
                node_pairings[(nid, e2)] = e1
            continue

        # Junction node (degree >= 3): compute features for each incident edge
        features: dict[int, tuple[np.ndarray, float, float]] = {}
        for eid in eids:
            u, v, _k, coords = edge_records[eid]
            if u == v:
                continue
            oriented = _orient_edge_from_node(coords, nx_px, ny_px)
            features[eid] = compute_edge_junction_features(
                oriented, r_j, edt_radius_px, density_map
            )

        # Build candidate pairs with deflection < max_deflection_deg (65 deg)
        candidate_pairs: list[tuple[float, int, int]] = []
        valid_eids = list(features.keys())
        for i in range(len(valid_eids)):
            for j in range(i + 1, len(valid_eids)):
                ea, eb = valid_eids[i], valid_eids[j]
                ta, ra, da = features[ea]
                tb, rb, db = features[eb]
                cost, deflect = compute_continuity_cost(ta, ra, da, tb, rb, db, config)
                if deflect < config.max_deflection_deg:
                    candidate_pairs.append((cost, ea, eb))

        # Greedy Path Merging: lowest cost C(a, b) first
        candidate_pairs.sort(key=lambda item: item[0])
        paired_at_node: set[int] = set()
        for _cost, ea, eb in candidate_pairs:
            if ea not in paired_at_node and eb not in paired_at_node:
                paired_at_node.add(ea)
                paired_at_node.add(eb)
                node_pairings[(nid, ea)] = eb
                node_pairings[(nid, eb)] = ea

    # Now trace maximally merged chains of edges into continuous polylines
    visited_edges: set[int] = set()
    untangled_paths: list[PixelPath] = []

    def _get_other_node(eid: int, curr_node: int) -> int:
        u, v, _k, _c = edge_records[eid]
        return v if u == curr_node else u

    # First trace open chains starting at unpaired edge endpoints
    for eid in sorted(edge_records.keys()):
        if eid in visited_edges:
            continue
        u, v, _k, _coords = edge_records[eid]
        u_paired = (u, eid) in node_pairings
        v_paired = (v, eid) in node_pairings
        if u_paired and v_paired:
            continue  # Interior edge of a chain or cycle; will be reached from an endpoint

        start_node = u if not u_paired else v
        chain_edges: list[tuple[int, int]] = []
        curr_edge = eid
        curr_node = start_node

        while curr_edge not in visited_edges:
            visited_edges.add(curr_edge)
            next_node = _get_other_node(curr_edge, curr_node)
            chain_edges.append((curr_edge, curr_node))
            nxt_edge = node_pairings.get((next_node, curr_edge))
            if nxt_edge is None or nxt_edge in visited_edges:
                break
            curr_node = next_node
            curr_edge = nxt_edge

        path = _assemble_pixel_path(
            len(untangled_paths), chain_edges, edge_records, graph, edt_radius_px, density_map
        )
        if path is not None and path.arc_length_px >= config.min_stroke_length_px:
            untangled_paths.append(path)

    # Second, trace any remaining closed cycles
    for eid in sorted(edge_records.keys()):
        if eid in visited_edges:
            continue
        u, _v, _k, _coords = edge_records[eid]
        chain_edges = []
        curr_edge = eid
        curr_node = u

        while curr_edge not in visited_edges:
            visited_edges.add(curr_edge)
            next_node = _get_other_node(curr_edge, curr_node)
            chain_edges.append((curr_edge, curr_node))
            nxt_edge = node_pairings.get((next_node, curr_edge))
            if nxt_edge is None or nxt_edge in visited_edges:
                break
            curr_node = next_node
            curr_edge = nxt_edge

        path = _assemble_pixel_path(
            len(untangled_paths), chain_edges, edge_records, graph, edt_radius_px, density_map
        )
        if path is not None and path.arc_length_px >= config.min_stroke_length_px:
            untangled_paths.append(path)

    return untangled_paths


def _assemble_pixel_path(
    path_id: int,
    chain_edges: list[tuple[int, int]],
    edge_records: dict[int, tuple[int, int, int, np.ndarray]],
    graph: nx.MultiGraph,
    edt_radius_px: np.ndarray,
    density_map: np.ndarray,
) -> PixelPath | None:
    """Concatenate oriented edge coordinates and sample EDT radius and optical density."""
    if not chain_edges:
        return None

    segments: list[np.ndarray] = []
    for idx, (eid, start_node) in enumerate(chain_edges):
        nx_px = float(graph.nodes[start_node]["x_px"])
        ny_px = float(graph.nodes[start_node]["y_px"])
        oriented = _orient_edge_from_node(edge_records[eid][3], nx_px, ny_px)
        if idx > 0 and len(oriented) > 1:
            oriented = oriented[1:]
        segments.append(oriented)

    coords = np.vstack(segments)
    if len(coords) < 2:
        return None

    # Deduplicate consecutive identical points
    diffs = np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1]))
    keep = np.concatenate([[True], diffs > 1e-4])
    coords = coords[keep]
    if len(coords) < 2:
        return None

    radii_px = _sample_field_bilinear(edt_radius_px, coords[:, 0], coords[:, 1])
    densities = _sample_field_bilinear(density_map, coords[:, 0], coords[:, 1])
    return PixelPath(
        path_id=path_id,
        coords_xy_px=coords,
        radii_px=np.maximum(radii_px, 0.5),
        densities=np.clip(densities, 0.05, 1.0),
    )


def run_stage3_graph_untangling(
    pruned_skeleton: np.ndarray,
    edt_radius_px: np.ndarray,
    density_map: np.ndarray,
    config: PhysicalCalibrationConfig,
) -> Stage3Result:
    """Execute Stage 3: Graph Instantiation, Bridge Collapsing, and Momentum Untangling."""
    raw_graph = build_skeleton_graph(pruned_skeleton, edt_radius_px)
    collapsed_graph, collapsed_count = collapse_short_bridges(
        raw_graph,
        bridge_collapse_factor=config.bridge_collapse_factor,
    )
    untangled_paths = untangle_graph_paths(
        collapsed_graph,
        edt_radius_px,
        density_map,
        config,
    )
    return Stage3Result(
        raw_graph=raw_graph,
        collapsed_graph=collapsed_graph,
        collapsed_bridge_count=collapsed_count,
        untangled_paths=untangled_paths,
    )
