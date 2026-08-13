#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified stent graph printing-path optimizer — V2
================================================

Reads directly:
    square_list.txt
    arowhead_list.txt
    honeycomb_list.txt
    reentrant_list.txt

Automatic classification:
    0 odd vertices -> Euler circuit
    2 odd vertices -> Euler trail
   >2 odd vertices -> Open Chinese Postman Problem (OCPP)

Hard manufacturing constraint:
    NO RETRACT between print edges. The final toolpath is one continuous trail.

V2 route priority:
    * Square / Arrowhead (Euler graphs): prefer the straightest possible
      continuation at INTERIOR nodes. Bends at the external boundary are
      weakly penalized because they are normally required to return into the
      lattice. This reproduces the requested "run straight / diagonal through
      interior transition nodes, turn mainly at the boundary" behavior.
    * Honeycomb / Re-entrant (non-Euler): OCPP first minimizes duplicated edge
      traversals, then duplicated geometric length. Among equivalent continuous
      Euler trails on the augmented graph, support quality and smoothness are
      optimized.

Support heuristic:
    STRONG_INTERSECTION > CONTACT_SUPPORT > UNSUPPORTED.
    OCPP duplicate traversals are reported as OVERPRINT and never rewarded as
    support.  The heuristic is inspired by the attached paper's distinction
    between intersected and contact support nodes, while this program remains
    an edge-covering Euler/OCPP planner rather than the paper's layer-
    decomposition algorithm.

Outputs per pattern:
    optimized_route.csv, route_node_sequence.txt, optimized_route_2d.png,
    transition_report.csv, support_report.txt, optimization_summary.json,
    toolpath_rotary_AZ.gcode, viewer_3d_cylinder.html.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import random
import shutil
import sys
import time
import webbrowser
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

# Matplotlib is used only for static route validation images.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

try:
    import plotly.graph_objects as go
    from plotly.colors import sample_colorscale
except Exception:
    go = None
    sample_colorscale = None


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass
class PrintParams:
    diameter_mm: float = 8.0
    stent_length_mm: float = 20.0
    line_width_mm: float = 0.40
    layer_height_mm: float = 0.30
    filament_diameter_mm: float = 1.75
    feed_mm_min: float = 600.0
    flow_multiplier: float = 1.0
    candidates: int = 500
    seed: int = 12345
    sharp_turn_deg: float = 120.0
    straight_tol_deg: float = 18.0
    boundary_tol_ratio: float = 1e-6


@dataclass
class RouteMetrics:
    support_strong: int
    support_contact: int
    unsupported: int
    overprint_steps: int
    straight_transitions: int
    interior_bends: int
    avoidable_interior_bends: int
    boundary_turns: int
    sharp_turns: int
    interior_turn_deg: float
    total_turn_deg: float
    max_turn_deg: float
    route_2d_length: float

    def score_tuple(self, algorithm: str) -> tuple:
        # For Euler-capable square/arrowhead, straight-through interior motion is
        # the main manufacturing preference. Support breaks ties, while boundary
        # turns are intentionally not treated as interior bend violations.
        if algorithm in {"EULER_CIRCUIT", "EULER_TRAIL"}:
            return (
                self.avoidable_interior_bends,
                self.unsupported,
                -self.support_strong,
                self.interior_bends,
                self.interior_turn_deg,
                -self.support_contact,
                self.sharp_turns,
                self.total_turn_deg,
            )
        # For OCPP, duplicate count/length were already minimized during graph
        # augmentation. Route ordering then prioritizes printable support.
        return (
            self.unsupported,
            -self.support_strong,
            -self.support_contact,
            self.avoidable_interior_bends,
            self.interior_bends,
            self.sharp_turns,
            self.interior_turn_deg,
            self.total_turn_deg,
        )


# -----------------------------------------------------------------------------
# Input parsing / graph construction
# -----------------------------------------------------------------------------

def load_node_edge_file(path: Path):
    """Safely parse Python-style assignments `nodes = [...]`, `edges = [...]`."""
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    vals = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id in {"nodes", "edges"}:
                    vals[target.id] = ast.literal_eval(stmt.value)
    if "nodes" not in vals or "edges" not in vals:
        raise ValueError(f"{path.name}: must contain assignments 'nodes = [...]' and 'edges = [...]'.")

    nodes = [tuple(map(float, p[:2])) for p in vals["nodes"]]
    raw_edges = [tuple(map(int, e[:2])) for e in vals["edges"]]
    if not nodes or not raw_edges:
        raise ValueError(f"{path.name}: empty nodes or edges.")

    n = len(nodes)
    all_ids = [q for e in raw_edges for q in e]
    # Detect the arrowhead file's 1-based indexing automatically.
    if min(all_ids) >= 1 and max(all_ids) == n:
        input_offset = 1
        edges = [(u - 1, v - 1) for u, v in raw_edges]
        labels = {i: i + 1 for i in range(n)}
    elif min(all_ids) >= 0 and max(all_ids) <= n - 1:
        input_offset = 0
        edges = list(raw_edges)
        labels = {i: i for i in range(n)}
    else:
        raise ValueError(
            f"{path.name}: node IDs do not match either 0-based [0,{n-1}] or 1-based [1,{n}] indexing."
        )

    for u, v in edges:
        if u == v:
            raise ValueError(f"{path.name}: self-loop edge {(labels[u], labels[v])} is not supported.")
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"{path.name}: edge {(u, v)} references a missing node.")

    return nodes, edges, labels, input_offset


def build_graph(nodes, edges):
    G = nx.Graph()
    for i, (x, y) in enumerate(nodes):
        G.add_node(i, x=float(x), y=float(y))
    seen = set()
    for eid, (u, v) in enumerate(edges):
        key = tuple(sorted((u, v)))
        if key in seen:
            raise ValueError(f"Duplicate undirected input edge detected: {key}")
        seen.add(key)
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        length = math.hypot(x2 - x1, y2 - y1)
        G.add_edge(u, v, length=length, edge_id=eid)
    if not nx.is_connected(G):
        comps = [len(c) for c in nx.connected_components(G)]
        raise ValueError(f"Graph is disconnected; component sizes = {comps}. A no-retract single trail is impossible.")
    return G


def classify_graph(G: nx.Graph):
    odd = sorted([v for v, d in G.degree if d % 2 == 1])
    if len(odd) == 0:
        return "EULER_CIRCUIT", odd
    if len(odd) == 2:
        return "EULER_TRAIL", odd
    return "OPEN_CHINESE_POSTMAN", odd


# -----------------------------------------------------------------------------
# Open Chinese Postman augmentation — zero retracts
# -----------------------------------------------------------------------------

def _shortest_path_lexicographic_weights(G: nx.Graph, odd: Sequence[int]):
    """Return odd-pair shortest paths minimizing edge count first, then length."""
    total_len = sum(d["length"] for _, _, d in G.edges(data=True))
    # One extra duplicated edge must outweigh any possible total length difference
    # across all paired shortest paths. This makes the scalar cost lexicographic.
    BIG = (len(odd) + 2) * max(total_len, 1.0) + 1.0
    for u, v, d in G.edges(data=True):
        d["cpp_weight"] = BIG + d["length"]

    pair_info = {}
    for i, src in enumerate(odd):
        lengths, paths = nx.single_source_dijkstra(G, src, weight="cpp_weight")
        for dst in odd[i + 1:]:
            path = paths[dst]
            n_edges = len(path) - 1
            geom_len = sum(G[path[k]][path[k + 1]]["length"] for k in range(n_edges))
            scalar = lengths[dst]
            pair_info[(src, dst)] = {
                "path": path,
                "edge_count": n_edges,
                "length": geom_len,
                "weight": scalar,
            }
    return pair_info


def solve_open_cpp_augmentation(G: nx.Graph, odd: Sequence[int]):
    """
    Choose open endpoints and duplicated paths globally with two dummy nodes.
    The objective is minimum duplicated edge traversals, then duplicated length.
    """
    if len(odd) <= 2:
        return [], list(odd), 0, 0.0
    if len(odd) % 2:
        raise ValueError("Undirected graph must have an even number of odd vertices.")

    pair_info = _shortest_path_lexicographic_weights(G, odd)
    K = nx.Graph()
    for v in odd:
        K.add_node(v, kind="odd")
    D0 = "__OPEN_DUMMY_0__"
    D1 = "__OPEN_DUMMY_1__"
    K.add_node(D0, kind="dummy")
    K.add_node(D1, kind="dummy")

    for i, u in enumerate(odd):
        for v in odd[i + 1:]:
            info = pair_info[(u, v)] if (u, v) in pair_info else pair_info[(v, u)]
            K.add_edge(u, v, weight=info["weight"])
        # Tiny deterministic perturbation breaks ties without changing the
        # lexicographic edge-count/length objective in any practical way.
        K.add_edge(D0, u, weight=1e-8 * (u + 1))
        K.add_edge(D1, u, weight=1e-8 * (len(G) - u + 1))

    matching = nx.algorithms.matching.min_weight_matching(K, weight="weight")
    if len(matching) * 2 != len(K.nodes):
        raise RuntimeError("OCPP matching did not produce a perfect matching.")

    endpoints = []
    augmentation_pairs = []
    for a, b in matching:
        if a in {D0, D1} or b in {D0, D1}:
            odd_node = b if a in {D0, D1} else a
            endpoints.append(int(odd_node))
        else:
            u, v = sorted((int(a), int(b)))
            info = pair_info[(u, v)]
            augmentation_pairs.append((u, v, info["path"]))

    endpoints = sorted(endpoints)
    if len(endpoints) != 2:
        raise RuntimeError(f"Expected exactly 2 open endpoints, got {endpoints}")

    dup_count = sum(len(path) - 1 for _, _, path in augmentation_pairs)
    dup_len = 0.0
    for _, _, path in augmentation_pairs:
        dup_len += sum(G[path[k]][path[k + 1]]["length"] for k in range(len(path) - 1))
    return augmentation_pairs, endpoints, dup_count, dup_len


def build_augmented_multigraph(G: nx.Graph, augmentation_pairs):
    M = nx.MultiGraph()
    M.add_nodes_from(G.nodes(data=True))
    uid = 0
    for u, v, d in G.edges(data=True):
        M.add_edge(
            u, v,
            key=f"e{uid}",
            uid=uid,
            duplicate=False,
            source_edge_id=int(d["edge_id"]),
            length=float(d["length"]),
        )
        uid += 1

    for pair_idx, (a, b, path) in enumerate(augmentation_pairs):
        for k in range(len(path) - 1):
            u, v = path[k], path[k + 1]
            d = G[u][v]
            M.add_edge(
                u, v,
                key=f"d{uid}",
                uid=uid,
                duplicate=True,
                source_edge_id=int(d["edge_id"]),
                duplicate_pair=f"{a}-{b}",
                length=float(d["length"]),
            )
            uid += 1
    return M


# -----------------------------------------------------------------------------
# Euler-trail generation
# -----------------------------------------------------------------------------

def _edge_records(M: nx.MultiGraph):
    rec = {}
    incident = {v: [] for v in M.nodes}
    for u, v, key, d in M.edges(keys=True, data=True):
        uid = int(d["uid"])
        rec[uid] = (u, v, key, dict(d))
        incident[u].append(uid)
        incident[v].append(uid)
    return rec, incident


def _turn_angle(prev, cur, nxt, coords):
    if prev is None:
        return 0.0
    a = np.array(coords[cur], float) - np.array(coords[prev], float)
    b = np.array(coords[nxt], float) - np.array(coords[cur], float)
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(c))


def boundary_nodes_bbox(coords, tol_ratio=1e-6):
    xs = [p[0] for p in coords]; ys = [p[1] for p in coords]
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    scale = max(xmax - xmin, ymax - ymin, 1.0)
    tol = tol_ratio * scale
    return {
        i for i, (x, y) in enumerate(coords)
        if abs(x-xmin) <= tol or abs(x-xmax) <= tol or abs(y-ymin) <= tol or abs(y-ymax) <= tol
    }


def randomized_hierholzer(M: nx.MultiGraph, start: int, rng: random.Random, coords):
    """General randomized Euler generator used for OCPP augmented multigraphs."""
    rec, incident = _edge_records(M)
    unused = set(rec.keys())
    stack_v = [start]
    stack_e = []
    stack_prev = [None]
    circuit_e = []

    while stack_v:
        v = stack_v[-1]
        avail = [eid for eid in incident[v] if eid in unused]
        if avail:
            prev = stack_prev[-1]
            scored = []
            for eid in avail:
                u, w, key, d = rec[eid]
                nxt = w if v == u else u
                dup_pen = 0.20 if d.get("duplicate", False) else 0.0
                turn_pen = _turn_angle(prev, v, nxt, coords) / 180.0 if prev is not None else 0.0
                jitter = rng.random() * 1.15
                scored.append((turn_pen + dup_pen + jitter, eid, nxt))
            scored.sort(key=lambda t: t[0])
            _, eid, nxt = scored[0]
            unused.remove(eid)
            stack_v.append(nxt)
            stack_e.append(eid)
            stack_prev.append(v)
        else:
            stack_v.pop(); stack_prev.pop()
            if stack_e:
                circuit_e.append(stack_e.pop())

    edge_order = list(reversed(circuit_e))
    if len(edge_order) != M.number_of_edges():
        raise RuntimeError("Hierholzer did not consume every multigraph edge.")

    route = []
    cur = start
    for eid in edge_order:
        u, v, key, d = rec[eid]
        if cur == u: nxt = v
        elif cur == v: nxt = u
        else: raise RuntimeError("Generated Euler edge order is not continuous.")
        route.append((cur, nxt, eid, d))
        cur = nxt
    return route


def _unused_simple_graph(rec, unused):
    H = nx.Graph()
    pair_count = {}
    for eid in unused:
        u, v, _, _ = rec[eid]
        key = tuple(sorted((u, v)))
        pair_count[key] = pair_count.get(key, 0) + 1
        H.add_edge(u, v)
    return H, pair_count


def randomized_fleury_straight(M: nx.MultiGraph, start: int, rng: random.Random,
                               coords, boundary_nodes, straight_tol_deg=18.0):
    """
    Fleury-style direct Euler walk for Euler-capable graphs.

    Unlike Hierholzer cycle splicing, each local choice becomes the actual next
    printed edge. At interior nodes it strongly preserves a straight/diagonal
    continuation. At outer-boundary nodes turn penalty is relaxed so the path
    can turn back into the lattice.
    """
    rec, incident = _edge_records(M)
    unused = set(rec)
    route = []
    cur = start
    prev = None
    printed_segments = []

    while unused:
        avail = [eid for eid in incident[cur] if eid in unused]
        if not avail:
            raise RuntimeError("Fleury walk got stuck before all edges were consumed.")

        # Fleury rule: unless forced, avoid bridges of the remaining graph.
        nonbridge = list(avail)
        if len(avail) > 1:
            H, pair_count = _unused_simple_graph(rec, unused)
            bridge_pairs = {tuple(sorted(e)) for e in nx.bridges(H)} if H.number_of_edges() else set()
            tmp = []
            for eid in avail:
                u, v, _, _ = rec[eid]
                pair = tuple(sorted((u, v)))
                # Parallel multiedges cannot individually be bridges.
                is_bridge = pair in bridge_pairs and pair_count.get(pair, 0) == 1
                if not is_bridge:
                    tmp.append(eid)
            if tmp:
                nonbridge = tmp

        candidates = []
        possible_angles = []
        if prev is not None:
            for eid in nonbridge:
                u, v, _, _ = rec[eid]
                nxt = v if cur == u else u
                possible_angles.append(_turn_angle(prev, cur, nxt, coords))
        has_straight_option = bool(possible_angles) and min(possible_angles) <= straight_tol_deg

        for eid in nonbridge:
            u, v, _, d = rec[eid]
            nxt = v if cur == u else u
            ang = _turn_angle(prev, cur, nxt, coords) if prev is not None else 0.0
            at_boundary = cur in boundary_nodes

            # Interior straightness dominates. If a nearly straight continuation
            # exists, bending away from it receives a large penalty.
            avoidable = 1.0 if (prev is not None and not at_boundary and has_straight_option and ang > straight_tol_deg) else 0.0
            if at_boundary:
                turn_cost = 0.04 * (ang / 180.0)
                jitter = rng.random() * 0.85
            else:
                turn_cost = 1.8 * (ang / 180.0)
                jitter = rng.random() * 0.28

            # Small support preference: crossing / landing on older material is
            # useful, but it never overrules a straight interior continuation.
            p1, p2 = coords[cur], coords[nxt]
            support_bonus = 0.0
            for pu, pv in printed_segments:
                if same_undirected_segment(cur, nxt, pu, pv):
                    continue
                if proper_segment_intersection(p1, p2, coords[pu], coords[pv]):
                    support_bonus = -0.20
                    break
            score = 8.0 * avoidable + turn_cost + support_bonus + jitter
            candidates.append((score, eid, nxt))

        candidates.sort(key=lambda x: x[0])
        _, eid, nxt = candidates[0]
        u, v, key, d = rec[eid]
        unused.remove(eid)
        route.append((cur, nxt, eid, d))
        printed_segments.append((cur, nxt))
        prev, cur = cur, nxt

    return route



def _pair_turn_cost(v, e1, e2, rec, coords, boundary_nodes):
    u1, w1, _, _ = rec[e1]
    u2, w2, _, _ = rec[e2]
    n1 = w1 if v == u1 else u1
    n2 = w2 if v == u2 else u2
    ang = _turn_angle(n1, v, n2, coords)
    factor = 0.08 if v in boundary_nodes else 1.0
    return factor * ang, ang


def _three_pairings(e):
    a,b,c,d=e
    return [((a,b),(c,d)), ((a,c),(b,d)), ((a,d),(b,c))]


def _transition_components(pairing, edge_ids):
    T=nx.Graph(); T.add_nodes_from(edge_ids)
    seen=set()
    for v,pmap in pairing.items():
        for e1,e2 in pmap.items():
            if e2 is None: continue
            key=tuple(sorted((e1,e2)))
            if key in seen: continue
            seen.add(key); T.add_edge(e1,e2,vertex=v)
    comps=list(nx.connected_components(T))
    comp_of={e:i for i,c in enumerate(comps) for e in c}
    return comps,comp_of,T


def min_turn_meta_euler(M: nx.MultiGraph, G: nx.Graph, coords, boundary_nodes):
    """
    Build preferred local edge transitions, decompose them into maximal
    smooth strokes/cycles, then connect those components with a minimum-cost
    spanning tree on a meta-graph. This is the no-retract analogue of the
    user's reference "Euler on meta-graph": straight/diagonal transitions are
    preserved inside strokes and only the minimum necessary transition switches
    are introduced to obtain one continuous Euler trail.
    """
    rec,incident=_edge_records(M)
    if any(d.get('duplicate',False) for _,_,_,d in M.edges(keys=True,data=True)):
        raise ValueError('min_turn_meta_euler is intended for native Euler graphs without duplicated edges.')

    pairing={v:{} for v in M.nodes}
    local_choice={}
    for v,eids0 in incident.items():
        eids=list(eids0)
        deg=len(eids)
        if deg==0: continue
        if deg==1:
            pairing[v][eids[0]]=None
        elif deg==2:
            a,b=eids; pairing[v][a]=b; pairing[v][b]=a
            c,ang=_pair_turn_cost(v,a,b,rec,coords,boundary_nodes)
            local_choice[v]={'pairs':[(a,b)],'weighted_cost':c,'raw_turn_sum':ang,'degree':2}
        elif deg==4:
            options=[]
            for pairs in _three_pairings(eids):
                wc=raw=0.0
                for a,b in pairs:
                    c,ang=_pair_turn_cost(v,a,b,rec,coords,boundary_nodes)
                    wc+=c; raw+=ang
                options.append((wc,raw,pairs))
            options.sort(key=lambda x:(x[0],x[1],x[2]))
            wc,raw,pairs=options[0]
            for a,b in pairs:
                pairing[v][a]=b; pairing[v][b]=a
            local_choice[v]={'pairs':[tuple(x) for x in pairs],'weighted_cost':wc,'raw_turn_sum':raw,'degree':4}
        else:
            # Generic fallback for unexpected degree >4: greedily pair the
            # locally smoothest remaining half-edges. Current four datasets use
            # only degree 1,2,4, so this is defensive rather than primary logic.
            remain=set(eids); pairs=[]; wc=raw=0.0
            if deg%2==1:
                # leave one half-edge unpaired only if this is an odd endpoint
                # (not expected beyond degree1 in supplied Euler graphs)
                leave=min(remain); pairing[v][leave]=None; remain.remove(leave)
            while remain:
                best=None
                rr=sorted(remain)
                for i,a in enumerate(rr):
                    for b in rr[i+1:]:
                        c,ang=_pair_turn_cost(v,a,b,rec,coords,boundary_nodes)
                        if best is None or (c,ang,a,b)<best:
                            best=(c,ang,a,b)
                c,ang,a,b=best; remain.remove(a); remain.remove(b)
                pairing[v][a]=b; pairing[v][b]=a
                pairs.append((a,b)); wc+=c; raw+=ang
            local_choice[v]={'pairs':pairs,'weighted_cost':wc,'raw_turn_sum':raw,'degree':deg}

    edge_ids=sorted(rec)
    comps,comp_of,T=_transition_components(pairing,edge_ids)
    initial_components=len(comps)
    switches=[]

    if initial_components>1:
        meta=nx.Graph(); meta.add_nodes_from(range(initial_components))
        # A degree-4 vertex holding two preferred transition pairs can join the
        # two stroke components by cross-pairing them. Store the cheapest such
        # switch as a meta-edge.
        for v,info in local_choice.items():
            if info.get('degree')!=4: continue
            pairs=[tuple(x) for x in info['pairs']]
            if len(pairs)!=2: continue
            p1,p2=pairs
            c1=comp_of[p1[0]]; c2=comp_of[p2[0]]
            if c1==c2: continue
            a,b=p1; c,d=p2
            alts=[((a,c),(b,d)),((a,d),(b,c))]
            best_alt=None
            for alt in alts:
                wc=raw=0.0
                for x,y in alt:
                    cc,aa=_pair_turn_cost(v,x,y,rec,coords,boundary_nodes)
                    wc+=cc; raw+=aa
                inc=wc-info['weighted_cost']
                candidate=(inc,raw,alt)
                if best_alt is None or candidate<best_alt:
                    best_alt=candidate
            inc,raw,alt=best_alt
            data={'weight':max(0.0,inc),'vertex':v,'old_pairs':pairs,'new_pairs':[tuple(x) for x in alt],
                  'raw_turn_sum_new':raw,'raw_turn_sum_old':info['raw_turn_sum']}
            if meta.has_edge(c1,c2):
                if data['weight'] < meta[c1][c2]['weight']:
                    meta[c1][c2].update(data)
            else:
                meta.add_edge(c1,c2,**data)

        if not nx.is_connected(meta):
            raise RuntimeError(f'Preferred-transition meta-graph is disconnected: {initial_components} stroke components cannot be joined by degree-4 switches.')
        mst=nx.minimum_spanning_tree(meta,weight='weight')
        for ca,cb,data in mst.edges(data=True):
            v=data['vertex']
            # clear all degree-4 pairings at this vertex and apply cross pairing
            for e in list(pairing[v]): pairing[v].pop(e,None)
            for a,b in data['new_pairs']:
                pairing[v][a]=b; pairing[v][b]=a
            switches.append({'vertex':v,'meta_components':[int(ca),int(cb)],
                             'incremental_weighted_turn_cost':float(data['weight']),
                             'old_pairs':[list(x) for x in data['old_pairs']],
                             'new_pairs':[list(x) for x in data['new_pairs']],
                             'old_raw_turn_sum':float(data['raw_turn_sum_old']),
                             'new_raw_turn_sum':float(data['raw_turn_sum_new'])})

    final_comps,_,_=_transition_components(pairing,edge_ids)
    if len(final_comps)!=1:
        raise RuntimeError(f'Meta-graph transition switching did not produce one Euler component; got {len(final_comps)}.')

    odd=[v for v,d in M.degree if d%2]
    if len(odd)==2:
        start=odd[0]
    elif len(odd)==0:
        # Choose a boundary vertex; rotation is optimized later.
        candidates=[v for v in boundary_nodes if M.degree(v)>0]
        start=min(candidates) if candidates else min(M.nodes)
    else:
        raise RuntimeError('Unexpected odd-degree count in native Euler graph.')

    # Pick the first half-edge. For an open trail, degree-1 start fixes it. For a
    # circuit any incident edge yields a rotation/orientation of the same paired tour.
    first=min(incident[start])
    route=[]; used=set(); cur=start; eid=first
    while True:
        if eid in used:
            break
        u,v,key,d=rec[eid]
        nxt=v if cur==u else u
        route.append((cur,nxt,eid,d)); used.add(eid)
        next_e=pairing[nxt].get(eid)
        if next_e is None:
            cur=nxt; break
        cur=nxt; eid=next_e
    if len(used)!=len(edge_ids):
        raise RuntimeError(f'Transition-system trace covered {len(used)}/{len(edge_ids)} edges.')

    diagnostics={
        'initial_smooth_components':initial_components,
        'initial_smooth_strokes_edge_uids':[sorted(int(e) for e in c) for c in comps],
        'meta_switch_count':len(switches),
        'meta_switches':switches,
        'preferred_local_transition_cost':float(sum(i.get('weighted_cost',0.0) for i in local_choice.values())),
        'meta_added_turn_cost':float(sum(x['incremental_weighted_turn_cost'] for x in switches)),
    }
    return route,diagnostics


def rotate_circuit_route(route,k):
    if not route: return route
    k%=len(route)
    return route[k:]+route[:k]


def reverse_route(route):
    return [(v, u, eid, d) for (u, v, eid, d) in reversed(route)]


# -----------------------------------------------------------------------------
# Support and turn scoring
# -----------------------------------------------------------------------------

def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def proper_segment_intersection(p1, p2, q1, q2, eps=1e-9):
    """True only for a strict interior crossing; endpoint touch/collinear overlap is False."""
    o1 = _orient(p1, p2, q1)
    o2 = _orient(p1, p2, q2)
    o3 = _orient(q1, q2, p1)
    o4 = _orient(q1, q2, p2)
    return (o1 * o2 < -eps) and (o3 * o4 < -eps)


def same_undirected_segment(u1, v1, u2, v2):
    return (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2)


def annotate_and_score_route(route, coords, G: nx.Graph, algorithm: str,
                             sharp_turn_deg=120.0, straight_tol_deg=18.0,
                             boundary_tol_ratio=1e-6):
    printed = []
    incident_count = {i: 0 for i in range(len(coords))}
    boundary = boundary_nodes_bbox(coords, boundary_tol_ratio)
    rows = []
    strong = contact = unsupported = overprint = 0
    straight_transitions = interior_bends = avoidable_bends = boundary_turns = 0
    sharp = 0
    total_turn = interior_turn = max_turn = 0.0
    total_len = 0.0

    for i, (u, v, eid, d) in enumerate(route):
        p1, p2 = coords[u], coords[v]
        seg_len = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        total_len += seg_len

        # ---------------------- support classification ----------------------
        if d.get("duplicate", False):
            support_type = "OVERPRINT"
            overprint += 1
        else:
            has_strong = False
            has_contact = False
            for pu, pv, pp1, pp2 in printed:
                if same_undirected_segment(u, v, pu, pv):
                    continue
                if proper_segment_intersection(p1, p2, pp1, pp2):
                    has_strong = True
                    break

            # Strong support at a graph node: older deposited material already
            # forms an approximately through-going line at that node and the new
            # edge crosses that line rather than merely touching one endpoint.
            if not has_strong and printed:
                older = printed[:-1]  # immediate predecessor is continuity only
                for s_node in (u, v):
                    inc = []
                    for pu, pv, pp1, pp2 in older:
                        if s_node == pu: inc.append(pv)
                        elif s_node == pv: inc.append(pu)
                    if not inc:
                        continue
                    has_contact = True
                    if len(inc) < 2:
                        continue
                    cur_other = v if s_node == u else u
                    cur_vec = np.array(coords[cur_other], float)-np.array(coords[s_node], float)
                    ncur = float(np.linalg.norm(cur_vec))
                    if ncur <= 1e-12:
                        continue
                    for ia in range(len(inc)):
                        va=np.array(coords[inc[ia]],float)-np.array(coords[s_node],float)
                        na=float(np.linalg.norm(va))
                        if na <= 1e-12: continue
                        for ib in range(ia+1,len(inc)):
                            vb=np.array(coords[inc[ib]],float)-np.array(coords[s_node],float)
                            nb=float(np.linalg.norm(vb))
                            if nb <= 1e-12: continue
                            cab=float(np.clip(np.dot(va,vb)/(na*nb),-1.0,1.0))
                            old_pair_ang=math.degrees(math.acos(cab))
                            if old_pair_ang < 150.0: continue
                            cca=float(np.clip(np.dot(cur_vec,va)/(ncur*na),-1.0,1.0))
                            cross_ang=math.degrees(math.acos(cca))
                            if 20.0 <= cross_ang <= 160.0:
                                has_strong=True; break
                        if has_strong: break
                    if has_strong: break

            if has_strong:
                support_type="STRONG_INTERSECTION"; strong += 1
            else:
                older_at_start = incident_count[u] >= 2
                older_at_end = incident_count[v] >= 1
                if has_contact or older_at_start or older_at_end:
                    support_type="CONTACT_SUPPORT"; contact += 1
                else:
                    support_type="UNSUPPORTED"; unsupported += 1

        # ---------------------- turn / bend classification -----------------
        turn_deg = 0.0
        transition_location = "START"
        avoidable = False
        if i > 0:
            prev = route[i-1][0]
            if route[i-1][1] != u:
                raise RuntimeError("Route lost continuity while scoring.")
            turn_deg = _turn_angle(prev, u, v, coords)
            total_turn += turn_deg
            max_turn = max(max_turn, turn_deg)
            at_boundary = u in boundary
            transition_location = "BOUNDARY" if at_boundary else "INTERIOR"

            # Compare the chosen transition with the geometrically smoothest
            # continuation available at this graph node. Arrowhead's preferred
            # diagonal continuation is not perfectly collinear, so a RELATIVE
            # criterion is more useful than requiring a 0-degree straight line.
            alt_angles=[]
            for w in G.neighbors(u):
                if w==prev: continue
                alt_angles.append(_turn_angle(prev,u,w,coords))
            best_local=min(alt_angles) if alt_angles else turn_deg
            locally_preferred = turn_deg <= best_local + 5.0

            if at_boundary:
                if turn_deg > straight_tol_deg:
                    boundary_turns += 1
            else:
                if locally_preferred:
                    straight_transitions += 1
                if turn_deg > straight_tol_deg:
                    interior_bends += 1
                    interior_turn += turn_deg
                    if turn_deg >= sharp_turn_deg:
                        sharp += 1
                if turn_deg > best_local + 5.0:
                    avoidable=True
                    avoidable_bends += 1

        rows.append({
            "step": i+1,
            "u": u,
            "v": v,
            "eid": eid,
            "source_edge_id": int(d.get("source_edge_id",-1)),
            "duplicate": bool(d.get("duplicate",False)),
            "support_type": support_type,
            "turn_deg": turn_deg,
            "transition_location": transition_location,
            "avoidable_interior_bend": avoidable,
            "length_2d": seg_len,
        })
        printed.append((u,v,p1,p2))
        incident_count[u]+=1; incident_count[v]+=1

    metrics = RouteMetrics(
        support_strong=strong,
        support_contact=contact,
        unsupported=unsupported,
        overprint_steps=overprint,
        straight_transitions=straight_transitions,
        interior_bends=interior_bends,
        avoidable_interior_bends=avoidable_bends,
        boundary_turns=boundary_turns,
        sharp_turns=sharp,
        interior_turn_deg=float(interior_turn),
        total_turn_deg=float(total_turn),
        max_turn_deg=float(max_turn),
        route_2d_length=total_len,
    )
    return rows, metrics


def optimize_euler_route(M, endpoints, coords, params: PrintParams, G: nx.Graph, algorithm: str):
    boundary = boundary_nodes_bbox(coords, params.boundary_tol_ratio)
    best=None
    diagnostics={}

    if algorithm in {"EULER_CIRCUIT","EULER_TRAIL"}:
        # Deterministic minimum-turn transition decomposition + Euler on the
        # resulting meta-graph. This directly targets the behavior shown in the
        # user's arrowhead reference image.
        base_route,diagnostics=min_turn_meta_euler(M,G,coords,boundary)
        candidates=[]
        if algorithm=="EULER_CIRCUIT":
            # Same smooth transition system, different cyclic start positions.
            # Prefer start positions on the external boundary for support.
            idx=[i for i,r in enumerate(base_route) if r[0] in boundary]
            if not idx: idx=list(range(len(base_route)))
            for k in idx:
                rr=rotate_circuit_route(base_route,k)
                candidates.extend([rr,reverse_route(rr)])
        else:
            candidates=[base_route,reverse_route(base_route)]

        for candidate in candidates:
            rows,metrics=annotate_and_score_route(
                candidate,coords,G,algorithm,
                sharp_turn_deg=params.sharp_turn_deg,
                straight_tol_deg=params.straight_tol_deg,
                boundary_tol_ratio=params.boundary_tol_ratio,
            )
            key=metrics.score_tuple(algorithm)
            if best is None or key<best[0]:
                best=(key,candidate,rows,metrics)
    else:
        rng=random.Random(params.seed)
        starts=endpoints if len(endpoints)==2 else [v for v,d in M.degree if d>0]
        n=max(40,int(params.candidates))
        for k in range(n):
            start=starts[k%len(starts)]
            route=randomized_hierholzer(M,start,rng,coords)
            for candidate in (route,reverse_route(route)):
                rows,metrics=annotate_and_score_route(
                    candidate,coords,G,algorithm,
                    sharp_turn_deg=params.sharp_turn_deg,
                    straight_tol_deg=params.straight_tol_deg,
                    boundary_tol_ratio=params.boundary_tol_ratio,
                )
                key=metrics.score_tuple(algorithm)
                if best is None or key<best[0]:
                    best=(key,candidate,rows,metrics)

    if best is None:
        raise RuntimeError("No valid continuous Euler/OCPP route candidate was generated.")
    return best[1],best[2],best[3],diagnostics


# -----------------------------------------------------------------------------
# Cylinder mapping / G-code
# -----------------------------------------------------------------------------

def map_route_to_cylinder(route_rows, route, coords, params: PrintParams):
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xspan = xmax - xmin
    yspan = ymax - ymin
    if xspan <= 0 or yspan <= 0:
        raise ValueError("Graph must have non-zero X and Y span for cylindrical mapping.")

    def nominal_A(node):
        return 360.0 * (coords[node][0] - xmin) / xspan

    def Z(node):
        return params.stent_length_mm * (coords[node][1] - ymin) / yspan

    # Build continuous node sequence.
    node_seq = [route[0][0]] + [r[1] for r in route]
    A_unwrapped = [nominal_A(node_seq[0])]
    for node in node_seq[1:]:
        a0 = nominal_A(node)
        cur = A_unwrapped[-1]
        k0 = round((cur - a0) / 360.0)
        cands = [a0 + 360.0 * (k0 + dk) for dk in (-1, 0, 1)]
        nxt = min(cands, key=lambda a: abs(a - cur))
        A_unwrapped.append(float(nxt))
    Zs = [float(Z(n)) for n in node_seq]

    R = params.diameter_mm / 2.0
    filament_area = math.pi * (params.filament_diameter_mm / 2.0) ** 2
    E_abs = 0.0
    enriched = []
    for i, row in enumerate(route_rows):
        A0, A1 = A_unwrapped[i], A_unwrapped[i + 1]
        Z0, Z1 = Zs[i], Zs[i + 1]
        dtheta = math.radians(A1 - A0)
        ds = math.hypot(R * dtheta, Z1 - Z0)
        vol = params.line_width_mm * params.layer_height_mm * ds * params.flow_multiplier
        dE = vol / filament_area if filament_area > 0 else 0.0
        E_abs += dE
        theta0 = math.radians(A0)
        theta1 = math.radians(A1)
        out = dict(row)
        out.update({
            "A_start_deg": A0,
            "Z_start_mm": Z0,
            "A_end_deg": A1,
            "Z_end_mm": Z1,
            "X3_start_mm": R * math.cos(theta0),
            "Y3_start_mm": R * math.sin(theta0),
            "X3_end_mm": R * math.cos(theta1),
            "Y3_end_mm": R * math.sin(theta1),
            "surface_length_mm": ds,
            "dE_mm": dE,
            "E_abs_mm": E_abs,
        })
        enriched.append(out)
    return enriched, node_seq, A_unwrapped, Zs, E_abs


def write_gcode(path: Path, enriched_rows, labels, params: PrintParams, metadata):
    first = enriched_rows[0]
    lines = [
        "; Unified support-aware Euler / Open CPP stent toolpath",
        f"; pattern = {metadata['pattern']}",
        f"; algorithm = {metadata['algorithm']}",
        f"; nodes = {metadata['nodes']} ; original_edges = {metadata['edges']}",
        f"; odd_nodes = {metadata['odd_nodes_count']}",
        f"; duplicated_edge_traversals = {metadata['duplicate_edge_traversals']}",
        f"; retract_count = 0",
        f"; diameter_mm = {params.diameter_mm:.6f}",
        f"; stent_length_mm = {params.stent_length_mm:.6f}",
        "; IMPORTANT: Generic rotary A/Z post-processor. Verify axis sign, zero, units,",
        "; extrusion calibration, nozzle-mandrel gap, and controller semantics before use.",
        "G21 ; millimeters",
        "G90 ; absolute positioning",
        "M82 ; absolute extrusion",
        "G92 E0",
        f"G0 A{first['A_start_deg']:.6f} Z{first['Z_start_mm']:.6f} ; initial positioning only",
    ]
    for r in enriched_rows:
        u_lbl, v_lbl = labels[r["u"]], labels[r["v"]]
        flags = []
        if r["duplicate"]:
            flags.append("DUPLICATE")
        flags.append(r["support_type"])
        comment = f"step {r['step']} node {u_lbl}->{v_lbl} {'|'.join(flags)}"
        lines.append(
            f"G1 A{r['A_end_deg']:.6f} Z{r['Z_end_mm']:.6f} E{r['E_abs_mm']:.6f} "
            f"F{params.feed_mm_min:.1f} ; {comment}"
        )
    lines += ["; END - no retract commands were emitted", "M400"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Output writers / plots
# -----------------------------------------------------------------------------

def save_csv(path: Path, rows: List[dict], labels):
    if not rows:
        return
    cooked = []
    for r in rows:
        x = dict(r)
        x["from_node"] = labels[x.pop("u")]
        x["to_node"] = labels[x.pop("v")]
        cooked.append(x)
    fields = list(cooked[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(cooked)


def plot_route_2d(path: Path, G, coords, route_rows, labels, title):
    fig, ax = plt.subplots(figsize=(13, 9))
    boundary = boundary_nodes_bbox(coords)
    # Base graph
    for u, v in G.edges():
        ax.plot([coords[u][0],coords[v][0]],[coords[u][1],coords[v][1]],
                color="0.88", lw=1.4, zorder=1)

    segments=[]; vals=[]
    for r in route_rows:
        u,v=r["u"],r["v"]
        segments.append([coords[u],coords[v]])
        vals.append(r["step"])
    lc=LineCollection(segments,cmap="viridis",linewidths=3.0,zorder=3)
    lc.set_array(np.asarray(vals,dtype=float)); ax.add_collection(lc)

    # Direction arrows at segment midpoints, deliberately small like the user's
    # reference figure so the path remains readable.
    cmap=plt.get_cmap("viridis")
    n=max(len(route_rows)-1,1)
    for i,r in enumerate(route_rows):
        u,v=r["u"],r["v"]
        x0,y0=coords[u]; x1,y1=coords[v]
        mx=x0+0.56*(x1-x0); my=y0+0.56*(y1-y0)
        dx=0.16*(x1-x0); dy=0.16*(y1-y0)
        ax.annotate('',xy=(mx+dx,my+dy),xytext=(mx,my),
                    arrowprops=dict(arrowstyle='-|>',lw=1.0,color=cmap(i/n)),zorder=5)
        if r.get("duplicate"):
            ax.plot([x0,x1],[y0,y1],'--',lw=1.1,color='black',alpha=.8,zorder=4)

    for i,(x,y) in enumerate(coords):
        isb=i in boundary
        ax.scatter([x],[y],s=26 if isb else 20,facecolor='white',edgecolor='black',zorder=6)
        ax.text(x,y,str(labels[i]),fontsize=7,ha='center',va='bottom',zorder=7)

    # Mark avoidable interior bends so optimization quality can be inspected.
    for r in route_rows:
        if r.get("avoidable_interior_bend"):
            u=r["u"]; x,y=coords[u]
            ax.scatter([x],[y],s=85,facecolors='none',edgecolors='red',linewidths=1.5,zorder=8)

    st=route_rows[0]["u"]; en=route_rows[-1]["v"]
    ax.scatter([coords[st][0]],[coords[st][1]],s=105,zorder=9,label=f"START {labels[st]}")
    ax.scatter([coords[en][0]],[coords[en][1]],s=105,marker='s',zorder=9,label=f"END {labels[en]}")
    ax.set_aspect('equal',adjustable='box'); ax.set_title(title)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.grid(True,alpha=.2); ax.legend(loc='best')
    cb=fig.colorbar(lc,ax=ax,pad=.02); cb.set_label('Print sequence')
    fig.tight_layout(); fig.savefig(path,dpi=220); plt.close(fig)


def write_viewer_html(path: Path, enriched_rows, params: PrintParams, title: str):
    if go is None:
        path.write_text(
            "<html><body><h3>Plotly is not installed.</h3><p>Run: pip install plotly</p></body></html>",
            encoding="utf-8",
        )
        return

    fig = go.Figure()
    n = len(enriched_rows)
    for i, r in enumerate(enriched_rows):
        frac = 0 if n <= 1 else i / (n - 1)
        color = sample_colorscale("Viridis", [frac])[0]
        width = 6 if not r["duplicate"] else 8
        hover = (
            f"Step {r['step']}<br>Support: {r['support_type']}<br>"
            f"Transition: {r.get('transition_location','')}<br>"
            f"Avoidable interior bend: {r.get('avoidable_interior_bend',False)}<br>"
            f"Duplicate: {r['duplicate']}<br>Turn: {r['turn_deg']:.1f}°<br>"
            f"A: {r['A_start_deg']:.2f}→{r['A_end_deg']:.2f}°<br>"
            f"Z: {r['Z_start_mm']:.2f}→{r['Z_end_mm']:.2f} mm"
        )
        fig.add_trace(go.Scatter3d(
            x=[r["X3_start_mm"], r["X3_end_mm"]],
            y=[r["Y3_start_mm"], r["Y3_end_mm"]],
            z=[r["Z_start_mm"], r["Z_end_mm"]],
            mode="lines",
            line=dict(color=color, width=width),
            hovertext=[hover, hover],
            hoverinfo="text",
            showlegend=False,
        ))

    # Transparent cylinder reference surface.
    theta = np.linspace(0, 2 * math.pi, 64)
    z = np.linspace(0, params.stent_length_mm, 24)
    T, Z = np.meshgrid(theta, z)
    R = params.diameter_mm / 2.0
    X = R * np.cos(T)
    Y = R * np.sin(T)
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        opacity=0.10,
        showscale=False,
        hoverinfo="skip",
        colorscale=[[0, "rgb(180,180,180)"], [1, "rgb(220,220,220)"]],
        name="Mandrel",
    ))

    first = enriched_rows[0]
    last = enriched_rows[-1]
    fig.add_trace(go.Scatter3d(
        x=[first["X3_start_mm"]], y=[first["Y3_start_mm"]], z=[first["Z_start_mm"]],
        mode="markers+text", text=["START"], textposition="top center",
        marker=dict(size=7), name="START",
    ))
    fig.add_trace(go.Scatter3d(
        x=[last["X3_end_mm"]], y=[last["Y3_end_mm"]], z=[last["Z_end_mm"]],
        mode="markers+text", text=["END"], textposition="top center",
        marker=dict(size=7, symbol="diamond"), name="END",
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=45),
    )
    fig.write_html(str(path), include_plotlyjs="inline", full_html=True)


def write_support_report(path: Path, route_rows, labels):
    counts={}
    for r in route_rows:
        counts[r["support_type"]]=counts.get(r["support_type"],0)+1
    avoid=sum(bool(r.get("avoidable_interior_bend")) for r in route_rows)
    interior=sum(1 for r in route_rows if r.get("transition_location")=="INTERIOR" and r.get("turn_deg",0)>18.0)
    boundary=sum(1 for r in route_rows if r.get("transition_location")=="BOUNDARY" and r.get("turn_deg",0)>18.0)
    lines=[
        "SUPPORT + STRAIGHTNESS ROUTE REPORT",
        "===================================",
        "",
        "Hard constraint: retract_count = 0; one continuous Euler/OCPP trail.",
        "",
        "Support interpretation:",
        "  STRONG_INTERSECTION : new segment crosses older deposited material.",
        "  CONTACT_SUPPORT     : older material exists at the transition/contact node.",
        "  UNSUPPORTED         : no prior support detected for that segment.",
        "  OVERPRINT           : OCPP duplicate traversal; not rewarded as support.",
        "",
        "Straightness interpretation for Euler-capable square/arrowhead:",
        "  Interior nodes strongly prefer straight/diagonal continuation.",
        "  Outer-boundary turns are relaxed because they are usually required to return into the lattice.",
        "  A red ring in optimized_route_2d.png marks an avoidable interior bend.",
        "",
        "Support counts:",
    ]
    for k in ["STRONG_INTERSECTION","CONTACT_SUPPORT","UNSUPPORTED","OVERPRINT"]:
        lines.append(f"  {k:20s}: {counts.get(k,0)}")
    lines += [
        "",
        f"Avoidable interior bends : {avoid}",
        f"Interior bends (>18 deg) : {interior}",
        f"Boundary turns (>18 deg) : {boundary}",
    ]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")


# -----------------------------------------------------------------------------
# Main processing function
# -----------------------------------------------------------------------------

def process_pattern(input_path: Path, output_root: Path, params: PrintParams):
    t0 = time.time()
    nodes, edges, labels, offset = load_node_edge_file(input_path)
    G = build_graph(nodes, edges)
    algo, odd = classify_graph(G)

    if algo == "OPEN_CHINESE_POSTMAN":
        aug_pairs, endpoints, dup_count, dup_len = solve_open_cpp_augmentation(G, odd)
    elif algo == "EULER_TRAIL":
        aug_pairs, endpoints, dup_count, dup_len = [], odd, 0, 0.0
    else:
        aug_pairs, endpoints, dup_count, dup_len = [], [], 0, 0.0

    M = build_augmented_multigraph(G, aug_pairs)
    odd_after = sorted([v for v, d in M.degree if d % 2 == 1])
    if algo == "EULER_CIRCUIT" and odd_after:
        raise RuntimeError("Euler circuit graph unexpectedly has odd vertices after augmentation.")
    if algo != "EULER_CIRCUIT" and len(odd_after) != 2:
        raise RuntimeError(f"Expected 2 odd vertices after augmentation, got {odd_after}")
    if endpoints and sorted(endpoints) != odd_after:
        # This should always match; keep the actual multigraph values authoritative.
        endpoints = odd_after

    route, route_rows, metrics, euler_meta = optimize_euler_route(M, endpoints, nodes, params, G, algo)
    # Add user-facing node labels to meta-switch diagnostics (important for
    # arrowhead input, which is 1-based).
    euler_meta_user = json.loads(json.dumps(euler_meta)) if euler_meta else {}
    for sw in euler_meta_user.get("meta_switches", []):
        sw["vertex_label"] = labels.get(int(sw["vertex"]), int(sw["vertex"]))

    outdir = output_root / input_path.stem
    outdir.mkdir(parents=True, exist_ok=True)

    enriched, node_seq, As, Zs, E_total = map_route_to_cylinder(route_rows, route, nodes, params)

    pattern = input_path.stem.replace("_list", "")
    metadata = {
        "pattern": pattern,
        "source_file": input_path.name,
        "input_index_base": offset,
        "algorithm": algo,
        "nodes": len(nodes),
        "edges": len(edges),
        "odd_nodes_count": len(odd),
        "odd_nodes": [labels[v] for v in odd],
        "start_node": labels[route[0][0]],
        "end_node": labels[route[-1][1]],
        "augmented_multigraph_edges": M.number_of_edges(),
        "duplicate_edge_traversals": dup_count,
        "duplicate_length_2d": dup_len,
        "retract_count": 0,
        "continuous_route": True,
        "all_original_edges_covered": True,
        "support_metrics": asdict(metrics),
        "diameter_mm": params.diameter_mm,
        "stent_length_mm": params.stent_length_mm,
        "line_width_mm": params.line_width_mm,
        "layer_height_mm": params.layer_height_mm,
        "filament_diameter_mm": params.filament_diameter_mm,
        "feed_mm_min": params.feed_mm_min,
        "total_extrusion_E_mm": E_total,
        "total_surface_print_length_mm": sum(r["surface_length_mm"] for r in enriched),
        "straight_tol_deg": params.straight_tol_deg,
        "boundary_rule": "interior smooth transitions preserved; outer-boundary turns relaxed",
        "euler_meta_graph": euler_meta_user,
        "candidate_routes_evaluated_approx": params.candidates * 2,
        "runtime_s": time.time() - t0,
    }

    # Save augmentation detail.
    aug_json = []
    for a, b, path in aug_pairs:
        aug_json.append({
            "pair": [labels[a], labels[b]],
            "path": [labels[v] for v in path],
            "duplicated_edges": len(path) - 1,
            "duplicated_length": sum(G[path[k]][path[k+1]]["length"] for k in range(len(path)-1)),
        })

    (outdir / "optimization_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outdir / "ocpp_augmentation.json").write_text(
        json.dumps(aug_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outdir / "euler_meta_graph.json").write_text(
        json.dumps(euler_meta_user, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outdir / "route_node_sequence.txt").write_text(
        " -> ".join(str(labels[v]) for v in node_seq) + "\n", encoding="utf-8"
    )

    save_csv(outdir / "optimized_route.csv", enriched, labels)
    save_csv(outdir / "transition_report.csv", enriched, labels)
    dup_rows = [r for r in enriched if r["duplicate"]]
    if dup_rows:
        save_csv(outdir / "duplicated_edges_route.csv", dup_rows, labels)
    else:
        (outdir / "duplicated_edges_route.csv").write_text(
            "No duplicated edge traversals.\n", encoding="utf-8"
        )
    write_support_report(outdir / "support_report.txt", route_rows, labels)
    plot_route_2d(
        outdir / "optimized_route_2d.png", G, nodes, route_rows, labels,
        f"{pattern}: {algo} | duplicate={dup_count} | retract=0 | straight/support-aware",
    )
    write_viewer_html(
        outdir / "viewer_3d_cylinder.html", enriched, params,
        f"{pattern} — continuous straight/support-aware toolpath on cylinder",
    )
    write_gcode(outdir / "toolpath_rotary_AZ.gcode", enriched, labels, params, metadata)

    return metadata, outdir


# -----------------------------------------------------------------------------
# CLI / GUI
# -----------------------------------------------------------------------------

DEFAULT_FILES = {
    "square": "square_list.txt",
    "arowhead": "arowhead_list.txt",
    "honeycomb": "honeycomb_list.txt",
    "reentrant": "reentrant_list.txt",
}


def resolve_input_file(input_dir: Path, pattern: str):
    if pattern == "arrowhead":
        pattern = "arowhead"
    if pattern in DEFAULT_FILES:
        p = input_dir / DEFAULT_FILES[pattern]
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        return p
    p = Path(pattern)
    if p.exists():
        return p
    raise FileNotFoundError(f"Unknown pattern/file: {pattern}")


def run_cli(args):
    params = PrintParams(
        diameter_mm=args.diameter,
        stent_length_mm=args.length,
        line_width_mm=args.line_width,
        layer_height_mm=args.layer_height,
        filament_diameter_mm=args.filament_diameter,
        feed_mm_min=args.feed,
        flow_multiplier=args.flow,
        candidates=args.candidates,
        seed=args.seed,
        sharp_turn_deg=args.sharp_turn,
        straight_tol_deg=args.straight_tol,
    )
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    patterns = list(DEFAULT_FILES.keys()) if args.all else [args.pattern]
    summaries = []
    for pat in patterns:
        ip = resolve_input_file(input_dir, pat)
        meta, out = process_pattern(ip, output_dir, params)
        summaries.append(meta)
        print(
            f"[{meta['pattern']}] {meta['algorithm']} | odd={meta['odd_nodes_count']} | "
            f"dup={meta['duplicate_edge_traversals']} | retract=0 | "
            f"support strong/contact/unsupported="
            f"{meta['support_metrics']['support_strong']}/"
            f"{meta['support_metrics']['support_contact']}/"
            f"{meta['support_metrics']['unsupported']} | {out}"
        )

    (output_dir / "all_patterns_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def launch_gui(script_dir: Path):
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as e:
        print("Tkinter is unavailable. Use CLI mode with --all or --pattern.")
        print(e)
        return

    root = tk.Tk()
    root.title("Stent Graph Optimizer V2 — Straight Euler / Open CPP — No Retract")
    root.geometry("820x650")

    default_input = script_dir / "inputs"
    default_output = script_dir / "outputs"
    vars_ = {
        "input": tk.StringVar(value=str(default_input)),
        "output": tk.StringVar(value=str(default_output)),
        "pattern": tk.StringVar(value="ALL"),
        "D": tk.StringVar(value="8.0"),
        "L": tk.StringVar(value="20.0"),
        "lw": tk.StringVar(value="0.40"),
        "lh": tk.StringVar(value="0.30"),
        "fd": tk.StringVar(value="1.75"),
        "feed": tk.StringVar(value="600"),
        "cand": tk.StringVar(value="500"),
        "straight": tk.StringVar(value="18.0"),
    }
    latest_viewers = []

    frm = ttk.Frame(root, padding=12)
    frm.pack(fill="both", expand=True)
    frm.columnconfigure(1, weight=1)

    row = 0
    def add_path(label, key, choose_dir=True):
        nonlocal row
        ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frm, textvariable=vars_[key]).grid(row=row, column=1, sticky="ew", pady=4)
        def choose():
            d = filedialog.askdirectory(initialdir=vars_[key].get())
            if d:
                vars_[key].set(d)
        ttk.Button(frm, text="Browse", command=choose).grid(row=row, column=2, padx=5)
        row += 1

    add_path("Input folder", "input")
    add_path("Output folder", "output")

    ttk.Label(frm, text="Pattern").grid(row=row, column=0, sticky="w", pady=4)
    cb = ttk.Combobox(frm, textvariable=vars_["pattern"], state="readonly",
                      values=["ALL", "square", "arowhead", "honeycomb", "reentrant"])
    cb.grid(row=row, column=1, sticky="ew", pady=4)
    row += 1

    fields = [
        ("Stent diameter D (mm)", "D"),
        ("Stent length L (mm)", "L"),
        ("Line width (mm)", "lw"),
        ("Layer/strand height (mm)", "lh"),
        ("Filament diameter (mm)", "fd"),
        ("Feed (mm/min)", "feed"),
        ("Euler candidates", "cand"),
        ("Straight tolerance (deg)", "straight"),
    ]
    for label, key in fields:
        ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frm, textvariable=vars_[key], width=18).grid(row=row, column=1, sticky="w", pady=4)
        row += 1

    status = tk.Text(frm, height=13, wrap="word")
    status.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=10)
    frm.rowconfigure(row, weight=1)
    row += 1

    def log(s):
        status.insert("end", s + "\n")
        status.see("end")
        root.update_idletasks()

    def run_selected():
        latest_viewers.clear()
        try:
            params = PrintParams(
                diameter_mm=float(vars_["D"].get()),
                stent_length_mm=float(vars_["L"].get()),
                line_width_mm=float(vars_["lw"].get()),
                layer_height_mm=float(vars_["lh"].get()),
                filament_diameter_mm=float(vars_["fd"].get()),
                feed_mm_min=float(vars_["feed"].get()),
                candidates=int(vars_["cand"].get()),
                straight_tol_deg=float(vars_["straight"].get()),
            )
            idir = Path(vars_["input"].get())
            odir = Path(vars_["output"].get())
            odir.mkdir(parents=True, exist_ok=True)
            pats = list(DEFAULT_FILES) if vars_["pattern"].get() == "ALL" else [vars_["pattern"].get()]
            log("--- RUN ---")
            for pat in pats:
                ip = resolve_input_file(idir, pat)
                log(f"Optimizing {ip.name} ...")
                meta, out = process_pattern(ip, odir, params)
                latest_viewers.append(out / "viewer_3d_cylinder.html")
                sm = meta["support_metrics"]
                log(
                    f"{meta['pattern']}: {meta['algorithm']}; odd={meta['odd_nodes_count']}; "
                    f"duplicate={meta['duplicate_edge_traversals']}; retract=0; "
                    f"support strong/contact/unsupported={sm['support_strong']}/{sm['support_contact']}/{sm['unsupported']}"
                )
            log("Finished. Outputs written to: " + str(odir))
        except Exception as e:
            messagebox.showerror("Error", str(e))
            log("ERROR: " + str(e))

    def open_viewers():
        if not latest_viewers:
            messagebox.showinfo("3D Viewer", "Run the optimizer first.")
            return
        for p in latest_viewers:
            webbrowser.open(p.resolve().as_uri())

    btns = ttk.Frame(frm)
    btns.grid(row=row, column=0, columnspan=3, sticky="w")
    ttk.Button(btns, text="RUN OPTIMIZER", command=run_selected).pack(side="left", padx=4)
    ttk.Button(btns, text="OPEN 3D VIEWER", command=open_viewers).pack(side="left", padx=4)
    ttk.Button(btns, text="EXIT", command=root.destroy).pack(side="left", padx=4)

    root.mainloop()


def make_parser():
    p = argparse.ArgumentParser(description="Unified Euler/Open-CPP stent toolpath optimizer")
    p.add_argument("--input-dir", default=str(Path(__file__).resolve().parent / "inputs"))
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "outputs"))
    p.add_argument("--pattern", default="square", help="square|arowhead|honeycomb|reentrant or a .txt path")
    p.add_argument("--all", action="store_true", help="Process all four standard files")
    p.add_argument("--diameter", type=float, default=8.0)
    p.add_argument("--length", type=float, default=20.0)
    p.add_argument("--line-width", type=float, default=0.40)
    p.add_argument("--layer-height", type=float, default=0.30)
    p.add_argument("--filament-diameter", type=float, default=1.75)
    p.add_argument("--feed", type=float, default=600.0)
    p.add_argument("--flow", type=float, default=1.0)
    p.add_argument("--candidates", type=int, default=500)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--sharp-turn", type=float, default=120.0)
    p.add_argument("--straight-tol", type=float, default=18.0, help="Interior transition angle considered straight")
    p.add_argument("--gui", action="store_true", help="Open desktop Tkinter GUI")
    return p


def main():
    parser = make_parser()
    # Double-click / `python stent_path_optimizer.py` -> GUI.
    if len(sys.argv) == 1:
        launch_gui(Path(__file__).resolve().parent)
        return
    args = parser.parse_args()
    if args.gui:
        launch_gui(Path(__file__).resolve().parent)
    else:
        run_cli(args)


if __name__ == "__main__":
    main()
