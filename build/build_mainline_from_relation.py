import json
import math
import time
import re
from typing import List, Dict, Any, Tuple, Optional
import requests
import pathlib
import pandas as pd
from pathlib import Path



# -------------------------
# Overpass configuration
# -------------------------

MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
    "https://overpass-api.nextgis.com/api/interpreter",
]

TIMEOUT = 120
CHUNK_WAYS = 20


# -------------------------
# Geometry helpers
# -------------------------

def hav_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance (haversine) between two points in *meters*.
    """
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# -------------------------
# Overpass I/O
# -------------------------

def post_overpass(ql: str, attempt_limit: int = 3) -> dict:
    """
    POST a query to Overpass, rotating through mirrors with retries.
    """
    last_exc = None
    for attempt in range(1, attempt_limit + 1):
        for base in MIRRORS:
            try:
                resp = requests.post(
                    base,
                    data=ql.encode("utf-8"),
                    headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"},
                    timeout=TIMEOUT,
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_exc = e
        time.sleep(1.5 * attempt)
    raise RuntimeError(f"Overpass failed after retries: {last_exc}")


def fetch_relation(
    rel_name: str | None = None,
    rel_id: int | None = None,
    route_type: str = "road",
) -> Tuple[int, List[int], Dict[int, dict], Dict[int, Tuple[float, float]]]:
    """
    Fetch an OSM route relation (by id or by name/ref), then fetch its member ways
    (with node lists) and all referenced nodes.

    Returns:
        (rel_id, way_ids_ordered, ways_dict, nodes_dict)

        ways_dict[way_id] = OSM way object (with "nodes" list and "tags")
        nodes_dict[node_id] = (lat, lon)
    """
    # --- 0) Fetch relation object ---
    if rel_id is not None:
        q_rel = f'[out:json][timeout:{TIMEOUT}];relation({rel_id});out body;'
        rel_json = post_overpass(q_rel)
        rels = [e for e in rel_json.get("elements", []) if e["type"] == "relation"]
        if not rels:
            raise RuntimeError(f"Relation id {rel_id} not found")
        rel = rels[0]
    else:
        if not rel_name:
            raise ValueError("fetch_relation: provide rel_id or rel_name")

        # Try name= first
        ql_rel = (
            f'[out:json][timeout:{TIMEOUT}];'
            f'rel["type"="route"]["route"="{route_type}"]["name"="{rel_name}"];'
            f'out body;'
        )
        rel_json = post_overpass(ql_rel)
        rels = [e for e in rel_json.get("elements", []) if e["type"] == "relation"]

        if not rels:
            # Fallback to ref= (e.g., rel_name="101" matches ref="US 101")
            ql_ref = (
                f'[out:json][timeout:{TIMEOUT}];'
                f'rel["type"="route"]["route"="{route_type}"]["ref"="{rel_name}"];'
                f'out body;'
            )
            rel_json = post_overpass(ql_ref)
            rels = [e for e in rel_json.get("elements", []) if e["type"] == "relation"]
            if not rels:
                raise RuntimeError(f'Relation not found by name/ref: "{rel_name}"')

        rel = rels[0]

    rel_id = rel["id"]
    members = [m for m in rel.get("members", []) if m["type"] == "way"]
    way_ids_ordered = [m["ref"] for m in members]

    # --- 1) Fetch member ways (need 'out body' to get node lists) ---
    ways: Dict[int, dict] = {}
    for i in range(0, len(way_ids_ordered), CHUNK_WAYS):
        ids = ",".join(str(x) for x in way_ids_ordered[i : i + CHUNK_WAYS])
        ql = f"[out:json][timeout:{TIMEOUT}];way(id:{ids});out body;"
        j = post_overpass(ql)
        for e in j.get("elements", []):
            if e["type"] == "way":
                ways[e["id"]] = e
        time.sleep(0.2)

    # Sanity: refetch any stragglers that somehow lacked 'nodes'
    missing = [wid for wid, w in ways.items() if "nodes" not in w]
    if missing:
        ids = ",".join(map(str, missing))
        ql = f"[out:json][timeout:{TIMEOUT}];way(id:{ids});out body;"
        j = post_overpass(ql)
        for e in j.get("elements", []):
            if e["type"] == "way":
                ways[e["id"]] = e

    # --- 2) Collect & fetch nodes ---
    node_ids: List[int] = []
    for wid in way_ids_ordered:
        w = ways.get(wid)
        if w and "nodes" in w:
            node_ids.extend(w["nodes"])

    # dedupe while preserving order
    node_ids = list(dict.fromkeys(node_ids))

    nodes: Dict[int, Tuple[float, float]] = {}
    for i in range(0, len(node_ids), 500):
        ids = ",".join(str(x) for x in node_ids[i : i + 500])
        ql = f"[out:json][timeout:{TIMEOUT}];node(id:{ids});out skel qt;"
        j = post_overpass(ql)
        for e in j.get("elements", []):
            if e["type"] == "node":
                nodes[e["id"]] = (e["lat"], e["lon"])
        time.sleep(0.2)

    return rel_id, way_ids_ordered, ways, nodes


# -------------------------
# Graph construction & A*
# -------------------------

def build_graph_no_bearing(
    way_ids_ordered: List[int],
    ways: Dict[int, dict],
    nodes: Dict[int, Tuple[float, float]],
    facility_number: str,
    link_penalty: float = 1200.0,
    off_fac_penalty: float = 900.0,
    forbid_links: bool = False,
) -> Tuple[Dict[int, List[Tuple[int, float]]], Dict[Tuple[int, int], int]]:
    """
    Build a directed adjacency graph over the relation's ways.

    Edges:
      - cost = haversine(u, v) + (link_penalty if *_link) + (off_fac_penalty if way is not tagged with facility_number)
      - direction respects oneway tags.
    """
    adj: Dict[int, List[Tuple[int, float]]] = {nid: [] for nid in nodes.keys()}
    fac_re = re.compile(rf"\b{re.escape(str(facility_number))}\b")
    way_of_edge: Dict[Tuple[int, int], int] = {}

    for wid in way_ids_ordered:
        w = ways.get(wid)
        if not w:
            continue
        tags = w.get("tags", {})
        hwy = tags.get("highway", "") or ""
        is_link = hwy.endswith("_link")
        if forbid_links and is_link:
            continue

        wn = w.get("nodes") or []
        oneway = (tags.get("oneway", "") or "").lower() == "yes"
        ref_text = ((tags.get("ref", "") or "") + " " + (tags.get("name", "") or "")).strip()
        has_fac = bool(fac_re.search(ref_text))

        for i in range(len(wn) - 1):
            u, v = wn[i], wn[i + 1]
            if u not in nodes or v not in nodes:
                continue
            latu, lonu = nodes[u]
            latv, lonv = nodes[v]
            base = hav_m(latu, lonu, latv, lonv)

            cost_fwd = base + (link_penalty if is_link else 0.0) + (0.0 if has_fac else off_fac_penalty)
            adj[u].append((v, cost_fwd))
            way_of_edge[(u, v)] = wid

            if not oneway:
                cost_rev = base + (link_penalty if is_link else 0.0) + (0.0 if has_fac else off_fac_penalty)
                adj[v].append((u, cost_rev))
                way_of_edge[(v, u)] = wid

    return adj, way_of_edge


def astar(
    nodes: Dict[int, Tuple[float, float]],
    adj: Dict[int, List[Tuple[int, float]]],
    start: int,
    goal: int,
) -> List[int]:
    """
    A* shortest path on the relation graph, using haversine to goal as heuristic.
    Returns the node-id chain from start to goal (inclusive).
    """
    import heapq

    def h(nid: int) -> float:
        a = nodes[nid]
        b = nodes[goal]
        return hav_m(a[0], a[1], b[0], b[1])

    INF = 1e18
    g = {start: 0.0}
    prev: Dict[int, int] = {}
    pq: List[Tuple[float, int]] = [(h(start), start)]
    seen: set[int] = set()

    while pq:
        _, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == goal:
            break
        for v, c in adj.get(u, []):
            ng = g[u] + c
            if ng < g.get(v, INF):
                g[v] = ng
                prev[v] = u
                heapq.heappush(pq, (ng + h(v), v))

    if goal not in g:
        raise RuntimeError("No path between endpoints inside relation.")

    path: List[int] = []
    cur = goal
    while True:
        path.append(cur)
        if cur == start:
            break
        cur = prev[cur]
    path.reverse()
    return path


def chain_cum(nodes: Dict[int, Tuple[float, float]], chain: List[int]) -> List[float]:
    """
    Compute cumulative distance along a chain of node ids in *meters*.
    cum[0] = 0, cum[i] = sum_{k < i} d(chain[k], chain[k+1]).
    """
    cum = [0.0]
    for i in range(1, len(chain)):
        a = nodes[chain[i - 1]]
        b = nodes[chain[i]]
        cum.append(cum[-1] + hav_m(a[0], a[1], b[0], b[1]))
    return cum


def nearest_chain_index(
    nodes: Dict[int, Tuple[float, float]],
    chain: List[int],
    lat: float,
    lon: float,
) -> Tuple[int, float]:
    """
    Find index of the chain node nearest to (lat, lon), return (index, distance_m).
    """
    best_i, best_d = None, float("inf")
    for i, nid in enumerate(chain):
        a = nodes[nid]
        d = hav_m(lat, lon, a[0], a[1])
        if d < best_d:
            best_d, best_i = d, i
    return best_i, best_d


# -------------------------
# Mainline builder
# -------------------------

def build_mainline_from_relation(
    RELATION_NAME: str,
    facility: str,
    direction: str,
    initial_node: int,
    initial_way: int,
    final_node: int,
    TARGETS: List[Tuple[float, float]],
    node_id_map: Optional[Dict[int, str]] = None,   # osm_id -> your DB id
    forbid_links: bool = False,
    rel_id: Optional[int] = None,
    strict_endpoints: bool = True,
):
    """
    Build a mainline chain for a route relation and snap a set of target lat/lon points
    onto that chain.

    Returns:
        out_nodes: list of node dicts (one per snapped target)
        out_edges: list of edge dicts (between consecutive snapped nodes)
        meta:      dict with relation/facility metadata

    Distances:
        - chain_m: cumulative meters along the mainline chain (dev-side only)
        - distance_m (edges): meters along the chain between snapped nodes
                              (high-accuracy geodesic sum; suitable as canonical
                              distance in your corridor JSON).
    """
    # 1) fetch relation members
    rel_id, way_ids_ordered, ways, nodes = fetch_relation(RELATION_NAME, rel_id=rel_id)

    # map way -> node sequence (needed for endpoint debugging / normalization)
    way_nodes: Dict[int, List[int]] = {
        wid: w["nodes"]
        for wid, w in ways.items()
        if "nodes" in w
    }

    # find the member way that contains final_node (if any)
    final_way = None
    for wid in way_ids_ordered:
        nlist = way_nodes.get(wid, [])
        if final_node in nlist:
            final_way = wid
            break

    # 2) Endpoint handling
    # ------------------------------------------------------------------
    # STRICT MODE: fail fast if endpoints are not in the relation, so
    # you catch metadata errors early.
    # RELAXED MODE: reuse older "rescue" logic that tries harder to
    # adjust endpoints; useful when debugging weird OSM situations.
    # ------------------------------------------------------------------

    def _first_present_on_way(way_id: int, nodes_dict: Dict[int, Tuple[float, float]], prefer_end: str = "head") -> Optional[int]:
        seq = way_nodes.get(way_id) or []
        it = seq if prefer_end == "head" else reversed(seq)
        for nid in it:
            if nid in nodes_dict:
                return nid
        return None

    def _nearest_present_on_way(way_id: int, nodes_dict: Dict[int, Tuple[float, float]], ref_nid: int) -> Optional[int]:
        seq = way_nodes.get(way_id) or []
        if not seq:
            return None
        if ref_nid in seq:
            i = seq.index(ref_nid)
        else:
            i = len(seq) - 1
        L = len(seq)
        for k in range(L):
            for j in (i - k, i + k):
                if 0 <= j < L and seq[j] in nodes_dict:
                    return seq[j]
        return None

    if strict_endpoints:
        # Hard checks: if something is wrong, raise and fix your metadata.
        if initial_way not in way_ids_ordered:
            raise ValueError(f"initial_way {initial_way} is not a member of relation {rel_id}")
        if initial_node not in nodes:
            raise ValueError(f"initial_node {initial_node} not found in nodes for relation {rel_id}")
        if final_node not in nodes:
            raise ValueError(f"final_node {final_node} not found in nodes for relation {rel_id}")
    else:
        # Relaxed / debug mode: attempt to rescue endpoints.
        if initial_way not in way_ids_ordered:
            # ask Overpass which relations contain this way and pick the first road relation
            q = f'[out:json][timeout:{TIMEOUT}];way({initial_way});rel(bw);out body;'
            j = post_overpass(q)
            rels = [
                e for e in j.get("elements", [])
                if e["type"] == "relation" and e.get("tags", {}).get("route") == "road"
            ]
            if not rels:
                raise RuntimeError(
                    f"initial_way {initial_way} is not in fetched relation and no alternative road relations were found"
                )
            alt_rel_id = rels[0]["id"]
            # refetch with alt_rel_id
            rel_id, way_ids_ordered, ways, nodes = fetch_relation(None, rel_id=alt_rel_id)
            # rebuild way_nodes and final_way
            way_nodes.clear()
            way_nodes.update({
                wid: w["nodes"]
                for wid, w in ways.items()
                if "nodes" in w
            })
            final_way = None
            for wid in way_ids_ordered:
                nlist = way_nodes.get(wid, [])
                if final_node in nlist:
                    final_way = wid
                    break

        # Normalize START
        if initial_node not in nodes:
            if initial_way in way_nodes:
                cand = _first_present_on_way(initial_way, nodes, prefer_end="head")
                if cand is not None:
                    initial_node = cand
            if initial_node not in nodes:
                # very last resort
                initial_node = next(iter(nodes))

        # Normalize GOAL
        if final_node not in nodes:
            if final_way is not None:
                cand = (
                    _nearest_present_on_way(final_way, nodes, ref_nid=final_node)
                    or _first_present_on_way(final_way, nodes, prefer_end="tail")
                )
                if cand is not None:
                    final_node = cand
            if final_node not in nodes:
                # very last resort
                final_node = next(iter(nodes))

    # 3) infer facility number for "off-fac" penalty
    m = re.search(r"\d+", facility or "")
    facility_number = m.group(0) if m else facility

    # 4) build graph (relation-only)
    adj, _ = build_graph_no_bearing(
        way_ids_ordered, ways, nodes,
        facility_number=facility_number,
        forbid_links=forbid_links,
    )

    # 5) Solve mainline path (A*) from initial_node to final_node
    chain = astar(nodes, adj, initial_node, final_node)
    cum = chain_cum(nodes, chain)  # meters

    # 6) Snap TARGETS to chain (monotone, nudge duplicates)
    picks = []
    for (lat, lon) in TARGETS:
        i, d = nearest_chain_index(nodes, chain, lat, lon)
        picks.append({"idx": i, "dist_m": d})

    picks.sort(key=lambda p: p["idx"])
    used: set[int] = set()
    for p in picks:
        i = p["idx"]
        if i not in used:
            used.add(i)
        else:
            # nudge to nearest free neighbor
            for j in [i - 1, i + 1, i - 2, i + 2]:
                if 0 <= j < len(chain) and j not in used:
                    p["idx"] = j
                    used.add(j)
                    break

    # 7) Emit nodes (id blank; osm_id set; seq_local from order; chain_m for ordering/debug)
    out_nodes = []
    for seq_num, p in enumerate(picks):
        idx = p["idx"]
        nid = chain[idx]
        lat, lon = nodes[nid]
        out_nodes.append({
            "id": (node_id_map[nid] if node_id_map and nid in node_id_map else ""),
            "osm_id": int(nid),
            "uid": f"{facility}_{direction}_{seq_num:02d}",
            "facility": facility,
            "direction": direction,
            "role": "main",    # you will edit this later in Excel
            "label": "",       # you will fill this later in Excel
            "lat": lat,
            "lon": lon,
            "chain_m": float(round(cum[idx], 3)),  # dev-side cumulative meters
            "seq_local": seq_num,
        })

    # 8) Emit edges between consecutive picks
    # distance_m = meters along mainline chain between snapped nodes
    out_edges = []

    def lookup(nid: int) -> str:
        return node_id_map[nid] if (node_id_map and nid in node_id_map) else ""

    for i in range(len(picks) - 1):
        a_idx = picks[i]["idx"]
        b_idx = picks[i + 1]["idx"]
        if a_idx > b_idx:
            a_idx, b_idx = b_idx, a_idx

        osm_a = int(chain[a_idx])
        osm_b = int(chain[b_idx])

        dist_m = float(cum[b_idx] - cum[a_idx])  # meters along chain

        edge = {
            "id": f"E_{i:03d}",
            "facility": facility,
            "direction": direction,
            "kind": "mainline",
            "distance_m": dist_m,
            "method": "chain_geodesic",   # sum of small haversines along relation
            "quality": "ok",
            "notes": "",
        }

        if node_id_map:
            edge["from_id"] = lookup(osm_a)
            edge["to_id"] = lookup(osm_b)
        else:
            edge["from_id"] = ""
            edge["to_id"] = ""
            edge["from_osm_id"] = osm_a
            edge["to_osm_id"] = osm_b

        out_edges.append(edge)

    meta = {
        "relation": {"id": int(rel_id), "name": RELATION_NAME},
        "facility": facility,
        "direction": direction,
        "total_chain_nodes": len(chain),
    }

    return out_nodes, out_edges, meta

def as_map(seq: List[Dict[str, Any]], key: str) -> Dict[Any, Dict[str, Any]]:
    """
    Build a dict keyed by `key` from a list of row dicts, skipping blank/None keys.
    """
    return {
        row[key]: row
        for row in seq
        if key in row and row[key] not in ("", None)
    }


# ----------------------------
# 1) JSON IO
# ----------------------------

def load_json(path: str | pathlib.Path) -> Tuple[List[Dict[str, Any]],
                                                List[Dict[str, Any]],
                                                Dict[str, Any]]:
    """
    Load corridor JSON with shape:
        { "nodes": [...], "edges": [...], "meta": {...} }

    Also tolerates legacy "ways" key as edges.
    """
    path = pathlib.Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    nodes = data.get("nodes", [])
    edges = data.get("edges", data.get("ways", []))  # accept either key
    meta = data.get("meta", {})
    return nodes, edges, meta


def save_json(path: str | pathlib.Path,
              nodes: List[Dict[str, Any]],
              edges: List[Dict[str, Any]],
              meta: Optional[Dict[str, Any]] = None) -> None:
    """
    Save corridor JSON in canonical form:
        { "nodes": [...], "edges": [...], "meta": {...} }
    """
    path = pathlib.Path(path)
    payload: Dict[str, Any] = {"nodes": nodes, "edges": edges}
    if meta is not None:
        payload["meta"] = meta
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ----------------------------
# 2) Excel export/import (2 sheets)
# ----------------------------

NODE_COLUMNS_ORDER = [
    "id", "osm_id", "uid", "facility", "direction", "role",
    "label",
    "lat", "lon",
    "chain_m", "seq_local", "notes",
]

EDGE_COLUMNS_ORDER = [
    "id", "from_id", "to_id", "facility", "direction", "kind",
    "distance_m", "method", "quality",
    "osm_way_id",  # optional: keep if you store way ids
    "penalty_m",
    "notes",
]


def export_xlsx(nodes: List[Dict[str, Any]],
                edges: List[Dict[str, Any]],
                out_path: str | pathlib.Path) -> None:
    """
    Export nodes/edges to a 2-sheet Excel file:
      - "Nodes"
      - "Edges"
    Column orders are guided by NODE_COLUMNS_ORDER / EDGE_COLUMNS_ORDER
    but extra columns are preserved.
    """
    out_path = pathlib.Path(out_path)

    def reorder(df: pd.DataFrame, preferred: List[str]) -> pd.DataFrame:
        existing = [c for c in preferred if c in df.columns]
        rest = [c for c in df.columns if c not in existing]
        return df[existing + rest]

    ndf = pd.DataFrame(nodes)
    edf = pd.DataFrame(edges)
    ndf = reorder(ndf, NODE_COLUMNS_ORDER)
    edf = reorder(edf, EDGE_COLUMNS_ORDER)

    # NOTE: make sure XlsxWriter is installed in your environment:
    #   pip install XlsxWriter
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as xw:
        ndf.to_excel(xw, index=False, sheet_name="Nodes")
        edf.to_excel(xw, index=False, sheet_name="Edges")


def import_xlsx(in_path: str | pathlib.Path) -> Tuple[List[Dict[str, Any]],
                                                     List[Dict[str, Any]]]:
    """
    Import nodes/edges from a 2-sheet Excel file ("Nodes" & "Edges").
    NaNs are converted to "" for string fields; numeric columns are coerced.
    """
    in_path = pathlib.Path(in_path)
    xls = pd.ExcelFile(in_path)
    ndf = pd.read_excel(xls, "Nodes").fillna("")
    edf = pd.read_excel(xls, "Edges").fillna("")

    # Coerce numeric-ish node fields
    for col in ("lat", "lon", "chain_m"):
        if col in ndf.columns:
            ndf[col] = pd.to_numeric(ndf[col], errors="coerce")

    # Coerce numeric-ish edge fields
    for col in ("distance_m", "penalty_m"):
        if col in edf.columns:
            edf[col] = pd.to_numeric(edf[col], errors="coerce")

    nodes = ndf.to_dict(orient="records")
    edges = edf.to_dict(orient="records")
    return nodes, edges


# ----------------------------
# 3) Validation (lightweight)
# ----------------------------

def validate(nodes: List[Dict[str, Any]],
             edges: List[Dict[str, Any]]) -> List[str]:
    """
    Run basic consistency checks:
      - node ids present
      - lat/lon present
      - edge from_id / to_id exist as node ids
      - no self-loops
    """
    msgs: List[str] = []
    nmap = as_map(nodes, "id")

    # Nodes
    for i, n in enumerate(nodes):
        node_id = n.get("id", "")
        if not node_id:
            msgs.append(f"Node[{i}] missing id")
        for k in ("lat", "lon"):
            if k not in n or pd.isna(n[k]):
                msgs.append(f"Node[{node_id or i}] missing {k}")

    # Edges
    for j, e in enumerate(edges):
        eid = e.get("id", j)
        a, b = e.get("from_id"), e.get("to_id")
        if a not in nmap:
            msgs.append(f"Edge[{eid}] from_id not found: {a}")
        if b not in nmap:
            msgs.append(f"Edge[{eid}] to_id not found: {b}")
        if a == b and a in nmap:
            msgs.append(f"Edge[{eid}] from_id == to_id ({a})")

    return msgs


# ----------------------------
# 4) Distance filling
# ----------------------------

def fill_haversine_for(edges: List[Dict[str, Any]],
                       nodes: List[Dict[str, Any]],
                       predicate=lambda e: False) -> int:
    """
    Set distance_m via direct haversine for edges where:
        - distance_m is missing/blank, OR
        - predicate(edge) is True.

    Default predicate=False means:
        - Only fill missing distances; do NOT overwrite existing ones.
    If you *want* to override (e.g. recompute all connectors), call with
        predicate=lambda e: e.get("kind") == "connector"
    or similar.

    This avoids clobbering mainline distances already computed via
    chain_geodesic in the relation builder.
    """
    nmap = as_map(nodes, "id")
    changed = 0

    for e in edges:
        need = pd.isna(e.get("distance_m")) or e.get("distance_m") in ("", None)
        if not (need or predicate(e)):
            continue

        a, b = nmap.get(e.get("from_id")), nmap.get(e.get("to_id"))
        if not a or not b:
            continue

        d = hav_m(a["lat"], a["lon"], b["lat"], b["lon"])
        e["distance_m"] = round(float(d), 2)
        if not e.get("method"):
            e["method"] = "haversine"
        if not e.get("quality"):
            e["quality"] = "ok"
        changed += 1

    return changed


def osrm_route_distance(lat1: float, lon1: float,
                        lat2: float, lon2: float,
                        base_url: str = "https://router.project-osrm.org",
                        profile: str = "driving",
                        timeout: float = 15.0,
                        retries: int = 2) -> Optional[float]:
    """
    Returns route distance in meters using OSRM; None on failure.
    """
    url = (
        f"{base_url}/route/v1/{profile}/"
        f"{lon1:.6f},{lat1:.6f};{lon2:.6f},{lat2:.6f}"
        f"?overview=false"
    )
    last_err: Optional[str] = None

    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            routes = j.get("routes") or []
            if routes:
                return float(routes[0]["distance"])
            last_err = f"no routes ({j.get('code')})"
        except Exception as e:
            last_err = str(e)
        time.sleep(0.5 * (attempt + 1))

    # Optional: log last_err somewhere if desired
    return None


