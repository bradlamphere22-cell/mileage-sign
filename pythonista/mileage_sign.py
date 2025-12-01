#!/usr/bin/env python3
# mileage_sign.py
#
# Pythonista/Shortcut-friendly script:
# - Reads trip.json in current directory
# - Takes current lat/lon (argv or input)
# - Picks the correct *forward* node using topology + OSRM
# - Adds OSRM distance from here -> forward node
# - Prints a 3-line mileage sign

import json
import math
import sys
import requests
from typing import Dict, List, Tuple, Any

OSRM_BASE = "https://router.project-osrm.org/route/v1/driving"
M_PER_MILE = 1609.344


# ---------- Helpers: geometry & OSRM ----------

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in meters between two lat/lon points."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def osrm_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Query OSRM for driving distance in meters between two points."""
    url = f"{OSRM_BASE}/{lon1},{lat1};{lon2},{lat2}?overview=false"
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    data = r.json()
    return data["routes"][0]["distance"]


# ---------- Graph utilities ----------

def build_graph(trip: Dict[str, Any]) -> Tuple[
    Dict[Any, Dict[str, Any]],
    Dict[Any, List[Tuple[Any, float]]],
    Dict[Any, List[Tuple[Any, float]]]
]:
    """
    Build:
      - nodes_by_id: id -> node dict
      - adj_fwd: id -> list of (neighbor_id, distance_m)
      - adj_rev: id -> list of (neighbor_id, distance_m) for reverse Dijkstra
    from trip.json structure: trip["nodes"], trip["edges"].
    Expects edges to have "from" and "to" and "distance_m".
    """
    nodes_by_id: Dict[Any, Dict[str, Any]] = {n["id"]: n for n in trip["nodes"]}
    adj_fwd: Dict[Any, List[Tuple[Any, float]]] = {n_id: [] for n_id in nodes_by_id}
    adj_rev: Dict[Any, List[Tuple[Any, float]]] = {n_id: [] for n_id in nodes_by_id}

    for e in trip["edges"]:
        fr = e.get("from") or e.get("from_id")
        to = e.get("to") or e.get("to_id")
        dist = float(e.get("distance_m", 0.0))
        if fr not in nodes_by_id or to not in nodes_by_id:
            continue
        adj_fwd[fr].append((to, dist))
        adj_rev[to].append((fr, dist))

    return nodes_by_id, adj_fwd, adj_rev


def dijkstra_remaining(dest_id: Any,
                       adj_rev: Dict[Any, List[Tuple[Any, float]]]) -> Dict[Any, float]:
    """
    Reverse Dijkstra from dest_id using adj_rev to get shortest-path distance
    *to* dest_id for every node (remaining distance).
    """
    import heapq

    remaining: Dict[Any, float] = {}
    heap: List[Tuple[float, Any]] = [(0.0, dest_id)]

    while heap:
        d, u = heapq.heappop(heap)
        if u in remaining:
            continue
        remaining[u] = d
        for v, w in adj_rev.get(u, []):
            if v not in remaining:
                heapq.heappush(heap, (d + w, v))

    return remaining


def dijkstra_path(start_id: Any,
                  dest_id: Any,
                  adj_fwd: Dict[Any, List[Tuple[Any, float]]]) -> Tuple[List[Any], List[float]]:
    """
    Dijkstra to get *one* shortest path from start_id to dest_id.
    Returns (path_node_ids, edge_lengths_m in that order).
    """
    import heapq

    dist: Dict[Any, float] = {}
    prev: Dict[Any, Any] = {}
    heap: List[Tuple[float, Any]] = [(0.0, start_id)]

    while heap:
        d, u = heapq.heappop(heap)
        if u in dist:
            continue
        dist[u] = d
        if u == dest_id:
            break
        for v, w in adj_fwd.get(u, []):
            if v not in dist:
                heapq.heappush(heap, (d + w, v))
                if v not in prev or d + w < dist.get(v, float("inf")):
                    prev[v] = u

    if dest_id not in dist:
        # No path; return just the start
        return [start_id], []

    # Reconstruct path
    path_nodes: List[Any] = []
    cur = dest_id
    while True:
        path_nodes.append(cur)
        if cur == start_id:
            break
        cur = prev[cur]
    path_nodes.reverse()

    # Build edge list (lengths) along the path
    edge_lengths: List[float] = []
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i + 1]
        length = None
        for (nbr, w) in adj_fwd[u]:
            if nbr == v:
                length = w
                break
        if length is None:
            length = 0.0
        edge_lengths.append(length)

    return path_nodes, edge_lengths


# ---------- Here-node selection ----------

def build_remaining_distance_map(dest_node_id: Any,
                                 adj_rev: Dict[Any, List[Tuple[Any, float]]]
                                 ) -> Dict[Any, float]:
    """Wrapper to compute remaining distance from every node to dest_node_id."""
    return dijkstra_remaining(dest_node_id, adj_rev)


def pick_here_node(lat_here: float,
                   lon_here: float,
                   nodes_by_id: Dict[Any, Dict[str, Any]],
                   remaining: Dict[Any, float],
                   dest_node_id: Any) -> Any:
    """
    Global forward/behind test:

      - D_here_dest = OSRM(here -> dest_node_id).
      - remaining[n] = corridor distance from n -> dest_node_id.
      - If remaining[n] > D_here_dest, node n is behind us.
      - Among nodes with remaining[n] <= D_here_dest (+ tol),
        pick the one with max remaining[n] (closest forward node).
    """
    dest_node = nodes_by_id[dest_node_id]

    # Distance from here to corridor destination along real roads
    try:
        d_here_dest = osrm_distance_m(lat_here, lon_here,
                                      dest_node["lat"], dest_node["lon"])
    except Exception:
        d_here_dest = haversine(lat_here, lon_here,
                                dest_node["lat"], dest_node["lon"])

    tol = 1000.0  # meters of tolerance

    forward_candidates: List[Tuple[float, Any]] = []
    max_remaining = -1.0
    max_remaining_node = None

    for nid, rem in remaining.items():
        # Track global max remaining for fallback
        if rem > max_remaining:
            max_remaining = rem
            max_remaining_node = nid

        # Forward if corridor remaining <= OSRM distance (plus tolerance)
        if rem <= d_here_dest + tol:
            forward_candidates.append((rem, nid))

    if forward_candidates:
        # Node with largest remaining among the forward ones = closest forward node
        rem_max, nid_best = max(forward_candidates, key=lambda x: x[0])
        return nid_best

    # If OSRM thinks we're before all nodes, fall back to start node
    return max_remaining_node


# ---------- Sign selection logic ----------

def build_mileage_sign(trip: Dict[str, Any],
                       nodes_by_id: Dict[Any, Dict[str, Any]],
                       adj_fwd: Dict[Any, List[Tuple[Any, float]]],
                       here_id: Any,
                       dest_node_id: Any,
                       here_offset_m: float) -> List[Tuple[str, float]]:
    """
    Build up to three sign lines as (label, miles).
    Uses:
      - path from here_id to dest_node_id
      - visible nodes (role != 'invisible')
    Destination distance includes:
      - OSRM(here -> here_node) offset (here_offset_m),
      - corridor path here_node -> dest_node,
      - final access hop if stored in edges,
        OR OSRM(here->dest) fallback if we have dest lat/lon.
    """
    # Path and cumulative corridor distances starting at here_node
    path_nodes, edge_lengths = dijkstra_path(here_id, dest_node_id, adj_fwd)

    cum_m: List[float] = [0.0]
    for w in edge_lengths:
        cum_m.append(cum_m[-1] + w)

    # Map node_id -> distance from *here* (include the OSRM offset)
    dist_from_here: Dict[Any, float] = {}
    for idx, nid in enumerate(path_nodes):
        dist_from_here[nid] = here_offset_m + cum_m[idx]

    # Identify visible nodes along path
    visible_nodes = []
    for idx, nid in enumerate(path_nodes):
        node = nodes_by_id[nid]
        if node.get("role", "invisible") == "invisible":
            continue
        # skip the starting node if it's effectively here
        if nid == here_id and dist_from_here[nid] < 50.0:
            continue
        # also skip anything so close it would round to 0 miles (< ~0.25 mi)
        if dist_from_here[nid] < 0.25 * M_PER_MILE:
            continue
        visible_nodes.append((nid, node))

    # Destination label / optional lat,lon
    dest_label = None
    dest_lat = dest_lon = None

    dest_value = trip.get("destination")
    if isinstance(dest_value, dict):
        dest_label = dest_value.get("label")
        dest_lat = dest_value.get("lat")
        dest_lon = dest_value.get("lon")
    elif isinstance(dest_value, str):
        dest_label = trip.get("destination_label", dest_value)

    # Base distance from here to final corridor node
    base_dest_m = dist_from_here.get(dest_node_id, here_offset_m)

    # Last-hop "access" edge if present
    extra_hop_m = 0.0
    dest_id_value = trip.get("destination")
    if dest_id_value is not None:
        for e in trip.get("edges", []):
            fr = e.get("from") or e.get("from_id")
            to = e.get("to") or e.get("to_id")
            if fr == dest_node_id and to == dest_id_value:
                extra_hop_m = float(e.get("distance_m", 0.0))
                break
            if to == dest_node_id and fr == dest_id_value:
                extra_hop_m = float(e.get("distance_m", 0.0))
                break

    # If no access edge and we have lat/lon, fall back to OSRM/haversine from final node
    if extra_hop_m == 0.0 and dest_lat is not None and dest_lon is not None:
        dest_node = nodes_by_id[dest_node_id]
        try:
            extra_hop_m = osrm_distance_m(dest_node["lat"], dest_node["lon"],
                                          dest_lat, dest_lon)
        except Exception:
            extra_hop_m = haversine(dest_node["lat"], dest_node["lon"],
                                    dest_lat, dest_lon)

    dest_total_m = base_dest_m + extra_hop_m
    dest_miles = dest_total_m / M_PER_MILE

    lines: List[Tuple[str, float]] = []

    if not visible_nodes:
        if dest_label:
            lines.append((dest_label, dest_miles))
        return lines

    if len(visible_nodes) == 1:
        nid, node = visible_nodes[0]
        miles = dist_from_here[nid] / M_PER_MILE
        lines.append((node.get("label", str(nid)), miles))
        if dest_label:
            lines.append((dest_label, dest_miles))
        return lines

    # 2+ visible nodes: first two hubs, then destination
    first_id, first_node = visible_nodes[0]
    second_id, second_node = visible_nodes[1]

    first_miles = dist_from_here[first_id] / M_PER_MILE
    second_miles = dist_from_here[second_id] / M_PER_MILE

    lines.append((first_node.get("label", str(first_id)), first_miles))
    lines.append((second_node.get("label", str(second_id)), second_miles))

    if dest_label:
        lines.append((dest_label, dest_miles))

    return lines


# ---------- Main entrypoint ----------

def main():
    # Load trip.json
    with open("trip.json") as f:
        trip = json.load(f)

    nodes_by_id, adj_fwd, adj_rev = build_graph(trip)

    # Destination corridor node: pick sink (no outgoing edges) if possible
    sink_candidates = [nid for nid, nbrs in adj_fwd.items() if len(nbrs) == 0]
    if sink_candidates:
        dest_node_id = sink_candidates[0]
    else:
        dest_node_id = trip["nodes"][-1]["id"]

    # Get current location: argv or input()
    if len(sys.argv) >= 3:
        lat_here = float(sys.argv[1])
        lon_here = float(sys.argv[2])
    else:
        lat_here = float(input("Latitude (e.g. 37.2898): ").strip())
        lon_here = float(input("Longitude (e.g. -121.9436): ").strip())

    # Remaining distance map (to dest_node_id)
    remaining = build_remaining_distance_map(dest_node_id, adj_rev)

    # Pick here-node using global forward/behind classification
    here_id = pick_here_node(lat_here, lon_here, nodes_by_id, remaining, dest_node_id)

    # Distance from actual here to that corridor node
    here_node = nodes_by_id[here_id]
    try:
        here_offset_m = osrm_distance_m(lat_here, lon_here,
                                        here_node["lat"], here_node["lon"])
    except Exception:
        here_offset_m = haversine(lat_here, lon_here,
                                  here_node["lat"], here_node["lon"])

    # Build sign
    lines = build_mileage_sign(trip, nodes_by_id, adj_fwd,
                               here_id, dest_node_id, here_offset_m)

    # Print in mileage-sign style
    print("\n--- Mileage Sign ---")
    for label, miles in lines:
        miles_rounded = int(round(miles))
        if miles_rounded < 1:
            miles_rounded = 1
        print(f"{label} {miles_rounded}")
    print("--------------------\n")


if __name__ == "__main__":
    main()
