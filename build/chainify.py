import json
import math
import heapq
from collections import deque

MASTER_FILE = "mileage_graph/corridor_master.json"
DESTS_FILE  = "mileage_graph/user_destinations.json"
TRIP_FILE   = "mileage_graph/trip.json"


# -----------------------
# Basic IO + geometry
# -----------------------

def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters between two lat/lon points."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# -----------------------
# Corridor parsing
# -----------------------

def parse_corridors(corridors_raw):
    """
    corridors_raw = list of strings like:
      'CA 17', 'I 280 NB', 'US 101 NB'
    Returns list of (facility, direction_or_None).
    """
    specs = []
    for c in corridors_raw:
        c = c.strip()
        if not c:
            continue
        parts = c.split()
        if parts[-1] in ("NB", "SB", "EB", "WB"):
            direction = parts[-1]
            facility = " ".join(parts[:-1])
        else:
            facility = c
            direction = None
        specs.append((facility, direction))
    return specs


def corridor_matches_node(specs, node):
    """
    Return True if node's (facility, direction) matches any corridor spec.
    Each spec is (facility, direction_or_None).
    """
    fac = (node.get("facility") or "").strip()
    direc = (node.get("direction") or "").strip()
    if not fac:
        return False
    for f, d in specs:
        if fac != f:
            continue
        if d is None or direc == d:
            return True
    return False


# -----------------------
# Graph builders
# -----------------------

def build_path_graph(master):
    """
    Build nodes + adjacency for corridor pathfinding.
    Uses mainline/trunk/turn edges from the master.
    Normalizes edges to have 'from'/'to' fields.
    """
    nodes = {n["id"]: n for n in master["nodes"]}
    adj = {nid: [] for nid in nodes}

    for e in master["edges"]:
        kind = e.get("kind")
        if kind not in ("mainline", "trunk", "turn"):
            continue

        u = e.get("from") or e.get("from_id")
        v = e.get("to")   or e.get("to_id")
        if u not in nodes or v not in nodes:
            continue

        e2 = dict(e)
        e2["from"] = u
        e2["to"]   = v
        adj[u].append(e2)

    return nodes, adj


def build_master_mainline_graph(master):
    """
    Build nodes + adjacency for mainline/trunk edges ONLY.
    Returns:
      - nodes
      - fwd_adj: u -> list of edges (u -> v)
      - rev_adj: v -> list of edges (u -> v)
    Used to walk forward and backward along a corridor.
    """
    nodes = {n["id"]: n for n in master["nodes"]}
    fwd = {nid: [] for nid in nodes}
    rev = {nid: [] for nid in nodes}

    for e in master["edges"]:
        kind = e.get("kind")
        if kind not in ("mainline", "trunk"):
            continue

        u = e["from_id"]
        v = e["to_id"]
        if u not in nodes or v not in nodes:
            continue

        fwd[u].append(e)
        rev[v].append(e)

    return nodes, fwd, rev


# -----------------------
# Pre-node / corridor helpers
# -----------------------

def is_pre_node(n):
    """
    Decide if a node is a pre-node:
      - role == 'invisible'
      - OR label ends with 'PN' (to catch your naming convention)
    """
    role = (n.get("role") or "").lower()
    if role == "invisible":
        return True
    label = (n.get("label") or "").upper()
    if label.endswith(" PN") or label.endswith("_PN") or label.endswith("PN"):
        return True
    return False


def _nearest_pre_node_for_anchor(anchor, specs, master_nodes):
    """
    Internal helper for ORIGIN pre-node: given an anchor and corridor specs,
    return nearest pre-node (by haversine distance) restricted to those
    (facility, direction) specs if specs is not empty.
    Returns (best_node, best_dist) or (None, None).
    """
    alat = anchor["lat"]
    alon = anchor["lon"]

    best_node = None
    best_dist = None

    for nid, n in master_nodes.items():
        if not is_pre_node(n):
            continue

        if specs and not corridor_matches_node(specs, n):
            continue

        nlat = n.get("lat")
        nlon = n.get("lon")
        if nlat is None or nlon is None:
            continue

        d = haversine_m(alat, alon, nlat, nlon)
        if best_node is None or d < best_dist:
            best_node = n
            best_dist = d

    return best_node, best_dist


def find_origin_pre_node(anchor, corridor_specs, master_nodes):
    """
    Origin pre-node: nearest pre-node by straight-line distance,
    restricted to the given (facility, direction) specs (if non-empty).
    """
    best_node, best_dist = _nearest_pre_node_for_anchor(anchor, corridor_specs, master_nodes)

    if best_node is None and corridor_specs:
        print("WARNING (origin): no pre-node found on requested corridors; "
              "falling back to all corridors.")
        best_node, best_dist = _nearest_pre_node_for_anchor(anchor, [], master_nodes)

    if best_node is None:
        raise RuntimeError("No suitable origin pre-node found for anchor.")

    return {"node": best_node, "dist": best_dist}


def find_dest_pre_node(anchor, corridor_specs, master_nodes, fwd_adj, rev_adj):
    """
    Destination pre-node:
      1) Find the nearest node ON THE ALLOWED CORRIDORS (any role).
      2) Walk BACKWARDS along mainline/trunk edges on the SAME facility+direction
         until we hit the nearest pre-node upstream.
    The pre-node's distance from the anchor is straight-line (for the access edge).
    """
    alat = anchor["lat"]
    alon = anchor["lon"]

    # Step 1: nearest corridor node (any role) on allowed specs
    best_node = None
    best_dist = None

    for nid, n in master_nodes.items():
        if corridor_specs and not corridor_matches_node(corridor_specs, n):
            continue

        nlat = n.get("lat")
        nlon = n.get("lon")
        if nlat is None or nlon is None:
            continue

        d = haversine_m(alat, alon, nlat, nlon)
        if best_node is None or d < best_dist:
            best_node = n
            best_dist = d
            print("updating best node", n, "\n distance", d)

    if best_node is None and corridor_specs:
        print("WARNING (dest): no node found on requested corridors; "
              "falling back to all corridors.")
        for nid, n in master_nodes.items():
            nlat = n.get("lat")
            nlon = n.get("lon")
            if nlat is None or nlon is None:
                continue
            d = haversine_m(alat, alon, nlat, nlon)
            if best_node is None or d < best_dist:
                best_node = n
                best_dist = d

    if best_node is None:
        raise RuntimeError("No suitable destination corridor node found for anchor.")

    # Step 2: walk backwards along same facility+direction to find upstream pre-node
    target_id = best_node["id"]
    fac = best_node.get("facility")
    direc = best_node.get("direction")

    INF = 1e18
    dist_corridor = {target_id: 0.0}
    q = deque([target_id])
    best_pre = None
    best_pre_corridor_d = None

    while q:
        u = q.popleft()
        for e in rev_adj.get(u, []):
            p = e["from_id"]
            nu = master_nodes[u]
            npn = master_nodes[p]

            # same facility+direction only
            if nu.get("facility") != fac or nu.get("direction") != direc:
                continue
            if npn.get("facility") != fac or npn.get("direction") != direc:
                continue

            d = dist_corridor[u] + e.get("distance_m", 0.0)
            if d < dist_corridor.get(p, INF):
                dist_corridor[p] = d
                if is_pre_node(npn):
                    best_pre = npn
                    best_pre_corridor_d = d
                    q.clear()
                    break
                q.append(p)

    if best_pre is None:
        # fallback: use nearest pre-node by straight-line distance, like origin
        print("WARNING (dest): no upstream pre-node found along corridor; "
              "falling back to nearest pre-node by distance.")
        best_pre, _ = _nearest_pre_node_for_anchor(anchor, corridor_specs, master_nodes)
        if best_pre is None:
            raise RuntimeError("No suitable destination pre-node found (even fallback).")

    # Distance for access edge is straight-line anchor -> pre-node
    pre_lat = best_pre.get("lat")
    pre_lon = best_pre.get("lon")
    access_d = haversine_m(alat, alon, pre_lat, pre_lon)

    return {"node": best_pre, "dist": access_d}


# -----------------------
# Shortest path along corridor graph
# -----------------------

def dijkstra_path(start_id, goal_id, nodes, adj):
    """
    Dijkstra from start_id to goal_id over adj.
    adj[u] is a list of edges with 'to' and 'distance_m'.
    Returns the list of edges (in forward order) along the shortest path.
    """
    INF = 1e18
    dist = {start_id: 0.0}
    parent_edge = {}
    visited = set()
    heap = [(0.0, start_id)]

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u == goal_id:
            break

        for e in adj.get(u, []):
            v = e["to"]
            w = e.get("distance_m", 0.0)
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                parent_edge[v] = e
                heapq.heappush(heap, (nd, v))

    if goal_id not in dist:
        raise RuntimeError(f"No path found from {start_id} to {goal_id}")

    # Reconstruct path edges
    path_edges = []
    cur = goal_id
    while cur != start_id:
        e = parent_edge[cur]
        path_edges.append(e)
        cur = e["from"]

    path_edges.reverse()
    return path_edges


# -----------------------
# Stub helper
# -----------------------

def find_next_visible_on_same_corridor(start_id, master_nodes, fwd_adj):
    """
    Starting from start_id (a pre-node) on some facility+direction,
    walk forward along mainline/trunk edges until we find the first
    node whose role != 'invisible' on the SAME facility+direction.
    Return (node_id, distance_m) where distance_m is the sum of
    distance_m along mainline/trunk edges from start_id to that node.
    If none found, return (None, None).
    """
    if start_id not in master_nodes:
        return None, None

    fac = master_nodes[start_id].get("facility")
    direc = master_nodes[start_id].get("direction")

    INF = 1e18
    dist = {start_id: 0.0}
    q = deque([start_id])

    while q:
        u = q.popleft()
        for e in fwd_adj.get(u, []):
            v = e["to_id"]
            nu = master_nodes[u]
            nv = master_nodes[v]

            # same facility+direction only
            if nu.get("facility") != fac or nu.get("direction") != direc:
                continue
            if nv.get("facility") != fac or nv.get("direction") != direc:
                continue

            d = dist[u] + e.get("distance_m", 0.0)
            if d < dist.get(v, INF):
                dist[v] = d
                if nv.get("role") != "invisible":
                    return v, d
                q.append(v)

    return None, None


# -----------------------
# Main chanify logic
# -----------------------

def chanify(origin_id, dest_id, corridors_raw):
    """
    Build a trip.json connecting origin_id -> dest_id along the given
    corridors (each 'corridor' is a facility with optional direction,
    e.g. 'CA 17', 'I 280 NB', 'US 101 NB').
    Uses:
      - corridor_master.json
      - destinations.json
    Injects 'stub' edges at each 'turn' edge so that the next visible node
    on the old facility can be used in mileage-sign logic.
    """

    master = load_json(MASTER_FILE)
    dests  = load_json(DESTS_FILE)

    corridor_specs = parse_corridors(corridors_raw)

    # Graphs
    path_nodes, path_adj = build_path_graph(master)
    master_nodes, master_fwd, master_rev = build_master_mainline_graph(master)

    # Destination map
    dest_map = {d["id"]: d for d in dests["destinations"]}
    origin_anchor = dest_map[origin_id]["anchor"]
    dest_anchor   = dest_map[dest_id]["anchor"]

    # Origin pre-node: nearest pre-node by distance (corridor_specs filtered)
    best_o = find_origin_pre_node(origin_anchor, corridor_specs, master_nodes)

    # Destination pre-node: nearest corridor node, then walk back along corridor
    best_d = find_dest_pre_node(dest_anchor, corridor_specs, master_nodes, master_fwd, master_rev)

    origin_pre_id = best_o["node"]["id"]
    dest_pre_id   = best_d["node"]["id"]

    print(f"Origin pre-node: {origin_pre_id} ({best_o['node'].get('facility')} {best_o['node'].get('direction')}, dist {best_o['dist']:.1f} m)")
    print(f"Dest   pre-node: {dest_pre_id} ({best_d['node'].get('facility')} {best_d['node'].get('direction')}, dist {best_d['dist']:.1f} m)")

    # Path along corridor graph
    corr_edges = dijkstra_path(origin_pre_id, dest_pre_id, path_nodes, path_adj)
    print(f"Corridor path has {len(corr_edges)} edges.")

    # Build trip structure
    trip = {
        "origin": origin_id,
        "destination": dest_id,
        "destination_label": dest_map[dest_id].get("label", dest_id),
        "nodes": [],
        "edges": []
    }

    used_node_ids = set()

    # 1) Origin access edge (personal -> origin_pre)
    trip["edges"].append({
        "kind": "access",
        "from": origin_id,
        "to": origin_pre_id,
        "distance_m": best_o["dist"],
    })
    used_node_ids.add(origin_id)
    used_node_ids.add(origin_pre_id)

    # 2) Corridor edges + stub edges at turns
    for e in corr_edges:
        u = e["from"]
        v = e["to"]

        e_copy = dict(e)
        e_copy["from"] = u
        e_copy["to"]   = v

        trip["edges"].append(e_copy)
        used_node_ids.add(u)
        used_node_ids.add(v)

        # If this edge is a turn, inject stub to next visible node on *this* corridor
        if e_copy.get("kind") == "turn":
            pre_id = u  # pre-node on old facility

            stub_target_id, stub_dist = find_next_visible_on_same_corridor(
                pre_id, master_nodes, master_fwd
            )

            if stub_target_id is not None and stub_dist is not None:
                stub_edge = {
                    "kind": "stub",
                    "from": pre_id,
                    "to":   stub_target_id,
                    "distance_m": stub_dist,
                    "method": "stub",
                    "notes": "sign stub at corridor turn",
                }
                trip["edges"].append(stub_edge)
                used_node_ids.add(pre_id)
                used_node_ids.add(stub_target_id)
                print(f"  Added stub from {pre_id} to {stub_target_id} ({stub_dist:.1f} m)")

    # 3) Destination access edge (dest_pre -> personal)
    trip["edges"].append({
        "kind": "access",
        "from": dest_pre_id,
        "to": dest_id,
        "distance_m": best_d["dist"],
    })
    used_node_ids.add(dest_pre_id)
    used_node_ids.add(dest_id)

    # 4) trip["nodes"]: corridor nodes only (you can also add D_* if you like)
    for nid in used_node_ids:
        if nid in master_nodes:
            trip["nodes"].append(master_nodes[nid])
        # else it's a D_* personal id, which signage doesn't need as a node

    save_json(TRIP_FILE, trip)
    print(f"\nWrote {TRIP_FILE} with {len(trip['nodes'])} nodes and {len(trip['edges'])} edges.")

    return trip


# -----------------------
# CLI entry point
# -----------------------

def main():
    print("\n=== Chanify: build trip.json from corridor_master + destinations ===\n")
    origin_id = input("Origin destination id (e.g. D_CAMPBELL_COTTAGE): ").strip()
    dest_id   = input("Dest   destination id (e.g. D_MARRIOTT_SAN_FRANCISCO_NABC): ").strip()
    corr_str  = input("Corridors (e.g. 'CA 17 NB,I 280 NB,US 101 NB'): ").strip()

    corridors = [c.strip() for c in corr_str.split(",") if c.strip()]
    print(f"\nUsing corridors: {corridors}\n")

    trip = chanify(origin_id, dest_id, corridors)

    print("\nDone. Inspect trip.json to verify the edges and distances.\n")


if __name__ == "__main__":
    main()
