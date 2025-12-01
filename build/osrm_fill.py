def fill_osrm_for_missing(edges: List[Dict[str, Any]],
                          nodes: List[Dict[str, Any]],
                          base_url: str = "https://router.project-osrm.org",
                          target_kinds=("connector", "access", "turn"),
                          overwrite: bool = False) -> int:
    """
    For edges with missing distance_m (or overwrite=True) and kind in target_kinds,
    call OSRM and fill distance_m, set method='osrm'.

    This is usually for non-mainline pieces: connectors, access edges, turn arcs, etc.
    """
    nmap = as_map(nodes, "id")
    changed = 0

    for e in edges:
        kind = (e.get("kind") or "").lower()
        if kind not in target_kinds:
            continue

        missing = pd.isna(e.get("distance_m")) or e.get("distance_m") in ("", None)
        if not (missing or overwrite):
            continue

        a, b = nmap.get(e.get("from_id")), nmap.get(e.get("to_id"))
        if not a or not b:
            continue

        d = osrm_route_distance(
            a["lat"], a["lon"], b["lat"], b["lon"],
            base_url=base_url
        )
        if d is None:
            # keep previous value if any; mark suspect
            e["quality"] = "suspect"
            if not e.get("method"):
                e["method"] = "unknown"
            continue

        e["distance_m"] = round(float(d), 2)
        e["method"] = "osrm"
        if not e.get("quality"):
            e["quality"] = "ok"
        changed += 1

    return changed

