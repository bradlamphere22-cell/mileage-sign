# Mileage Sign Project --- Development Path Outline

## 1. Corridor Facility Model

-   Treat each highway (e.g., CA‑17, I‑280, US‑101) as a **facility
    spine** with:
    -   Ordered OSM geometry.
    -   Direction views (NB/SB or EB/WB).
    -   A shared list of conceptual hubs (labeled human-style:
        "Shoreline", "Crystal Springs", "Harris Ranch").

## 2. Iterative Crawler (Semantic Layer)

-   Walks along a facility spine.
-   Detects candidate hubs (intersections with numbered highways or
    named interchanges).
-   Interactive prompts:
    -   Yes/No for taking each candidate as a visible hub.
    -   If "Yes": prompt for custom `label`, `role`
        (primary/secondary/tertiary), special flags.
-   Manual add mode for non-numbered but conceptually important hubs
    (e.g., Shoreline).
-   Records results in a **targets file** (CSV/JSON).

## 3. Merge vs Interchange Detection

-   Topological classification of ways touching the facility:
    -   **Interchange**: two-way linkage.
    -   **Merge**: B→A only.
-   For merge-only numbered highways:
    -   Identify nearest "proxy interchange" on A in the appropriate
        direction.
    -   Allow user confirmation to name this hub.

## 4. Prenode Logic

-   Each hub may require 0/1/2 prenodes **per direction**:
    -   `pre_turn`: Last chance to exit to facility B (off‑ramp seen
        upstream).
    -   `pre_attach`: Final collector/on‑ramp merge before the hub.
-   Use **ordering** along facility:
    -   if last off‑ramp \< last merge → need both prenodes.
    -   else → one prenode is sufficient.

## 5. Node Snapper (Geometric Layer)

-   Takes targets file + spine geometry.
-   Snaps hubs to nearest OSM node on correct carriageway.
-   Generates prenodes based on topology.
-   Builds mainline edges (ordered).
-   Identifies candidate turn edges between facilities.
-   Exports **nodes** and **edges** to Excel for inspection.

## 6. Trip Builder / Chainify

-   Uses master corridor data with:
    -   directional nodes,
    -   prenodes (turn/attach),
    -   turn edges,
    -   user destinations (snapped to `pre_attach`).
-   Builds complete trip.json from:
    -   origin,
    -   destination,
    -   via nodes (optional),
    -   inferred corridors.

## 7. Mileage Sign Engine

-   Finds correct "here" node using two-candidate OSRM comparison.
-   Uses stub turn nodes for signage (short segment after prenode, but
    not used for long-distance routing).
-   Correct ordering logic:
    -   Line 1: nearest visible node (tertiary).
    -   Line 2: nearest secondary (or next tertiary).
    -   Line 3: destination or nearest primary.
-   Destination distance = last mainline node distance + OSRM final hop.

## 8. Regionalization Plan (Future)

-   Partition US network into \~100-node conceptual regions by hub label
    uniqueness.
-   Adjacent regions share boundary edges.
-   Master graph is assembled dynamically (lazy loading) for chainify.

## 9. Long-Term Automation

-   Auto-crawler for large-scale corridor additions.
-   Auto-turn inference between facilities.
-   OSRM batch fill for missing distances.
-   Tooling for visualizing regional hub graphs.
