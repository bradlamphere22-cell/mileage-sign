# Mileage Sign

A small routing engine that mimics freeway mileage signs.

The idea: given a **corridor** (e.g. `CA-17 NB`) and your current position, the code chooses three lines like a real mileage sign:

1. Next local destination (nearest *tertiary* ahead),
2. Next larger town or junction (nearest *secondary/primary* ahead),
3. Final trip destination.

Distances are along the mainline only, rounded to whole miles.

---

## Data model: master vs personal

There are two types of JSON database files:

1. **Master file** with all universal shared nodes and edges (`corridor_master.json`).
2. **Personal destination file** with user-specific locations (`personal.json` or similar).

A trip can be:

1. Between two nodes in the master file.
2. Between one personal node and one master node.
3. Between two personal nodes.

For personal files we do **not** store edges, only node coordinates. During the **chainify** stage, personal nodes are attached to a nearby node in the master file, depending on the trip direction.

Eventually, when this is wrapped in an app, there will be a phone-based chainify that creates a node-and-edge corridor JSON specific to a trip between destinations. You call chainify once to create the travel corridor (`trip.json`). While driving, Pythonista’s `mileage_sign.py` runs using the phone’s lat/lon and that specific `trip.json` to generate the live mileage sign.

---

## Master corridor structure

The master corridor file (`corridor_master.json`) includes:

- **Node tags**:
  - `osm` — OpenStreetMap node identifier,
  - a local node identifier,
  - latitude, longitude,
  - `facility` — e.g. `CA-17`, `I-5`,
  - `type` — `mainline` (divided facilities like freeways) or `trunk` (undivided roads),
  - `label` — name to display on mileage signs,
  - `role` — `primary`, `secondary`, `tertiary`, or `invisible`.

On divided carriageways we place one or more invisible **prenodes** before all exits (and at least one after all access points) to handle:

1. Turns between facilities.
2. Attachments to/from personal nodes.

There is stub logic to include a node in a mileage sign if the trip turns at that node but the exit is prior to the actual interchange.

The master JSON also includes **edges** with:

- a unique local edge label,
- the two nodes the edge connects,
- the distance in meters.

On mainlines, the distance is calculated and loaded when the mainline nodes are loaded. Currently turns are identified and handled manually by inserting an edge with no distance and calling **OSRM (Open Source Routing Machine)** to determine the driving distance. Typically that edge connects a prenode on one facility to a prenode on the next facility and is direction-specific.

---

## Status

This repository currently contains:

- A skeleton **Pythonista** runtime for iPhone / CarPlay (`pythonista/`),
- Build scripts for the corridor graph (`build/`),
- Documentation for the corridor / node / edge model (`docs/`),
- Example data stubs (`data/`).

The real per-user data (like “home” coordinates) are **not** stored here. They live in a local `personal.json` file on-device and are ignored by Git.

---

## Layout

```text
mileage-sign/
  data/
    corridor_master.json        # master file of universal shared nodes
    trip.json                   # example trip-specific corridor DB
    personal_template.json      # user-specific destinations template (no real data)

  pythonista/
    mileage_sign.py             # main runtime client; may embed OSRM calls
    osrm_client.py              # OSRM HTTP helper (if we unbundle it later)

  build/
    build_mainline_from_relation.py  # primitive build of universal shared nodes (to be replaced by add_junction_bundle.py)
    chainify.py                 # build trip/corridor files; likely to migrate to on-phone use
    osrm_fill.py                # fill edge distances from OSRM
    add_junction_bundle.py      # add a junction + edges in one call

  docs/
    architecture.md             # description of nodes, edges, roles

  tests/
    sample_trip.json            # small sample trip

