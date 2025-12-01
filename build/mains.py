RELATION_NAME = "280"     # or whatever the OSM route name is
rel_id = 102127
facility = "I 280"
direction = "SB"

initial_node = 65355354 # your OSM node at start
initial_way  = 30454437 # OSM way id that contains initial_node
final_node   = 246295741 # OSM node at end

TARGETS = [
    (37.774612, -122.396434),
    (37.737897, -122.401740),
    (37.735316, -122.406845),
    (37.681325, -122.471826),
    (37.677818, -122.471576),
    (37.630212, -122.434854),
    (37.627651, -122.431488),
    (37.528352, -122.359881),
    (37.510451, -122.344529),
    (37.507034, -122.338926),
    (37.438593, -122.247328),
    (37.435282, -122.243384),
    (37.331018, -122.062384),
    (37.332333, -122.055702),   
    (37.316695, -121.948933),
    (37.317227, -121.940260),
    (37.320672, -121.900405),
    (37.323762, -121.892207),
    (37.335944, -121.857745),
    (37.339556, -121.851883)
]
# Optional: if you already have DB ids for some nodes:
node_id_map = {
    # osm_id : "N00123",
}



nodes, edges, meta = build_mainline_from_relation(
    RELATION_NAME, facility, direction,
    initial_node, initial_way, final_node,
    TARGETS,
    node_id_map=node_id_map,   # or None
    forbid_links=False,         # set True to forbid *_link ways entirely
    rel_id=rel_id
)

# Now export via your existing save/export helpers:
# save_json("/home/john/mileage_sign.json", nodes, edges, meta)
export_xlsx(nodes, edges, "/mnt/c/Users/John/Documents/Personal/Map_Project/mileage_sign_add.xlsx")
