# mileage-sign

\# Mileage Sign



A small routing engine that mimics freeway mileage signs.



The idea: given a \*\*corridor\*\* (e.g. CA-17 NB) and your current position,

the code chooses three lines like a real mileage sign:



1\. Next local destination (nearest \*tertiary\* ahead),

2\. Next larger town or junction (nearest \*secondary/primary\* ahead),

3\. Final trip destination.



Distances are along the mainline only, rounded to whole miles.



There are two types of JSON database files: 1) Master file with all universal shared nodes, and 2) Personal destination file.

A "trip" can be between 1) two nodes in Master file, 2) a personal origin/destination with the other origin/destination

from the personal file, or 3) an origin and destination from the personal file. With personal files we do not store edges, 

simply node coordinates and attach to a nearby node in the Master file during chainify stage, depending on trip direction. 

Eventual structure is to use app, there will be a phone-based chainify that creates a node and edge cooridor json specific

to a trip between destinations. Call the chainify once to create the travel cooridor. While driving, calling Pythonista

mileage\_sign.py generates the mileage sign with inputs of lat, lon from the phone pulling that data, using the specific 

trip.json. 



The tags in corridor\_master.json include osm (open street maps) node identifiers, a local node identifier, latitude, longitude,

facility, type (mainline = facilities like freeways with divided carriageways, trunk = road without divided carriageways),

label (name to display on mileage sign), and role (primary, secondary, tertiary, or invisible). On divided carriageway we place

one or more invisible prenodes before all exits (and at least one after all access points) to handle 1) turns, and 2) attachment

to/from personal nodes. There is stub logic to include a node in a mileage sign if the trip turns at that node but the

exit is prior to the actual interchange. The master json also includes edges that have a unique local edge label, the two

nodes the edge connects, and the distance. On mainlines the distance is calculated and loaded when the mainline nodes are 

loaded. Currently turns are identified and handled manually by inserting an edge with no distance and calling osrm (open 

street routing map) to determine the driving distance. Typically that edge is prenode from one facility to prenode in the next

facility and is directionally specific.



---



\## Status



This repository currently contains:



\- A skeleton \*\*Pythonista\*\* runtime for iPhone / CarPlay (`pythonista/`)

\- Build scripts for the corridor graph (`build/`)

\- Documentation for the corridor / node / edge model (`docs/`)

\- Example data stubs (`data/`)



The real per-user data (like “home” coordinates) are \*\*not\*\* stored here.

They live in a local `personal.json` file on-device and are ignored by Git.



---



\## Layout



```text

mileage-sign/

&nbsp; data/

&nbsp;   corridor\_master.json	    # master file of universal shared nodes

&nbsp;   trip.json                       # example corridor DB

&nbsp;   personal\_template.json          # user-specific destinations template (no real data)

&nbsp; pythonista/

&nbsp;   mileage\_sign.py                 # main runtime client has OSRM inside

&nbsp;   osrm\_client.py                  # OSRM HTTP helper in case we want to debundle it in the future

&nbsp; build/

&nbsp;   build\_mainline\_from\_relation.py # primitive build of universal shared nodes ... to be replaced by add\_junction\_bundle.py

&nbsp;   chainify.py                     # build trip / corridor files ... to migrate to Pythonista for on phone use

&nbsp;   osrm\_fill.py                    # fill edge distances from OSRM    

&nbsp;   add\_junction\_bundle.py          # add a junction + edges in one call

&nbsp; docs/

&nbsp;   architecture.md                 # description of nodes, edges, roles

&nbsp; tests/

&nbsp;   sample\_trip.json                # small sample trip



