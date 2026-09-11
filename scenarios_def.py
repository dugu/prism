"""Definition of the eight preliminary experimental scenarios.

Each entry fixes the source road scene, the ego speed and the per-object
trajectories.  `motion` is keyed by the object rank in the source frame (rank 0
is the largest annotated box); `v_long` is the object's speed along the ego
axis (positive means travelling in the same direction as the ego vehicle) and
`v_lat` its lateral speed (positive means moving towards the right).
"""
from pathlib import Path
import os
WORK = Path(os.environ.get("PRISM_WORK_DIR", str(Path(__file__).resolve().parent / "work")))

IMG = Path(str(WORK) + "/data/coco_traffic/images")
LBL = Path(str(WORK) + "/data/coco_traffic/labels")

SCENARIOS = [
    dict(name="S1", title="Urban arterial, cyclist drifting into lane",
         img="000000122166", v_ego=5.5, primary=4,
         motion={1: {"v_long": 5.0}, 2: {"v_long": 4.5}, 3: {"v_long": 4.7},
                 4: {"v_long": 3.5, "v_lat": 0.7}, 5: {"v_long": 4.2, "v_lat": 0.4},
                 6: {"v_long": 4.8}, 9: {"v_long": 4.0}, 10: {"v_long": 3.0},
                 7: {"v_lat": 0.5}, 8: {"v_lat": -0.4}},
         description="Mixed urban traffic at 20 km/h. A cyclist travelling more "
                     "slowly than the ego vehicle drifts progressively towards "
                     "the travel corridor while eleven further road users share "
                     "the scene."),
    dict(name="S2", title="Urban street, pedestrian approaching kerb",
         img="000000026204", v_ego=5.0, primary=6,
         motion={0: {"v_long": 4.0}, 1: {"v_long": 3.5}, 2: {"v_long": 4.5},
                 3: {"v_long": 4.6}, 4: {"v_long": 4.5}, 5: {"v_long": 4.0},
                 6: {"v_lat": 1.0}, 7: {"v_long": 4.2}, 10: {"v_long": 4.0},
                 9: {"v_lat": -0.6}},
         description="City street with a bus, two trucks and several cars. A "
                     "pedestrian on the left pavement walks towards the "
                     "carriageway."),
    dict(name="S3", title="Dense stop-and-go queue",
         img="000000017627", v_ego=4.0, primary=2,
         motion={0: {"v_long": 3.0}, 1: {"v_long": 3.5}, 2: {"v_lat": 0.8},
                 3: {"v_long": 3.5}, 4: {"v_long": 3.6}, 5: {"v_long": 3.2},
                 6: {"v_long": 3.4}, 7: {"v_long": 3.3}, 8: {"v_long": 3.0},
                 9: {"v_long": 3.0}, 10: {"v_long": 3.0}, 11: {"v_lat": -0.7},
                 12: {"v_long": 3.0}, 13: {"v_long": 3.0}},
         description="Fourteen simultaneously visible road users in a congested "
                     "queue. Highest object density of the set and therefore the "
                     "stress test for display clutter."),
    dict(name="S4", title="Motorway approach to slower lead vehicles",
         img="000000001532", v_ego=25.0, primary=2,
         motion={0: {"v_long": 23.5}, 1: {"v_long": 24.0}, 2: {"v_long": 21.0},
                 3: {"v_long": 24.0}, 4: {"v_long": 23.0}, 5: {"v_long": 24.0},
                 6: {"v_long": 24.0}, 7: {"v_long": 20.0}},
         description="Multi-lane motorway at 90 km/h under an overpass; the ego "
                     "vehicle closes on a slower vehicle in the adjacent lane."),
    dict(name="S5", title="Wet urban approach at dusk",
         img="000000093154", v_ego=12.0, primary=1,
         motion={0: {"v_long": 8.0}, 1: {"v_long": 8.5}, 2: {"v_long": 9.0},
                 3: {"v_long": 9.0}, 4: {"v_long": 9.5}, 5: {"v_long": 9.0},
                 6: {"v_long": 10.0}, 7: {"v_long": 9.5}, 8: {"v_long": 9.5},
                 9: {"v_long": 10.0}, 11: {"v_long": 10.0}},
         description="Wet carriageway with a low sun angle, signalised junction "
                     "ahead and a stream of vehicles at 43 km/h."),
    dict(name="S6", title="Level crossing with lateral rail traffic",
         img="000000057149", v_ego=5.0, primary=0,
         motion={0: {"v_long": 4.0}, 1: {"v_lat": 7.0}, 4: {"v_long": 4.0}},
         description="Creeping approach to a level crossing behind a stationary "
                     "vehicle while a train crosses laterally."),
    dict(name="S7", title="Urban street with pedestrians and a cyclist",
         img="000000067616", v_ego=6.0, primary=4,
         motion={0: {"v_lat": 1.0}, 1: {"v_long": 5.0}, 2: {"v_long": 5.0},
                 3: {"v_long": 5.5}, 4: {"v_long": 4.0}, 5: {"v_lat": 0.6},
                 6: {"v_long": 5.0}, 7: {"v_lat": -0.5}, 8: {"v_lat": 0.5},
                 9: {"v_lat": -0.4}},
         description="Narrow commercial street with five pedestrians, parked and "
                     "moving vehicles, and a cyclist sharing the lane."),
    dict(name="S8", title="Signalised avenue in reduced visibility",
         img="000000054967", v_ego=9.0, primary=2,
         motion={1: {"v_lat": -0.8}, 2: {"v_long": 7.0}, 3: {"v_long": 7.5},
                 4: {"v_long": 7.5}, 5: {"v_long": 7.2}, 6: {"v_long": 7.0},
                 7: {"v_long": 7.5}, 9: {"v_long": 8.0}, 10: {"v_long": 8.0},
                 11: {"v_long": 8.0}, 12: {"v_long": 8.0}, 13: {"v_long": 8.0}},
         description="Wide avenue with a queue of vehicles, several traffic "
                     "signals and a pedestrian at the kerb, at 32 km/h."),
]

# Environmental conditions applied on top of every scenario in experiment E4
DEGRADATIONS = ["none", "night", "fog", "rain"]


def paths(s):
    return IMG / (s["img"] + ".jpg"), LBL / (s["img"] + ".txt")


def build(s, degradation="none", fps=20):
    from prism.scenario import build_scenario
    ip, lp = paths(s)
    return build_scenario(ip, lp, s["name"], s["title"], v_ego=s["v_ego"],
                          primary_id=s.get("primary"), motion=s["motion"],
                          degradation=degradation, fps=fps)
