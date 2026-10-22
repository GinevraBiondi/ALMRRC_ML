import pandas as pd
import numpy as np






def solve_otsp_greedy(C, start_zone):

    unvisited = set(C.index)

    if start_zone in unvisited:
        unvisited.remove(start_zone)

    route = [start_zone]
    current = start_zone

    while unvisited:

        nxt = min(
            unvisited,
            key=lambda z: C.loc[current, z]
        )

        route.append(nxt)

        unvisited.remove(nxt)
        current = nxt

    return route

##funzione di predizione con distanze euclidee


def build_cost_distance(
    route_zones,
    start_zone,
    D,
    P,
    omega=0.05
):
    """
    Cost matrix:
    C = omega*D + (1-omega)*(1-P)
    """

    C = (
        omega * D +
        (1 - omega) * (1 - P)
    )

    np.fill_diagonal(C.values, 0)

    C_route = C.loc[
        route_zones,
        route_zones
    ]

    prediction = solve_otsp_greedy(
        C_route,
        start_zone
    )

    return prediction

#funzione di predizione con inter tempi

def build_cost_time(
    route_zones,
    start_zone,
    T,
    P,
    omega=0.4
):
    """
    Cost matrix:
    C = omega*T + (1-omega)*(1-P)
    """

    C = (
        omega * T +
        (1 - omega) * (1 - P)
    )

    np.fill_diagonal(C.values, 0)

    C_route = C.loc[
        route_zones,
        route_zones
    ]

    prediction = solve_otsp_greedy(
        C_route,
        start_zone
    )

    return prediction


def edge_accuracy(true_sequence, pred_sequence):

    true_edges = set(
        zip(
            true_sequence[:-1],
            true_sequence[1:]
        )
    )

    pred_edges = set(
        zip(
            pred_sequence[:-1],
            pred_sequence[1:]
        )
    )

    return len(true_edges & pred_edges) / len(true_edges)