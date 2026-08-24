# Copyright (C) 2024 Carnegie Mellon University

"""Builds the FlightPlan artifacts; Map.write2disk() exports them."""

from swiftmap.database.map import Map
from swiftmap.database.types import FlightPlan, GPS


def generate_flight_plan(map: Map, flight_plan: FlightPlan) -> FlightPlan:
    """GPS-tag the plan's ENU viewpoints and attach it to the Map."""
    georef = map.get_georeference()
    if georef is not None:
        for vp in flight_plan.viewpoints:
            vp.position_gps = GPS(*georef.enu_to_lla(vp.pose.position)[0])
            vp.target_gps = GPS(*georef.enu_to_lla(vp.target)[0])
            flight_plan.waypoints.append(vp.position_gps)

    map.update_flight_plan(flight_plan)
    return flight_plan
