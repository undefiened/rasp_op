import itertools
from math import floor
from typing import Tuple, List

import numpy as np
import psycopg2
from mip import Model, BINARY, maximize, minimize, xsum, INTEGER, OptimizationStatus, CONTINUOUS

from auth import *


class NoRouteException(BaseException):
    """There is no feasible route, the problem is unfeasible."""


class OrienteeringProblemInstance:
    BIG_NUMBER = 9999999
    START_NODE_ID = -1
    END_NODE_ID = -2

    def __init__(self, map: np.ndarray, starting_point: Tuple[int, int], end_point: Tuple[int, int], max_route_time_s: float, speed_cells_per_s: float):
        self.map = map

        self.starting_point = starting_point
        self.end_point = end_point

        self.start_node = self.START_NODE_ID
        self.dest_node = self.END_NODE_ID

        self.time_limit = max_route_time_s
        self.speed = speed_cells_per_s

    def _number_of_nodes(self):
        return len(self._all_nodes())

    def _node_id_to_coord(self, id: int) -> Tuple[int, int]:
        if id == self.start_node:
            return self.starting_point

        if id == self.dest_node:
            return self.end_point

        y = floor(id/self.map.shape[0])
        x = id - y*self.map.shape[0]
        return x, y

    def _coord_to_node_id(self, coord: Tuple[int, int]) -> int:
        return self.map.shape[0] * coord[1] + coord[0]

    def _edge_weight(self, from_node, to_node):
        from_coord = self._node_id_to_coord(from_node)
        to_coord = self._node_id_to_coord(to_node)

        if abs(from_coord[0] - to_coord[0]) <= 1 and abs(from_coord[1] - to_coord[1]) <= 1:
            return floor(1/self.speed)
        else:
            return self.BIG_NUMBER

    def _all_nodes(self):
        return [self.start_node] + [x for x in range(self.map.shape[0]*self.map.shape[1]) if x not in
                (self._coord_to_node_id(self.starting_point), self._coord_to_node_id(self.end_point))
                ] + [self.dest_node, ]

    def _all_without_start(self):
        return [x for x in self._all_nodes() if x != self.start_node]

    def _all_without_dest(self):
        return [x for x in self._all_nodes() if x != self.dest_node]

    def _all_without_start_and_dest(self):
        return [x for x in self._all_nodes() if x not in (self.start_node, self.dest_node)]

    def _compute_route(self, visited_nodes: List[int], minimize_scores) -> Tuple[List[Tuple[int, int]], float]:
        ip = Model("OP instance")
        x = {j: {i: ip.add_var(var_type=BINARY) for i in self._all_nodes()} for j in self._all_nodes()}
        u = {i: ip.add_var(var_type=CONTINUOUS) for i in self._all_nodes()}

        # the objective is to get as much "profit" as possible in the given time
        obj_func = maximize if not minimize_scores else minimize
        ip.objective = obj_func(xsum(
            [self.map[self._node_id_to_coord(i)[0]][self._node_id_to_coord(i)[1]] * x[i][j] for i in
             self._all_without_start_and_dest() for j in self._all_without_start()]))

        # ensure that the path starts from start node and end on the dest node
        ip.add_constr(xsum([x[self.start_node][j] for j in self._all_without_start()]) == 1)
        ip.add_constr(xsum([x[i][self.dest_node] for i in self._all_without_dest()]) == 1)

        # remove edges between the same nodes
        for i in self._all_nodes():
            ip.add_constr(x[i][i] == 0)

        # ensure that the path starts from start node and end on the dest node
        for k in self._all_without_start_and_dest():
            constr1 = xsum([x[i][k] for i in self._all_without_dest()])
            constr2 = xsum([x[k][j] for j in self._all_without_start()])

            ip.add_constr(constr1 == constr2)
            ip.add_constr(constr1 <= 1)
            ip.add_constr(constr2 <= 1)

        # ensure that our route is within the time limit
        ip.add_constr(xsum([x[i][j] * self._edge_weight(i, j) for i in self._all_without_dest() for j in
                            self._all_without_start()]) <= self.time_limit)

        # fix already visited nodes during replanning
        if len(visited_nodes) > 0:
            previous_node = visited_nodes[0]

            for visited_node in visited_nodes[1:]:
                ip.add_constr(x[previous_node][visited_node] == 1)
                previous_node = visited_node

        # subtour elimination constraints
        for i in self._all_without_start():
            ip.add_constr(2 <= u[i])
            ip.add_constr(u[i] <= self._number_of_nodes())

        for i in self._all_without_start():
            for j in self._all_without_start():
                ip.add_constr(u[i] - u[j] + 1 <= (self._number_of_nodes() - 1) * (1 - x[i][j]))

        status = ip.optimize()

        if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
            us = np.zeros_like(self.map)
            for i in self._all_nodes():
                pos = self._node_id_to_coord(i)
                us[pos[0], pos[1]] = float(u[i])

            route = self._unwrap_route(x)
            return route, float(ip.objective)
        else:
            raise NoRouteException('No route found at all!')

    def initial_route(self, minimize_scores=False):
        return self._compute_route([], minimize_scores)

    def replan_route(self, visited_coords: List[Tuple[int, int]], minimize_scores=False):
        if visited_coords[0] != self.starting_point:
            raise NoRouteException('The first visited point is not the starting point')

        visited = [self._coord_to_node_id(x) for x in visited_coords]
        visited[0] = self.start_node

        return self._compute_route(visited, minimize_scores)

    def _unwrap_route(self, x):
        route = []
        previous_node = self.start_node
        route.append(self.starting_point)

        while previous_node != self.dest_node:
            found_next = False
            for i in self._all_nodes():
                if x[previous_node][i].x > 0.9:
                    # print('{} - {}'.format(self._node_id_to_coord(previous_node), self._node_id_to_coord(i)))
                    route.append(self._node_id_to_coord(i))
                    previous_node = i

                    found_next = True

            if not found_next:
                raise NoRouteException("Cannot unwrap the route")

        return route


