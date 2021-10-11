import itertools
from math import floor
from typing import Tuple, List

import networkx
import numpy as np
import psycopg2
import simplejson
from mip import Model, BINARY, maximize, minimize, xsum, INTEGER, OptimizationStatus, CONTINUOUS
from osgeo import ogr

from auth import *


class NoRouteException(BaseException):
    """There is no feasible route, the problem is unfeasible."""


class OrienteeringProblemInstance:
    BIG_NUMBER = 9999999
    MAX_SECONDS = 60
    START_NODE_ID = -1
    END_NODE_ID = -2

    def __init__(self, map: np.ndarray, starting_point: Tuple[int, int], end_point: Tuple[int, int], max_route_time_s: float, speed_cells_per_s: float, solver='CBC'):
        self.map = map

        self.starting_point = starting_point
        self.end_point = end_point

        self.start_node = self.START_NODE_ID
        self.dest_node = self.END_NODE_ID

        self.time_limit = max_route_time_s
        self.speed = speed_cells_per_s
        self.solver = solver

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

    def _all_nodes_without(self, nodes_to_remove):
        return [x for x in self._all_nodes() if x not in nodes_to_remove]

    def _all_without_start(self):
        return [x for x in self._all_nodes() if x != self.start_node]

    def _all_without_dest(self):
        return [x for x in self._all_nodes() if x != self.dest_node]

    def _all_without_start_and_dest(self):
        return [x for x in self._all_nodes() if x not in (self.start_node, self.dest_node)]

    def _compute_route(self, visited_nodes: List[int], minimize_scores) -> Tuple[List[Tuple[int, int]], float]:
        ip = Model("OP instance", solver_name=self.solver)
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

        ip.add_constr(x[self.start_node][self.dest_node] == 0)

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

        # subtour elimination constraints MTZ
        for i in self._all_without_start():
            ip.add_constr(2 <= u[i])
            ip.add_constr(u[i] <= self._number_of_nodes())

        for i in self._all_without_start():
            for j in self._all_without_start():
                ip.add_constr(u[i] - u[j] + 1 <= (self._number_of_nodes() - 1) * (1 - x[i][j]))

        status = ip.optimize(max_seconds=self.MAX_SECONDS)

        if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
            route = self._unwrap_route(x)
            return route, float(ip.objective)
        else:
            raise NoRouteException('No route found at all!')

    def _detect_subtours(self, x):
        G = networkx.Graph()

        edges = []
        for i in self._all_nodes():
            for j in self._all_nodes():
                if x[i][j].x > 0.9:
                    edges.append((i, j))

        G.add_edges_from(edges)
        subtours = list(networkx.connected_components(G))

        if len(subtours) > 1:
            return subtours
        else:
            return None

    def _print_all_edges(self, x):
        for i in self._all_nodes():
            for j in self._all_nodes():
                if x[i][j].x > 0.9:
                    print('{} - {}'.format(self._node_id_to_coord(i), self._node_id_to_coord(j)))

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


class DBWorker:
    def __init__(self):
        self.conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        self.grid_width = 0
        self.grid_height = 0

        self.grid_original_ids = np.array((0, 0))
        self.grid_probabilities = np.array((0, 0))
        self.grid_geometries = {}

        self._read_grid()
        self.origin_point = self._get_origin_point_id()

    def _read_grid(self):
        curs = self.conn.cursor()
        curs.execute('SELECT row_id, prob, time, ST_AsGeoJSON(geom)  FROM search_area.grid_100_prob_drone;')

        all_entries = curs.fetchall()
        curs.close()

        all_entries = [
            (r[0], r[1], r[2], simplejson.loads(ogr.CreateGeometryFromJson(r[3]).ExportToJson())['coordinates'][0][0]) for r in all_entries
        ]

        unique_xs, unique_ys = self._find_unique_xs_and_ys(all_entries)

        self.grid_width = len(unique_xs) - 1
        self.grid_height = len(unique_ys) - 1

        self.grid_original_ids = np.zeros((self.grid_width, self.grid_height))
        self.grid_probabilities = np.zeros((self.grid_width, self.grid_height))

        for r in all_entries:
            min_x = min([point[0] for point in r[3]])
            min_y = min([point[1] for point in r[3]])

            x = unique_xs.index(min_x)
            y = unique_ys.index(min_y)

            self.grid_original_ids[x, y] = r[0]
            self.grid_probabilities[x, y] = r[1]

    def _find_unique_xs_and_ys(self, all_entries):
        unique_xs = []  # set([ogr.CreateGeometryFromJson(x[4]) for x in all_entries])
        unique_ys = []
        for r in all_entries:
            geom = r[3]

            self.grid_geometries[r[0]] = geom

            for point in geom:
                unique_xs.append(point[0])
                unique_ys.append(point[1])
        unique_xs = sorted(list(set(unique_xs)))
        unique_ys = sorted(list(set(unique_ys)))
        return unique_xs, unique_ys

    def _get_origin_point_id(self):
        curs = self.conn.cursor()

        curs.execute("select ST_AsGeoJSON(geom) from search_area.points where name = 'drone_start'")

        pg = curs.fetchone()[0]
        pg = simplejson.loads(ogr.CreateGeometryFromJson(pg).ExportToJson())['coordinates']
        curs.close()

        origin_id = None

        for key, geom in self.grid_geometries.items():
            min_x = min([x[0] for x in geom])
            max_x = max([x[0] for x in geom])

            min_y = min([x[1] for x in geom])
            max_y = max([x[1] for x in geom])

            if min_x < pg[0] < max_x and min_y < pg[1] < max_y:
                if origin_id is not None:
                    raise Exception('Two or more cells contain the start point')

                origin_id = key

        if origin_id == None:
            raise Exception('Cannot find the origin cell')

        return origin_id

    def get_origin_point(self):
        loc = np.where(self.grid_original_ids == self.origin_point)
        return loc[0][0], loc[1][0]

    def get_log_not_probabilities_grid(self):
        log_not_prob = np.log(1 - self.grid_probabilities)

        return log_not_prob

    def _cell_id_to_geom(self, cid):
        geom = self.grid_geometries[cid]
        min_x = min(x[0] for x in geom)
        max_x = max(x[0] for x in geom)

        min_y = min(x[1] for x in geom)
        max_y = max(x[1] for x in geom)

        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint((max_x+min_x)/2, (max_y+min_y)/2)
        point.FlattenTo2D()

        return point.ExportToJson()

    def write_route(self, drone_id, route):
        curs = self.conn.cursor()

        curs.execute("select max(trajectory_id) from drone.waypoints;")
        new_trajectory_id = curs.fetchone()[0] + 1

        for waypoint in route:
            waypoint_id = self.grid_original_ids[waypoint[0], waypoint[1]]

            geom = self._cell_id_to_geom(waypoint_id)
            curs.execute(
                "INSERT INTO drone.waypoints(drone_id, trajectory_id, waypoint_id, lon, lat, altitude, the_geom) VALUES (%s, %s, %s, %s, %s, %s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 3066));",
                (drone_id, new_trajectory_id, waypoint_id, None, None, None, geom)
            )

        self.conn.commit()


if __name__ == "__main__":
    db = DBWorker()
    log_not_prob_grid = db.get_log_not_probabilities_grid()
    origin_point = db.get_origin_point()

    # print(log_not_prob_grid)

    inst = OrienteeringProblemInstance(log_not_prob_grid, origin_point, origin_point, 10, 1, solver='GRB')
    route, _ = inst.initial_route(minimize_scores=True)

    db.write_route(-1, route)
    print(route)