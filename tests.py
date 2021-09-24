from unittest import TestCase

import numpy as np

from main import OrienteeringProblemInstance, NoRouteException


class OrienteeringProblemTestCase(TestCase):
    def test_id_coord_conversion(self):
        inst = OrienteeringProblemInstance(np.zeros([10, 15]), (0, 0), (1, 1), 100, 1)

        self.assertEqual(25, inst._coord_to_node_id(inst._node_id_to_coord(25)))
        self.assertEqual(0, inst._coord_to_node_id(inst._node_id_to_coord(0)))
        self.assertEqual(150, inst._coord_to_node_id(inst._node_id_to_coord(150)))

    def test_simple(self):
        map = np.array([
            [0, 1000, 1000, 1000],
            [1000, 0, 1000, 1000],
            [1000, 1000, 0, 1000],
            [1000, 1000, 1000, 0],
        ])

        inst = OrienteeringProblemInstance(map, (0, 0), (3, 3), 3, 1)
        route, weight = inst.initial_route()
        self.assertEqual(len(route), 4)
        self.assertEqual(weight, 0)

        map = np.array([
            [0, 1000, 1000, 1000],
            [1000, 0, 1000, 1000],
            [1000, 1000, 0, 1000],
            [1000, 1000, 1000, 0],
        ])

        inst = OrienteeringProblemInstance(map, (0, 0), (3, 3), 4, 1)
        route, weight = inst.initial_route()
        self.assertEqual(len(route), 5)
        self.assertEqual(weight, 3000)

        map = np.array([
            [0, 0, 1000, 1000],
            [1000, 1000, 1000, 1000],
            [1000, 1000, 0, 1000],
            [1000, 1000, 1000, 0],
        ])

        inst = OrienteeringProblemInstance(map, (0, 0), (0, 1), 3, 1)
        route, weight = inst.initial_route()
        self.assertEqual(len(route), 4)
        self.assertEqual(weight, 2000)

    def test_no_route(self):
        map = np.array([
            [0, 0, 1000, 1000],
            [1000, 1000, 1000, 1000],
            [1000, 1000, 0, 1000],
            [1000, 1000, 1000, 0],
        ])

        inst = OrienteeringProblemInstance(map, (0, 0), (3, 3), 2, 1)
        self.assertRaises(NoRouteException, inst.initial_route)

    def test_replanning(self):
        map = np.array([
            [0, 1000, 1000, 1000],
            [0, 0, 1000, 1000],
            [1000, 0, 0, 1000],
            [1000, 0, 1000, 0],
        ])

        inst = OrienteeringProblemInstance(map, (0, 0), (3, 3), 4, 1)
        route, weight = inst.replan_route([(0, 0), (1, 0), (1, 1)])
        self.assertEqual(len(route), 5)
        self.assertEqual(route, [(0, 0), (1, 0), (1, 1), (2, 2), (3, 3)])
        self.assertEqual(weight, 0)

    def test_no_route_during_replanning(self):
        map = np.array([
            [0, 1000, 1000, 1000],
            [0, 0, 1000, 1000],
            [1000, 0, 0, 1000],
            [1000, 0, 1000, 0],
        ])

        inst = OrienteeringProblemInstance(map, (0, 0), (3, 3), 4, 1)
        self.assertRaises(NoRouteException, inst.replan_route, [(0, 0), (1, 0), (2, 0)])
