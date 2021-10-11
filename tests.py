from unittest import TestCase

import numpy as np

from main import OrienteeringProblemInstance, NoRouteException


class OrienteeringProblemTestCase(TestCase):
    def test_id_coord_conversion(self):
        inst = OrienteeringProblemInstance(np.zeros([10, 15]), (0, 0), (1, 1), 100, 1)

        self.assertEqual(25, inst._coord_to_node_id(inst._node_id_to_coord(25)))
        self.assertEqual(0, inst._coord_to_node_id(inst._node_id_to_coord(0)))
        self.assertEqual(150, inst._coord_to_node_id(inst._node_id_to_coord(150)))

    def test_start_end_point_conversion(self):
        inst = OrienteeringProblemInstance(np.zeros([10, 15]), (0, 0), (1, 1), 100, 1)

        self.assertEqual((0, 0), inst._node_id_to_coord(inst.START_NODE_ID))
        self.assertEqual((1, 1), inst._node_id_to_coord(inst.END_NODE_ID))

        inst = OrienteeringProblemInstance(np.zeros([10, 15]), (0, 0), (0, 0), 100, 1)

        self.assertEqual((0, 0), inst._node_id_to_coord(inst.START_NODE_ID))
        self.assertEqual((0, 0), inst._node_id_to_coord(inst.END_NODE_ID))

    def test_simple_1(self):
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

    def test_simple_2(self):
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

    def test_simple_3(self):
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

    def test_one_endpoint(self):
        map = np.array([
            [0, 10],
            [0, 0],
            [0, 0],
            [0, 1000],
        ])

        inst = OrienteeringProblemInstance(map, (0, 0), (0, 0), 10, 1)
        route, weight = inst.initial_route()
        print(route)
        # self.assertEqual(len(route), 14)
        self.assertEqual(weight, 1010)

    def test_minimize(self):
        map = np.array([
            [0, -10],
            [0, 0],
            [0, 0],
            [0, -1000],
        ])

        inst = OrienteeringProblemInstance(map, (0, 0), (0, 0), 10, 1)
        route, weight = inst.initial_route(minimize_scores=True)
        print(route)
        # self.assertEqual(len(route), 14)
        self.assertEqual(weight, -1010)

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
