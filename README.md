# A proof-of-concept implementation of Orienteering Problem for RASP

## Installation
To run the program you need to install the dependencies from pip.
You may use `pip install -r requirements.txt` to install the dependencies.
You need to have any solver supported by `python-mip` to be installed locally. Please check https://www.python-mip.com/ for details.

## Running
`OrienteeringProblemInstance` contains only three functions important for external users:
* The constuctor, which creates a problem instance from a map (an `numpy` matrix where entries represent the cell score), start and destination coordinates, time limit (how much time can the drone fly) and the drone speed in cells per second.
* `initial_route` which computes the initial route for the drone to follow.
* `replan_route` which recomputes the route, given first several points of the route in the correct order.

See `test.py` for examples on how to run the `OrienteeringProblemInstance`.

## Disclaimer
This code is intended for research purposes. I am absolutely certain that there are bugs in this code.
There is no warranty for anything, the authors bears no responsibility for anything bad that happens because of this code.
Run the code on your discretion.

### Never ever use it for real applications!