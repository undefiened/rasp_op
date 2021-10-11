# A proof-of-concept implementation of Orienteering Problem for RASP

## Installation
To run the program you need to install the dependencies from pip.
You may use `pip install -r requirements.txt` to install the dependencies.
You need to have any solver supported by `python-mip` to be installed locally. Please check https://www.python-mip.com/ for details.
Currently, the solver by default is CBC, since it is open-source and doesn't need additional configuration, however Gurobi can easily be chosen.

To connect to the database an `auth.py` file with the following structure needs to be created:
```
DB_HOST = 'localhost'
DB_PORT = 5455
DB_USER = 'yourusername'
DB_PASSWORD = 'yourpassword'
DB_NAME = 'yourdatabasename'
```

## Disclaimer
This code is intended for research purposes. I am absolutely certain that there are bugs in this code.
There is no warranty for anything, the authors bears no responsibility for anything bad that happens because of this code.
Run the code on your discretion.

### Never ever use it for real applications!