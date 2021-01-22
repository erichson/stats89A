import os.path

import pandas as pd

HERE = os.path.dirname(__file__)

circuit_courts = pd.read_csv(os.path.join(HERE, "circuit_courts.csv"))

discretized_functions = pd.read_csv(os.path.join(HERE, "discretized_functions.csv"))

eastern_cities = pd.read_csv(os.path.join(HERE, "eastern_cities.csv"))

iris = pd.read_csv(os.path.join(HERE, "iris.csv"))

federal_reserve_districts = pd.read_csv(
    os.path.join(HERE, "federal_reserve_districts.csv")
)

metropolises = pd.read_csv(os.path.join(HERE, "metropolises.csv"))

mountains = pd.read_csv(os.path.join(HERE, "mountains.csv"))

states = pd.read_csv(os.path.join(HERE, "states.csv"))

students = pd.read_csv(os.path.join(HERE, "students.csv"))

western_cities = pd.read_csv(os.path.join(HERE, "western_cities.csv"))
