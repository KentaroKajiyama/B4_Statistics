import numpy as np
import pandas as pd


filename = 'tokyo23_wards_office.csv'
coordinate_strings = pd.read_csv(current_dir.joinpath(filename), header=None)
formatted_coordinates = ["(" + coord.replace(" ", ", ") + ")" for coord in coordinate_strings] 
print(formatted_coordinates)