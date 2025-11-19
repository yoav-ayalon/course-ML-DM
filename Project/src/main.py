import numpy as np
import pandas as pd
import os
import plotter



# Get the directory of the current file and construct path to data
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
data_dir = os.path.join(project_dir, "data", "raw")
raw_data = os.path.join(data_dir, "Car_Insurance_Claim.csv")

def load_data(file_path):
    """Load data from a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

def plotting(data):
    # plotter.histogram(data,"ANNUAL_MILEAGE")
    # plotter.histogeam_by_var(data, var="ANNUAL_MILEAGE", hue="OUTCOME")
    # plotter.histogram_by_outcome(data, column="ANNUAL_MILEAGE")
    plotter.bar_by_outcome(data, column="ANNUAL_MILEAGE")
    pass




def activate():
    print(f"Loading data from: {raw_data}")
    data = load_data(raw_data)

    # print(data.head())
    # print(data.describe())
    # print(data.info())
    # print(data.dtypes)
    print(data.columns)

    plotting(data)







if __name__ == "__main__":
    activate()



