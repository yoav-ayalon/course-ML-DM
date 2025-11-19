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




def activate():
    print(f"Loading data from: {raw_data}")
    data = load_data(raw_data)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    # print(data.head())
    # print(data.describe())
    # print(data.info())
    # print(data.dtypes)
    print(data.columns)

    plotter.histogeam_by_var(data, var="INCOME", hue="AGE")
    plotter.histogeam_by_var(data, var="CREDIT_SCORE", hue="INCOME")

    # plotter.histogram(data, column="CREDIT_SCORE", bins=20, title="Credit Score Distribution", xlabel="Credit Score")
    # plotter.histogram(data, column="SPEEDING_VIOLATIONS", title="Histogram of Speeding Violations", xlabel="Speeding Violations")
    # plotter.histogram(data, column="DUIS", title="Histogram of DUIs", xlabel="DUIs")
    # plotter.histogram(data, column="PAST_ACCIDENTS", title="Histogram of Past Accidents", xlabel="Past Accidents")







if __name__ == "__main__":
    activate()



