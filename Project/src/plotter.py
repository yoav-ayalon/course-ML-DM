import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


def histogram(data, column, bins=10, title="", xlabel="", ylabel="Frequency"):
    """Plot a histogram for a specified column in the DataFrame."""
    plt.figure(figsize=(10, 6))
    plt.hist(data[column], bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def full_pairplot(data):
    sns.pairplot(data, hue="OUTCOME", palette="viridis")

def custom_pairplot(data, columns, hue="OUTCOME"):
    pass

def histogeam_by_var(data, var, hue):
    """Plot a distribution plot for a specified variable with hue."""
    sns.displot(data=data, x=var, hue=hue, kind="kde", fill=True, alpha=0.4)
    plt.title(f"Histogram of {var} by {hue}")
    plt.xlabel(var)
    plt.ylabel("Density")
    plt.show()

def scatter_plot(data, x_column, y_column, title="", xlabel="", ylabel=""):
    """Plot a scatter plot for two specified columns in the DataFrame."""
    plt.figure(figsize=(10, 6))
    plt.scatter(data[x_column], data[y_column], alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def box_plot(data, column, title="", ylabel=""):
    """Plot a box plot for a specified column in the DataFrame."""
    plt.figure(figsize=(8, 6))
    plt.boxplot(data[column].dropna())
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()