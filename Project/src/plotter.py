import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
plots_dir = os.path.join(project_dir, "plots")


def _save_plot(title, fig=None):
    """Save a matplotlib plot with a title and timestamp."""

    if fig is None:
        fig = plt.gcf()
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create safe filename from title
    safe_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in title)
    safe_title = safe_title.replace(' ', '_')
    
    # Create filename with title and timestamp
    filename = f"{safe_title}_{timestamp}.png"
    filepath = os.path.join(plots_dir, filename)
    
    # Save the plot
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {filepath}")
    
    return filepath


def full_pairplot(data):
    sns.pairplot(data, hue="OUTCOME", palette="viridis")


def histogram(data, column, bins=10):
    """Plot a histogram for a specified column in the DataFrame."""
    plt.figure(figsize=(10, 6))
    plt.hist(data[column], bins=bins, edgecolor='black')
    plt.title("Histogram of " + column)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    _save_plot(title="Histogram_of_" + column)
    plt.show()


def histogram_by_outcome(data, column, bins=20):
    """Plot histograms for a specified column, separated by OUTCOME."""
    outcomes = data['OUTCOME'].unique()
    plt.figure(figsize=(12, 6))
    
    for outcome in outcomes:
        subset = data[data['OUTCOME'] == outcome]
        plt.hist(subset[column], bins=bins, alpha=0.6, label=f'OUTCOME={outcome}', edgecolor='black')
    
    plt.title(f'Histogram of {column} by OUTCOME')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    _save_plot(title=f"Histogram_of_{column}_by_OUTCOME")
    plt.show()


def bar_by_outcome(data, column):
    """plot a bar plot for catogorical column separated by OUTCOME."""
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=column, hue="OUTCOME", palette="Set2")
    for p in plt.gca().patches:
        height = p.get_height()
        plt.gca().annotate(f'{height:.0f}', (p.get_x() + p.get_width() / 2., height),
                           ha='center', va='center', fontsize=9, color='black', xytext=(0, 5),
                           textcoords='offset points')
    plt.title(f'Bar Plot of {column} by OUTCOME')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.legend(title='OUTCOME')
    _save_plot(title=f"Bar_Plot_of_{column}_by_OUTCOME")
    plt.show()


def histogeam_by_var(data, var, hue):
    """Plot a distribution plot for a specified variable with hue."""
    sns.displot(data=data, x=var, hue=hue, kind="kde", fill=True, alpha=0.4)
    plt.title(f"Histogram of {var} by {hue}")
    plt.xlabel(var)
    plt.ylabel("Density")
    _save_plot(title=f"Histogram_of_{var}_by_{hue}")
    plt.show()





# def box_plot(data, column, title="", ylabel=""):
#     """Plot a box plot for a specified column in the DataFrame."""
#     plt.figure(figsize=(8, 6))
#     plt.boxplot(data[column].dropna())
#     plt.title(title)
#     plt.ylabel(ylabel)
#     plt.grid(True)
#     plt.show()