import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
import os
import statsmodels.api as sm
import pingouin as pg
import seaborn as sns
import statsmodels.api as sm
import ipywidgets as widgets
import plotly.graph_objects as go
import itertools

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr
from IPython.display import display
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.metrics import accuracy_score
from statistics import mode
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr


# Define a function to plot multiple subplots with scatter and regression lines
def plot_multiple_scatter_plots(df, mapping, plot_data_func):
    """
    Plots multiple scatter plots with regression lines in a 2x3 subplot layout.

    Args:
    df (DataFrame): The input dataframe.
    mapping (dict): Mapping of sample categories to color and marker.
    plot_data_func (function): Function used to plot data with regression lines.

    Returns:
    None
    """
    # Define the columns for x and y axes
    x_col_clay = 'Clay'
    y_col_cec = 'CEC'
    x_col_Xhf = 'Khf'
    x_col_fe = 'Fe'
    x_col_humus = 'Humus'

    # Create subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # First row of plots using df
    # Plot scatter and regression lines for Clay vs CEC
    plot_data_func(axs[0, 0], df, x_col_clay, y_col_cec, mapping, include_label=True)
    axs[0, 0].set_xlabel('Clay')
    axs[0, 0].set_ylabel('CEC')
    axs[0, 0].grid(True)

    # Plot scatter and regression lines for Humus vs CEC
    plot_data_func(axs[0, 1], df, x_col_humus, y_col_cec, mapping, include_label=False)
    axs[0, 1].set_xlabel('Humus')
    axs[0, 1].set_ylabel('CEC')
    axs[0, 1].grid(True)
    axs[0, 1].set_yscale('log')

    # Plot scatter and regression lines for Xhf vs CEC
    plot_data_func(axs[0, 2], df, x_col_Xhf, y_col_cec, mapping, include_label=False)
    axs[0, 2].set_ylabel('CEC')
    axs[0, 2].set_xlabel('Xhf')
    axs[0, 2].grid(True)
    axs[0, 2].set_yscale('log')

    # Second row of plots using df
    # Plot scatter and regression lines for Fe vs CEC
    plot_data_func(axs[1, 0], df, x_col_fe, y_col_cec, mapping, include_label=True)
    axs[1, 0].set_xlabel('Fe')
    axs[1, 0].set_ylabel('CEC')
    axs[1, 0].grid(True)

    # Plot scatter and regression lines for Fe vs Clay
    plot_data_func(axs[1, 1], df, x_col_fe, x_col_clay, mapping, include_label=False)
    axs[1, 1].set_xlabel('Fe')
    axs[1, 1].set_ylabel('Clay')
    axs[1, 1].grid(True)

    # Plot scatter and regression lines for Fe vs Xhf
    plot_data_func(axs[1, 2], df, x_col_fe, x_col_Xhf, mapping, include_label=False)
    axs[1, 2].set_ylabel('Xhf')
    axs[1, 2].set_xlabel('Fe')
    axs[1, 2].grid(True)
    axs[1, 2].set_yscale('log')

    # Adjust layout for better readability
    plt.tight_layout()

    # Show the plot
    plt.show()


 # Function to plot the horizontal bar chart
def plot_feature_importance(ax, y_positions, scores, labels, title, color, xlabel="Median R² Test Score", labelsize=14):
    """
    Plots a horizontal bar chart for feature importance.

    Args:
    ax (matplotlib axis): The axis on which to plot.
    y_positions (array): Positions for the y-axis.
    scores (array): Feature scores for the bar lengths.
    labels (list): Labels for the y-axis ticks.
    title (str): The title of the plot.
    color (str): Color of the bars.
    xlabel (str): Label for the x-axis.
    labelsize (int): Font size for the axis labels.

    Returns:
    rects (BarContainer): The bar container for the plotted bars.
    """
    rects = ax.barh(y_positions, scores, height=0.6, color=color)
    
    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlim(0, 1)
    ax.tick_params(axis='x', labelsize=labelsize)
    
    return rects


def plot_sing(df, var1, var2, mapping, onexone_line=False, log_scale=False):
    """
    Plots a scatter plot for two variables with optional 1:1 line and logarithmic scale.

    Args:
    df (DataFrame): The input dataframe.
    var1 (str): First variable for the x-axis.
    var2 (str): Second variable for the y-axis.
    mapping (dict): Mapping of sample categories to color and marker.
    onexone_line (bool): Whether to plot a 1:1 line.
    log_scale (bool): Whether to use logarithmic scaling.

    Returns:
    None
    """
    # Drop rows where either var1 or var2 has NaN values
    df_clean = df.dropna(subset=[var1, var2])

    for start_str, (color, marker) in mapping.items():
        mask = df_clean['SAMPLE'].str.startswith(start_str) & df_clean[var1].notna() & df_clean[var2].notna()
        filtered_df = df_clean[mask]
        plt.scatter(filtered_df[var1], filtered_df[var2], color=color, marker=marker, label=start_str)
        plt.grid(True)

    # Recalculate the R^2 score with the clean data
    if df_clean[var1].size > 0 and df_clean[var2].size > 0:
        r2 = r2_score(df_clean[var1], df_clean[var2])
    else:
        r2 = None

    # Optionally add a 1:1 line with R^2 in the label if it exists
    if onexone_line and r2 is not None:
        x = np.linspace(min(df_clean[var1]), max(df_clean[var1]), 100)
        plt.plot(x, x, 'k--', label=f'R^2={r2:.2f}')
    
    # Set log scale if requested
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')

    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.legend(fontsize='small')

    if not log_scale and var2=='Ms':
        plt.ylim(0, 0.05)

    # Ensure the output directory exists
    folder_path = 'figures_output/'
    os.makedirs(folder_path, exist_ok=True)
    filename = f"{var1}_{var2}_{'log' if log_scale else 'linear'}.png"
    plt.savefig(folder_path + filename)
        

def plot_data(df, axis, x_col, y_col, mapping, include_label=False, aa=0.7, ss=60, lw=0, label_fontsize=10, legend_fontsize=10):
    """
    Plots scatter data and adds a linear regression line along with the correlation coefficient.

    Args:
    df (DataFrame): The input dataframe.
    axis (matplotlib axis): Axis object to plot on.
    x_col (Series): X-axis data.
    y_col (Series): Y-axis data.
    mapping (dict): Mapping of sample categories to color and marker.
    include_label (bool): Whether to include legend labels.
    aa (float): Transparency (alpha) for scatter points.
    ss (int): Size of scatter points.
    lw (float): Line width for scatter.
    label_fontsize (int): Font size for axis labels.
    legend_fontsize (int): Font size for legend.

    Returns:
    None
    """
    corr = round(np.corrcoef(x_col, y_col)[0][1], 2)
    for start_str, (color, marker) in mapping.items():
        mask = df['SAMPLE'].str.startswith(start_str)
        label = f"{start_str} Site" if include_label else None
        axis.scatter(x_col[mask], y_col[mask], alpha=aa, s=ss, linewidth=lw, c=color, marker=marker, label=label)
    # Apply logarithmic scale to the y-axis

    axis.set_yscale('log')
    axis.text(0, 0.98, s=f'Corr = {corr}', fontsize=label_fontsize, 
              verticalalignment='top', horizontalalignment='left', 
              transform=axis.transAxes)
    if include_label:
        axis.legend(fontsize=legend_fontsize)


def plot_data1(axis, df, x_col_name, y_col_name, mapping, include_label=False, aa=0.7, ss=60, lw=0, label_fontsize=10, legend_fontsize=10):
    """
    Plots scatter data and fits individual and global regression lines with R² scores.

    Args:
    axis (matplotlib axis): Axis object to plot on.
    df (DataFrame): The input dataframe.
    x_col_name (str): Name of the x-axis variable.
    y_col_name (str): Name of the y-axis variable.
    mapping (dict): Mapping of sample categories to color and marker.
    include_label (bool): Whether to include legend labels.
    aa (float): Transparency (alpha) for scatter points.
    ss (int): Size of scatter points.
    lw (float): Line width for scatter.
    label_fontsize (int): Font size for axis labels.
    legend_fontsize (int): Font size for legend.

    Returns:
    tuple: Slopes, intercepts, and average Xlf_IP values for individual sites.
    """
    slopes = []
    intercepts = []
    avg_xlf_ips = []

    for start_str, (color, marker) in mapping.items():
        mask = df['SAMPLE'].str.startswith(start_str)
        filtered_df = df[mask]

        if not filtered_df.empty:  # Ensure there are points to plot and fit
            # Scatter plot
            axis.scatter(filtered_df[x_col_name], filtered_df[y_col_name], s=ss, linewidth=lw, c=color, marker=marker, label=f"{start_str} Site" if include_label else None)
            
            # Linear regression for individual sites
            #print('filtered_df[x_col_name]', filtered_df[x_col_name])
            #print('yfiltered_df[y_col_name]', filtered_df[y_col_name])
            
            slope, intercept = np.polyfit(filtered_df[x_col_name], filtered_df[y_col_name], 1)
            x_fit = np.linspace(filtered_df[x_col_name].min(), filtered_df[x_col_name].max(), 100)
            y_fit = slope * x_fit + intercept
            axis.plot(x_fit, y_fit, color=color, linestyle='-', linewidth=2, alpha=0.5)
            
            # Save the slope, intercept, and average Xlf_IP
            slopes.append(slope)
            intercepts.append(intercept)
            avg_xlf_ips.append(filtered_df['Klf'].mean())
            
            # Calculate and display R^2 score for individual sites
            y_pred = slope * filtered_df[x_col_name] + intercept
            r2 = r2_score(filtered_df[y_col_name], y_pred)
            axis.text(x_fit[-1], y_fit[-1], f"$R^2={r2:.2f}$", color=color, fontsize=12, ha='left', va='center')

    # Global linear regression for all points in the current subplot
    global_slope, global_intercept = np.polyfit(df[x_col_name], df[y_col_name], 1)
    x_fit_global = np.linspace(df[x_col_name].min(), df[x_col_name].max(), 100)
    y_fit_global = global_slope * x_fit_global + global_intercept
    axis.plot(x_fit_global, y_fit_global, color='black', linestyle='-', linewidth=3)

    # Calculate and display global R^2 score
    y_pred_global = global_slope * df[x_col_name] + global_intercept
    r2_global = r2_score(df[y_col_name], y_pred_global)
    axis.text(x_fit_global[-1], y_fit_global[-1], f"$R^2={r2_global:.2f}$", color='black', fontsize=12, ha='left', va='center')

    if include_label:
        axis.legend(fontsize=legend_fontsize)
    
    return slopes, intercepts, avg_xlf_ips


# Function to plot data with linear regression and display R^2 score
def plot_data2(axis, df, x_col_name, y_col_name, mapping, include_label=False, aa=0.7, ss=60, lw=0, label_fontsize=10, legend_fontsize=10):
    """
    Plots data with simple and multiple linear regression, comparing R² scores.

    Args:
    axis (matplotlib axis): Axis object to plot on.
    df (DataFrame): The input dataframe.
    x_col_name (str): Name of the x-axis variable.
    y_col_name (str): Name of the y-axis variable.
    mapping (dict): Mapping of sample categories to color and marker.
    include_label (bool): Whether to include legend labels.
    aa (float): Transparency (alpha) for scatter points.
    ss (int): Size of scatter points.
    lw (float): Line width for scatter.
    label_fontsize (int): Font size for axis labels.
    legend_fontsize (int): Font size for legend.

    Returns:
    tuple: Standard deviations and R² differences between simple and multiple regression.
    """
    std_devs = []
    r2_diffs = []
    
    for start_str, (color, marker) in mapping.items():
        mask = df['SAMPLE'].str.startswith(start_str)
        label = f"{start_str} Site" if include_label else None
        
        # Scatter plot
        axis.scatter(df[x_col_name][mask], df[y_col_name][mask], alpha=aa, s=ss, linewidth=lw, c=color, marker=marker, label=label)
        
        # Simple linear regression
        if mask.sum() > 0:  # Ensure there are points to fit
            slope, intercept = np.polyfit(df[x_col_name][mask], df[y_col_name][mask], 1)
            x_fit = np.linspace(df[x_col_name][mask].min(), df[x_col_name][mask].max(), 100)
            y_fit = slope * x_fit + intercept
            axis.plot(x_fit, y_fit, color=color, linestyle='-', linewidth=2)
            
            # Calculate and display simple linear regression R^2 score
            y_pred_simple = slope * df[x_col_name][mask] + intercept
            r2_simple = r2_score(df[y_col_name][mask], y_pred_simple)
            axis.text(x_fit[-1], y_fit[-1], f"$R^2={r2_simple:.2f}$", color=color, fontsize=12)
            
            # Multiple linear regression with Xlf_IP as an additional feature
            X = df[[x_col_name, 'Klf']][mask]
            y = df[y_col_name][mask]
            model = LinearRegression().fit(X, y)
            y_pred_multi = model.predict(X)
            r2_multi = r2_score(y, y_pred_multi)
            
            # Display multiple linear regression R^2 score
            #axis.text(x_fit[-1], y_fit[-1] * 0.9, f"$R^2 Xlf={r2_multi:.2f}$", color=color, fontsize=12)
            
            # Save standard deviation and R^2 difference
            std_dev = df[y_col_name][mask].std()
            std_devs.append(std_dev)
            r2_diff = r2_multi - r2_simple
            r2_diffs.append(r2_diff)
    
    if include_label:
        axis.legend(fontsize=legend_fontsize)
    
    return std_devs, r2_diffs


# Enhanced 3D plotting function with axis labels
def plot_3d(df, x, y, z, X, Y, Z, elev=30, azim=30):
    """
    Plots a 3D scatter plot and surface plot with specified elevation and azimuth angles.

    Args:
    df (DataFrame): The input dataframe.
    x (str): Name of the x-axis variable.
    y (str): Name of the y-axis variable.
    z (str): Name of the z-axis variable.
    X (ndarray): X-axis grid for the surface plot.
    Y (ndarray): Y-axis grid for the surface plot.
    Z (ndarray): Z-axis values for the surface plot.
    elev (int): Elevation angle for the 3D plot.
    azim (int): Azimuth angle for the 3D plot.

    Returns:
    None
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[x], df[y], df[z], c="navy", s=15)
    ax.plot_surface(X, Y, Z, alpha=0.5, color='orange', edgecolor='none')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    plt.show()


def bars_plot(feature_sets, test_errors_summary, train_errors_summary, target_name):
    """
    Plots a bar chart comparing test and train errors for different feature sets.

    Args:
    feature_sets (list): List of feature set names.
    test_errors_summary (array): Test error summary values.
    train_errors_summary (array): Train error summary values.
    target_name (str): Target variable name for y-axis label.

    Returns:
    None
    """
    # Ensure the output folder exists, create it if it doesn't
    output_folder = 'figures_output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    fig, ax = plt.subplots()
    width = 0.35  # the width of the bars

    x = np.arange(len(feature_sets))
    rects1 = ax.bar(x - width/2, test_errors_summary, width, color = 'red', label='Test')
    rects2 = ax.bar(x + width/2, train_errors_summary, width, color = 'blue', label='Train')

    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_ylabel(target_name+' R2 scores')
    ax.set_xticks(range(len(test_errors_summary)), feature_sets, rotation = 15)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    # Full file path
    full_file_path = os.path.join(output_folder, 'Bar_plot_'+target_name)

    # Save and show the figure
    plt.savefig(full_file_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_kfdepth(df, var1, var2, mapping, onexone_line=False, log_scale=False):
    """
    Plots scatter plots of Kfd against depth in two subplots with trend lines.

    Args:
    df (DataFrame): The input dataframe.
    var1 (str): The variable name for Kfd.
    var2 (str): The variable name for depth.
    mapping (dict): Mapping of sample categories to color and marker.
    onexone_line (bool): Whether to include a 1:1 line.
    log_scale (bool): Whether to use logarithmic scale.

    Returns:
    None
    """
    # Drop rows where either var1 or var2 has NaN values
    df_clean = df.dropna(subset=[var1, var2])

    # Create a figure with two subplots sharing the y-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    fig.subplots_adjust(wspace=0.02)  # Adjust the space between the subplots

    # Variables to store min and max for Kfd (var1)
    kfd_min, kfd_max = float('inf'), float('-inf')

    for start_str, (color, marker) in mapping.items():
        mask = df_clean['SAMPLE'].str.startswith(start_str) & df_clean[var1].notna() & df_clean[var2].notna()
        filtered_df = df_clean[mask]

        if filtered_df.empty:
            continue

        # Update min and max for Kfd to ensure consistent scale
        kfd_min = min(kfd_min, filtered_df[var1].min())
        kfd_max = max(kfd_max, filtered_df[var1].max())

        # Linear regression to find the trend
        X = filtered_df[var1].values.reshape(-1, 1)
        Y = filtered_df[var2].values
        reg = LinearRegression().fit(X, Y)
        trend = reg.coef_[0]

        # Calculate Pearson correlation coefficient
        corr, _ = pearsonr(filtered_df[var1], filtered_df[var2])

        # Generate linear trend line
        x_range = np.linspace(min(X), max(X), 100)
        y_trend_line = reg.predict(x_range.reshape(-1, 1))

        # Determine which subplot to use based on the trend
        if trend > 0:
            # Plot data points and trend line in the left subplot
            ax1.scatter(filtered_df[var1], -filtered_df[var2], color=color, marker=marker, label=f'{start_str} (corr={corr:.2f})')
            ax1.plot(x_range, -y_trend_line, color=color, linestyle='--')
        else:
            # Plot data points and trend line in the right subplot
            ax2.scatter(filtered_df[var1], -filtered_df[var2], color=color, marker=marker, label=f'{start_str} (corr={corr:.2f})')
            ax2.plot(x_range, -y_trend_line, color=color, linestyle='--')

    # Set properties for both subplots
    for ax in [ax1, ax2]:
        ax.set_xlabel(var1)
        if ax == ax1:
            ax.set_ylabel('Depth [cm]')
        ax.grid(True)
        ax.legend(fontsize='small')

        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')

        # Invert the Y-axis for depth from maximum to zero
        ax.invert_yaxis()

    # Set the same x-axis limits for both subplots for Kfd
    ax1.set_xlim(kfd_min, kfd_max)
    ax2.set_xlim(kfd_min, kfd_max)

    # Ensure the output directory exists
    folder_path = 'figures_output/'
    os.makedirs(folder_path, exist_ok=True)
    filename = f"{var1}_{var2}_{'log' if log_scale else 'linear'}_subplots.png"
    plt.savefig(folder_path + filename)

    plt.show()


# Function to fit a polynomial regression model and plot the results in 2D/3D
def fit_and_plot(df, x_cols, y_col, degree, mapping, ss=60, lw=0):
    """
    Fits a polynomial regression model and plots the results in 2D or 3D.

    Args:
    df (DataFrame): The input dataframe.
    x_cols (list): List of predictor variable names.
    y_col (str): The target variable name.
    degree (int): The degree of the polynomial fit.
    mapping (dict): Mapping of sample categories to color.
    ss (int): Scatter point size.
    lw (float): Line width for scatter plot.

    Returns:
    None
    """
    x = df[x_cols]
    y = df[y_col]
    
    # Generate polynomial features
    poly = PolynomialFeatures(degree)
    x_poly = poly.fit_transform(x)
    
    # Fit a linear model
    model = LinearRegression()
    model.fit(x_poly, y)
    print(pd.DataFrame(zip(x_poly, model.coef_)))

    print('model.coef_', model.coef_)
    print('model.intercept_', model.intercept_)

    if len(x_cols) == 1:  # If there's only one predictor
        x_plot = np.linspace(x[x_cols[0]].min(), x[x_cols[0]].max(), 300).reshape(-1, 1)
        x_plot_poly = poly.transform(x_plot)
        y_plot = model.predict(x_plot_poly)
        y_plot = np.maximum(0, y_plot)  # Ensure y_plot is non-negative
        
        fig = go.Figure(data=[go.Scatter(x=x[x_cols[0]], y=y, mode='markers', name='Data points'),
                              go.Scatter(x=x_plot[:, 0], y=y_plot, mode='lines', name=f'Polynomial fit degree {degree}')])
        fig.update_layout(title='Polynomial Fit', xaxis_title=x_cols[0], yaxis_title=y_col, yaxis=dict(range=[0, max(y_plot)]))
        fig.show()
        
    elif len(x_cols) == 2:  # If there are two predictors
        # Generate grid for plots
        x0_range = np.linspace(x[x_cols[0]].min(), x[x_cols[0]].max(), 50)
        x1_range = np.linspace(x[x_cols[1]].min(), x[x_cols[1]].max(), 50)
        x0_grid, x1_grid = np.meshgrid(x0_range, x1_range)
        x_grid = np.c_[x0_grid.ravel(), x1_grid.ravel()]
        
        # Predict y values on grid
        x_grid_poly = poly.transform(x_grid)
        y_grid = model.predict(x_grid_poly).reshape(x0_grid.shape)
        y_grid = np.maximum(0, y_grid)  # Ensure y_grid is non-negative

        # Calculate R² score
        y_pred = model.predict(x_poly)
        r2 = r2_score(y, y_pred)
        
        # Assign color to each point based on mapping
        colors = []
        for sample in df['SAMPLE']:
            for start_str, (color, marker) in mapping.items():
                if sample.startswith(start_str):
                    colors.append(color)
                    break
            else:
                colors.append('gray')  # default color if no match
        
        # 3D surface plot with transparency and reversed color gradient
        fig = go.Figure(data=[
            go.Scatter3d(
                x=x[x_cols[0]], 
                y=x[x_cols[1]], 
                z=y, 
                mode='markers', 
                marker=dict(size=5, color=colors), 
                name='Data points'
            ),
            go.Surface(
                x=x0_range, 
                y=x1_range, 
                z=y_grid, 
                opacity=0.6,  # Adjust transparency here
                colorscale='Bluered_r',  # Reverse gradient from blue to red (yellow)
                name='Polynomial Surface'
            )
        ])
        
        fig.update_layout(
            title=f'3D Polynomial Fit (R² = {r2:.2f})',
            scene=dict(
                xaxis_title=x_cols[0], 
                yaxis_title=x_cols[1], 
                zaxis_title=y_col
            )
        )
        
        # Show the interactive figure
        fig.show()
        
        # Save high-resolution image
        #pio.write_image(fig, 'figures_output/3D_Plot_High_Resolution.png', width=1920, height=1080, scale=3)
        
    # Create static images from different perspectives using Matplotlib
    fig_static, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})

    # Set the color map for the surface plot to be similar to Plotly's color scale
    cmap = plt.get_cmap('coolwarm')

    # Plot data points and surface for all three perspectives
    # First perspective (default)
    axes[0].scatter(x[x_cols[0]], x[x_cols[1]], y, c=colors, s=ss, edgecolor='k')
    axes[0].plot_surface(x0_grid, x1_grid, y_grid, alpha=0.6, cmap=cmap)
    axes[0].view_init(elev=15, azim=45)
    axes[0].set_xlabel(x_cols[0])
    axes[0].set_ylabel(x_cols[1])
    axes[0].set_zlabel(y_col)
    axes[0].set_title(f"3D Polynomial Fit (R² = {r2:.2f})")

    # Second perspective (top view)
    axes[1].scatter(x[x_cols[0]], x[x_cols[1]], y, c=colors, s=ss, edgecolor='k')
    axes[1].plot_surface(x0_grid, x1_grid, y_grid, alpha=0.6, cmap=cmap)
    axes[1].view_init(elev=25, azim=135)
    axes[1].set_xlabel(x_cols[0])
    axes[1].set_ylabel(x_cols[1])
    axes[1].set_zlabel(y_col)

    # Third perspective (side view)
    axes[2].scatter(x[x_cols[0]], x[x_cols[1]], y, c=colors, s=ss, edgecolor='k')
    axes[2].plot_surface(x0_grid, x1_grid, y_grid, alpha=0.6, cmap=cmap)
    axes[2].view_init(elev=45, azim=225)
    axes[2].set_xlabel(x_cols[0])
    axes[2].set_ylabel(x_cols[1])
    axes[2].set_zlabel(y_col)

    plt.tight_layout()
    plt.show()


def partial_correlation(df, x_col, y_col, control_cols):
    """
    Calculate the partial correlation between two columns of a dataframe controlling for other columns.

    Args:
    df (DataFrame): The input dataframe.
    x_col (str): The name of the first variable column.
    y_col (str): The name of the second variable column.
    control_cols (list of str): The names of the control variable columns.

    Returns:
    float: The partial correlation coefficient.
    """
    # Ensure control_cols is a list even if a single column name is provided
    if isinstance(control_cols, str):
        control_cols = [control_cols]

    # Drop missing values from the columns of interest
    df = df[[x_col, y_col] + control_cols].dropna()

    # Function to get residuals from a regression on control variables
    def get_residuals(col):
        X = df[control_cols]
        X = sm.add_constant(X)  # Adds a constant term to the predictor
        y = df[col]
        model = sm.OLS(y, X).fit()
        residuals = model.resid
        return residuals

    # Calculate residuals for both variables of interest
    x_residuals = get_residuals(x_col)
    y_residuals = get_residuals(y_col)

    # Calculate the correlation of the residuals
    partial_corr, _ = pearsonr(x_residuals, y_residuals)

    # Plotting for visualization
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot x_col vs the first control variable
    sns.regplot(x=df[x_col], y=df[control_cols[0]], ax=axs[0])
    axs[0].set_title(f'{x_col} vs {control_cols[0]}')

    # Plot y_col vs the first control variable (or same variable if only one control)
    sns.regplot(x=df[y_col], y=df[control_cols[0]], ax=axs[1])
    axs[1].set_title(f'{y_col} vs {control_cols[0]}')

    # Plot the residuals scatter plot
    sns.scatterplot(x=x_residuals, y=y_residuals, ax=axs[2])
    axs[2].set_title(f'Residuals: {x_col} vs {y_col}\nPartial Corr: {partial_corr:.2f}')

    plt.tight_layout()
    plt.show()

    return partial_corr


# Function to plot a correlation matrix with a significance mask
def plot_correlation_matrix(corr_df, p_value_df, labels, p_value_mask=0.05, filename="Fig2.png", folder_path="figures_output/"):
    """
    Plots a correlation matrix with a significance mask for p-values.

    Args:
    corr_df (DataFrame): The correlation matrix dataframe.
    p_value_df (DataFrame): The p-value matrix dataframe.
    labels (list): List of labels for the heatmap axes.
    p_value_mask (float): Significance level for masking p-values.
    filename (str): The filename for saving the plot.
    folder_path (str): The folder path to save the plot.

    Returns:
    None
    """
    # Create a mask for significant p-values
    significant_mask = p_value_df < p_value_mask

    # Mask for the upper triangle
    mask = np.triu(np.ones_like(corr_df, dtype=bool))

    # Combine the masks
    final_mask = mask | ~significant_mask

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr_df, 
        mask=final_mask, 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        square=True, 
        linewidths=.5, 
        vmin=-1, 
        vmax=1, 
        annot_kws={"size": 10, "color": "black"}
    )

    # Adjust layout for better readability
    plt.xticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=90, ha='right', fontsize=10)
    plt.yticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=0, fontsize=10)

    # Save the plot
    plt.savefig(folder_path + filename, dpi=300)

    # Show the plot
    plt.show()