import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score
import seaborn as sns
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objects as go
from scipy.stats import pearsonr


def plot_sing(df, var1, var2, mapping, onexone_line=False, log_scale=False):
    for start_str, (color, marker) in mapping.items():
        mask = df['SAMPLE'].str.startswith(start_str)
        filtered_df = df[mask]
        plt.scatter(filtered_df[var1], filtered_df[var2], color=color, marker=marker, label=start_str)
        plt.grid(True)

    r2 = r2_score(df[var1], df[var2])
    #plt.text(e-5, 0.2, f'R^2={r2:.2f}')

    # Adding a 1:1 line
    if onexone_line:
        x = np.linspace(min(df[var1]), max(df[var1]), 100)
        plt.plot(x, x, color='black', linestyle='--', label=f'R^2={r2:.2f}' )

    # Setting both axes to logarithmic scale
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')    
        
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.legend(fontsize='small')  # Adjusting the font size of the legend

    folder_path = 'figures_output/'
    filename = var1+var2+str(log_scale)+".png"
    plt.savefig(folder_path + filename)


def plot_data2(df, axis, x_col, y_col, mapping, include_label=False, aa=0.7, ss=60, lw=0, label_fontsize=10, legend_fontsize=10):
    corr = round(np.corrcoef(x_col, y_col)[0][1], 2)
    for start_str, (color, marker) in mapping.items():
        mask = df['SAMPLE'].str.startswith(start_str)
        label = f"{start_str} Site" if include_label else None
        axis.scatter(x_col[mask], y_col[mask], alpha=aa, s=ss, linewidth=lw, c=color, marker=marker, label=label)
    # Apply logarithmic scale to the y-axis
    #axis.set_xscale('log')

    axis.text(0, 0.98, s=f'Corr = {corr}', fontsize=label_fontsize, 
              verticalalignment='top', horizontalalignment='left', 
              transform=axis.transAxes)
    if include_label:
        axis.legend(fontsize=legend_fontsize)
        

def create_and_save_plots2(df, target_var, pred1, pred2, mapping):
    # Ensure the output folder exists, create it if it doesn't
    output_folder = 'figures_output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create subplots with shared x-axes within each column
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharey=True, sharex='col')
    axes = axes.flatten()  # Flatten to easily iterate over

    # Define your conditions for plotting
    conditions = [
        (df[pred1], df[target_var], 0, max(df[pred1])),  
        (df[pred2], df[target_var], 0, max(df[pred2])),  
        (df[pred1][df['Archaeology'] == 1], df[target_var][df['Archaeology'] == 1], 0, max(df[pred1]), True),  
        (df[pred2][df['Archaeology'] == 1], df[target_var][df['Archaeology'] == 1], 0, max(df[pred2])),
        (df[pred1][df['Archaeology'] == 0], df[target_var][df['Archaeology'] == 0], 0, max(df[pred1])),
        (df[pred2][df['Archaeology'] == 0], df[target_var][df['Archaeology'] == 0], 0, max(df[pred2])),
    ]
    # Loop through conditions and plot
    for i, (ax, (x, y, xlim_lower, xlim_upper, *include_label)) in enumerate(zip(axes, conditions)):
        plot_data2(df, ax, x, y, mapping, include_label=bool(include_label))
        ax.set_xlim(xlim_lower, xlim_upper)  # Set x-axis limits
        ax.grid(True)  # Enable the grid
        ax.tick_params(axis='y', labelsize=12) 
        ax.tick_params(axis='x', labelsize=12) 
        # Set y-axis label for the first and fourth plot (left column)
        if i == 0 or i == 2 or i == 4:
            ax.set_ylabel(f'{target_var} [-]', fontsize=16)

    # Set legend for the third subplot (index 2)
    axes[2].legend(loc='upper right', fontsize=10)
    axes[4].set_xlabel(pred1, fontsize=16)  # Set x-axis label
    axes[5].set_xlabel(pred2, fontsize=16)  # Set x-axis label

    plt.tight_layout(pad=1.0)  # Adjust layout

    # Full file path
    file_name = target_var + pred1 + pred2 + '.png'
    full_file_path = os.path.join(output_folder, file_name)

    # Save and show the figure
    plt.savefig(full_file_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_data(df, axis, x_col, y_col, mapping, include_label=False, aa=0.7, ss=60, lw=0, label_fontsize=10, legend_fontsize=10):
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
        

def create_and_save_plots(df, target_var, pred1, pred2):
    # Ensure the output folder exists, create it if it doesn't
    output_folder = 'figures_output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create subplots with shared x-axes within each column
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharey=True, sharex='col')
    axes = axes.flatten()  # Flatten to easily iterate over

    # Define your conditions for plotting
    conditions = [
        (df[pred1], df[target_var], 0, 80),  
        (df[pred2], df[target_var], 0, 40),  
        (df[pred1][df['Archaeology'] == 1], df[target_var][df['Archaeology'] == 1], 0, 80, True),  
        (df[pred2][df['Archaeology'] == 1], df[target_var][df['Archaeology'] == 1], 0, 40),
        (df[pred1][df['Archaeology'] == 0], df[target_var][df['Archaeology'] == 0], 0, 80),
        (df[pred2][df['Archaeology'] == 0], df[target_var][df['Archaeology'] == 0], 0, 40),
    ]
    # Loop through conditions and plot
    for i, (ax, (x, y, xlim_lower, xlim_upper, *include_label)) in enumerate(zip(axes, conditions)):
        plot_data(ax, x, y, include_label=bool(include_label))
        ax.set_xlim(xlim_lower, xlim_upper)  # Set x-axis limits
        ax.grid(True)  # Enable the grid
        ax.tick_params(axis='y', labelsize=12) 
        ax.tick_params(axis='x', labelsize=12) 
        # Set y-axis label for the first and fourth plot (left column)
        if i == 0 or i == 2 or i == 4:
            ax.set_ylabel(f'{target_var} [-]', fontsize=16)

    # Set legend for the third subplot (index 2)
    axes[2].legend(loc='upper right', fontsize=10)
    axes[4].set_xlabel(pred1, fontsize=16)  # Set x-axis label
    axes[5].set_xlabel(pred2, fontsize=16)  # Set x-axis label

    plt.tight_layout(pad=1.0)  # Adjust layout

    # Full file path
    file_name = target_var + pred1 + pred2 + '.png'
    full_file_path = os.path.join(output_folder, file_name)

    # Save and show the figure
    plt.savefig(full_file_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_data1(axis, df, x_col_name, y_col_name, mapping, include_label=False, aa=0.7, ss=60, lw=0, label_fontsize=10, legend_fontsize=10):
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
            avg_xlf_ips.append(filtered_df['Xlf_IP'].mean())
            
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
    slopes = []
    intercepts = []
    avg_xlf_ips = []
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
            X = df[[x_col_name, 'Xlf_IP']][mask]
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


def plot1(fig1, ax1, ax2, ax3, ax4, ax5, ax6):
    fig1.tight_layout()

    #ax1.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax1.tick_params(axis='y', labelsize=12) 
    ax1.tick_params(axis='x', labelsize=12) 
    #ax1.set_xlabel('Clay [%]', fontsize = 16) 
    ax1.set_ylabel('Klf_IP [m-3] all', fontsize = 16) 
    ax1.grid(True) 
    ax1.set_xlim(0, 80) 

    #ax2.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax2.tick_params(axis='y', labelsize=12) 
    ax2.tick_params(axis='x', labelsize=12) 
    #ax2.set_xlabel('CEC [meq/100g]', fontsize = 16) 
    #ax2.set_ylabel('Klf [m3/kg]', fontsize = 16) 
    ax2.grid(True) 
    ax2.set_xlim(0, 45) 

    #ax3.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax3.tick_params(axis='y', labelsize=12) 
    ax3.tick_params(axis='x', labelsize=12) 
    #ax3.set_xlabel('Clay [%]', fontsize = 16) 
    ax3.set_ylabel('Klf_IP [m-3] arch', fontsize = 16) 
    ax3.grid(True) 
    ax3.set_xlim(0, 80) 

    #ax4.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax4.tick_params(axis='y', labelsize=12) 
    ax4.tick_params(axis='x', labelsize=12) 
    #ax4.set_xlabel('CEC [meq/100g]', fontsize = 16) 
    #ax4.set_ylabel('Klf [m3/kg]', fontsize = 16) 
    ax4.grid(True) 
    ax4.set_xlim(0, 45) 

    #ax5.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax5.tick_params(axis='y', labelsize=12) 
    ax5.tick_params(axis='x', labelsize=12) 
    ax5.set_xlabel('Clay [%]', fontsize = 16) 
    ax5.set_ylabel('Klf_IP [m-3] no arch', fontsize = 16) 
    ax5.grid(True) 
    ax5.set_xlim(0, 80) 

    #ax6.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax6.tick_params(axis='y', labelsize=12) 
    ax6.tick_params(axis='x', labelsize=12) 
    ax6.set_xlabel('CEC [meq/100g]', fontsize = 16) 
    #ax6.set_ylabel('Klf [m3/kg]', fontsize = 16) 
    ax6.grid(True) 
    ax6.set_xlim(0, 45) 


def plot2(fig2, ax1, ax2, ax3, ax4, ax5, ax6):
    fig2.tight_layout()

    #ax1.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax1.tick_params(axis='y', labelsize=12) 
    ax1.tick_params(axis='x', labelsize=12) 
    #ax1.set_xlabel('Clay [%]', fontsize = 16) 
    ax1.set_ylabel('Kfd_abs [m3/kg] all', fontsize = 16) 
    ax1.grid(True) 
    ax1.set_xlim(0, 80) 

    #ax2.legend(loc='upper right', fontsize = 8)
    #ax2.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax2.tick_params(axis='y', labelsize=12) 
    ax2.tick_params(axis='x', labelsize=12) 
    #ax2.set_xlabel('CEC [meq/100g]', fontsize = 16) 
    #ax2.set_ylabel('Kfd_abs [m3/kg]', fontsize = 16) 
    ax2.grid(True) 
    ax2.set_xlim(0, 45) 

    #ax3.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax3.tick_params(axis='y', labelsize=12) 
    ax3.tick_params(axis='x', labelsize=12) 
    #ax3.set_xlabel('Clay [%]', fontsize = 16) 
    ax3.set_ylabel('Kfd_abs [m3/kg] arch', fontsize = 16) 
    ax3.grid(True) 
    ax3.set_xlim(0, 80) 

    #ax4.legend(loc='upper right', fontsize = 8)
    #ax4.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax4.tick_params(axis='y', labelsize=12) 
    ax4.tick_params(axis='x', labelsize=12) 
    #ax4.set_xlabel('CEC [meq/100g]', fontsize = 16) 
    #ax4.set_ylabel('Kfd_abs [m3/kg]', fontsize = 16) 
    ax4.grid(True) 
    ax4.set_xlim(0, 45) 

    #ax5.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax5.tick_params(axis='y', labelsize=12) 
    ax5.tick_params(axis='x', labelsize=12) 
    ax5.set_xlabel('Clay [%]', fontsize = 16) 
    ax5.set_ylabel('Kfd_abs [m3/kg] no arch', fontsize = 16) 
    ax5.grid(True) 
    ax5.set_xlim(0, 80) 

    #ax6.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax6.tick_params(axis='y', labelsize=12) 
    ax6.tick_params(axis='x', labelsize=12) 
    ax6.set_xlabel('CEC [meq/100g]', fontsize = 16) 
    #ax6.set_ylabel('Kfd_abs [m3/kg]', fontsize = 16) 
    ax6.grid(True) 
    ax6.set_xlim(0, 45) 


def plot3(fig3, ax1, ax2, ax3, ax4, ax5, ax6):
    fig3.tight_layout()

    ax1.legend(loc='upper right', fontsize = 8)
    #ax1.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax1.tick_params(axis='y', labelsize=12) 
    ax1.tick_params(axis='x', labelsize=12) 
    #ax1.set_xlabel('Clay [%]', fontsize = 16) 
    ax1.set_ylabel('Kfd [%] all', fontsize = 16) 
    ax1.grid(True) 
    ax1.set_ylim(0, 5e-3)  
    ax1.set_xlim(0, 80) 
    ax1.legend(loc='upper right', fontsize = 10)

    #ax2.legend(loc='upper right', fontsize = 8)
    #ax2.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax2.tick_params(axis='y', labelsize=12) 
    ax2.tick_params(axis='x', labelsize=12) 
    #ax2.set_xlabel('CEC [meq/100g]', fontsize = 16) 
    #ax2.set_ylabel('Kfd [%]', fontsize = 16) 
    ax2.grid(True) 
    ax2.set_ylim(0, 5e-3)  
    ax2.set_xlim(0, 45) 
    ax2.legend(loc='upper right', fontsize = 10)

    ax3.legend(loc='upper right', fontsize = 8)
    #ax3.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax3.tick_params(axis='y', labelsize=12) 
    ax3.tick_params(axis='x', labelsize=12) 
    #ax3.set_xlabel('Clay [%]', fontsize = 16) 
    ax3.set_ylabel('Kfd [%] arch', fontsize = 16) 
    ax3.grid(True) 
    ax3.set_ylim(0, 5e-3)  
    ax3.set_xlim(0, 80) 
    ax3.legend(loc='upper right', fontsize = 10)

    #ax4.legend(loc='upper right', fontsize = 8)
    #ax4.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax4.tick_params(axis='y', labelsize=12) 
    ax4.tick_params(axis='x', labelsize=12) 
    #ax4.set_xlabel('CEC [meq/100g]', fontsize = 16) 
    #ax4.set_ylabel('Kfd [%]', fontsize = 16) 
    ax4.grid(True) 
    ax4.set_ylim(0, 5e-3)  
    ax4.set_xlim(0, 45) 
    ax4.legend(loc='upper right', fontsize = 10)

    #ax5.legend(loc='upper right', fontsize = 8)
    #ax5.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax5.tick_params(axis='y', labelsize=12) 
    ax5.tick_params(axis='x', labelsize=12) 
    ax5.set_xlabel('Clay [%]', fontsize = 16) 
    ax5.set_ylabel('Kfd [%] no arch', fontsize = 16) 
    ax5.grid(True) 
    ax5.set_ylim(0, 5e-3)  
    ax5.set_xlim(0, 80) 
    ax5.legend(loc='upper right', fontsize = 10)

    #ax6.legend(loc='upper right', fontsize = 8)
    #ax6.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax6.tick_params(axis='y', labelsize=12) 
    ax6.tick_params(axis='x', labelsize=12) 
    ax6.set_xlabel('CEC [meq/100g]', fontsize = 16) 
    #ax6.set_ylabel('Kfd [%]', fontsize = 16) 
    ax6.grid(True) 
    ax6.set_ylim(0, 5e-3)  
    ax6.set_xlim(0, 45) 
    ax6.legend(loc='upper right', fontsize = 10)


def ThreeD1(axa, plt):
    plt.rcParams["figure.figsize"] = (6,4) 
    plt.rcParams["figure.dpi"] = 150
    axa.tick_params(axis='y', labelsize=10) 
    axa.tick_params(axis='x', labelsize=10) 
    axa.set_xlabel(" Clay" , fontweight='bold', fontsize=12) 
    axa.set_ylabel(" F1mass" , fontweight='bold', fontsize=12) 
    axa.set_zlabel(" CEC [meq/100g]" , fontweight='bold', fontsize=12) 
    axa.set_title("  " , fontweight='bold', fontsize=16) 
    axa.set_zlim(0, 80) 


def valthe(ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, plt):
    import numpy as np
    ax1.legend(loc='lower right', fontsize = 12)
    ax1.set_title(" " , fontweight='bold', fontsize=25) 
    ax1.tick_params(axis='y', labelsize=18) 
    ax1.tick_params(axis='x', labelsize=18) 
    ax1.set_xlabel('Volumetric content [%]', fontsize = 18) 
    ax1.set_ylabel('Depth [cm]', fontsize = 18) 
    #ax1.set_yticks(np.arange(0, -120, -3))
    ax1.set_xticks(np.arange(1, 60, 5))
    ax1.grid(True) 
    #ax1.set_ylim(0, yc)  
    ax1.set_xlim(0, 60) 

    ax2.legend(loc='lower left', fontsize = 12) 
    ax2.set_title(" " , fontweight='bold', fontsize=25) 
    ax2.tick_params(axis='y', labelsize=18) 
    ax2.tick_params(axis='x', labelsize=18) 
    #ax2.set_ylabel('Depth [cm]', fontsize = 16) 
    ax2.set_xlabel('Original Bulk EC [mS/m]', fontsize = 18) 
    ax2.grid(True) 
    #ax2.set_ylim(0, yc) 
    ax2.set_xlim(0, 1.6)

    ax3.legend(loc='upper right', fontsize = 12) 
    ax3.set_title(" " , fontweight='bold', fontsize=25) 
    ax3.tick_params(axis='y', labelsize=18) 
    ax3.tick_params(axis='x', labelsize=18) 
    ax3.set_ylabel('Depth [cm]', fontsize = 18) 
    ax3.set_xlabel('Real Relative Permittivity [-]', fontsize = 18) 
    ax3.grid(True) 
    #ax3.set_ylim(0, yc) 
    ax3.set_xlim(0, 20)

    ax4.legend(loc='upper right', fontsize = 12) 
    ax4.set_title(" " , fontweight='bold', fontsize=25) 
    ax4.tick_params(axis='y', labelsize=18) 
    ax4.tick_params(axis='x', labelsize=18) 
    #ax4.set_ylabel('Depth [cm]', fontsize = 16) 
    ax4.set_xlabel('Corrected Bulk EC [mS/m]', fontsize = 18) 
    ax4.grid(True) 
    #ax4.set_ylim(0, yc) 
    ax4.set_xlim(0, 1.6)

    ax6.legend(loc='upper left', fontsize = 12) 
    ax6.set_title(" " , fontweight='bold', fontsize=25) 
    ax6.tick_params(axis='y', labelsize=18) 
    ax6.tick_params(axis='x', labelsize=18) 
    #ax6.set_ylabel('Depth [cm]', fontsize = 16) 
    ax6.set_xlabel('Water Conductivity [S/m]', fontsize = 18) 
    ax6.grid(True) 
    #ax6.set_ylim(-80, 0) 
    ax6.set_xlim(0, 0.22)
    #ax6.set_yticks(np.arange(0, 0.25, 0.02))
    ax6.set_xticks(np.arange(0, 0.22, 0.02))

    ax5.legend(loc='lower right', fontsize = 12) 
    ax5.set_title(" " , fontweight='bold', fontsize=25) 
    ax5.tick_params(axis='y', labelsize=18) 
    ax5.tick_params(axis='x', labelsize=18) 
    ax5.set_ylabel('Depth', fontsize = 16) 
    ax5.set_xlabel('Magnetic suceptibility [SI$*10^{-3}$]', fontsize = 18) 
    ax5.grid(True) 
    #ax5.set_ylim(0, 35) 
    ax5.set_xlim(0, 0.25)
    
    ax7.legend(loc='upper left', fontsize = 12) 
    #ax7.set_title("Field calibration curve " , fontweight='bold', fontsize=25) 
    ax7.tick_params(axis='y', labelsize=18) 
    ax7.tick_params(axis='x', labelsize=18) 
    ax7.set_xlabel('Real Rel. Permittivity', fontsize = 16) 
    ax7.set_ylabel('Volumetric water contect [m3/m3]', fontsize = 18) 
    ax7.grid(True) 
    ax7.set_xlim(0.001, 40) 
    ax7.set_ylim(5, 45)
    #ax7.set_yticks(np.arange(0, 0.25, 0.02))
    #ax7.set_xticks(np.arange(0, 0.22, 0.02))

    ax8.legend(loc='lower right', fontsize = 12) 
    #ax8.set_title("Field calibration curve " , fontweight='bold', fontsize=25) 
    ax8.tick_params(axis='y', labelsize=18) 
    ax8.tick_params(axis='x', labelsize=18) 
    ax8.set_xlabel('Real Rel. Permittivity', fontsize = 16) 
    ax8.set_ylabel('Volumetric water contect [m3/m3]', fontsize = 18) 
    ax8.grid(True) 
    #ax8.set_xlim(0.001, 40) 
    #ax8.set_ylim(5, 45)
    #ax8.set_yticks(np.arange(0, 0.25, 0.02))
    #ax8.set_xticks(np.arange(0, 0.22, 0.02))
    
    
def fielworkgraph(ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, plt):
    import numpy as np
    ax1.legend(loc='upper left', fontsize = 8)
    ax1.set_title(" " , fontweight='bold', fontsize=25) 
    ax1.tick_params(axis='y', labelsize=18) 
    ax1.tick_params(axis='x', labelsize=18) 
    ax1.set_xlabel('Volumetric content [%]', fontsize = 18) 
    ax1.set_ylabel('Depth [cm]', fontsize = 18) 
    #ax1.set_yticks(np.arange(0, -120, -3))
    ax1.set_xticks(np.arange(0, 60, 5))
    ax1.grid(True) 
    #ax1.set_ylim(0, yc)  
    ax1.set_xlim(0, 60) 

    ax2.legend(loc='upper right', fontsize = 8) 
    ax2.set_title(" " , fontweight='bold', fontsize=25) 
    ax2.tick_params(axis='y', labelsize=18) 
    ax2.tick_params(axis='x', labelsize=18) 
    #ax2.set_ylabel('Depth [cm]', fontsize = 16) 
    ax2.set_xlabel('Original Bulk EC [mS/m]', fontsize = 18) 
    ax2.grid(True) 
    #ax2.set_ylim(0, yc) 
    ax2.set_xlim(0, 70)

    ax3.legend(loc='upper right', fontsize = 14) 
    ax3.set_title(" " , fontweight='bold', fontsize=25) 
    ax3.tick_params(axis='y', labelsize=18) 
    ax3.tick_params(axis='x', labelsize=18) 
    ax3.set_ylabel('Depth [cm]', fontsize = 18) 
    ax3.set_xlabel('Real Relative Permittivity [-]', fontsize = 18) 
    ax3.grid(True) 
    #ax3.set_ylim(0, yc) 
    ax3.set_xlim(0, 35)

    ax4.legend(loc='upper right', fontsize = 8) 
    ax4.set_title(" " , fontweight='bold', fontsize=25) 
    ax4.tick_params(axis='y', labelsize=18) 
    ax4.tick_params(axis='x', labelsize=18) 
    #ax4.set_ylabel('Depth [cm]', fontsize = 16) 
    ax4.set_xlabel('Corrected Bulk EC [mS/m]', fontsize = 18) 
    ax4.grid(True) 
    #ax4.set_ylim(0, yc) 
    ax4.set_xlim(0, 70)

    ax5.legend(loc='upper left', fontsize = 8) 
    ax5.set_title(" " , fontweight='bold', fontsize=25) 
    ax5.tick_params(axis='y', labelsize=18) 
    ax5.tick_params(axis='x', labelsize=18) 
    #ax5.set_ylabel('Depth [cm]', fontsize = 16) 
    ax5.set_xlabel('Water Conductivity [S/m]', fontsize = 18) 
    ax5.grid(True) 
    #ax5.set_ylim(-80, 0) 
    ax5.set_xlim(0, 0.22)
    #ax5.set_yticks(np.arange(0, 0.25, 0.02))
    ax5.set_xticks(np.arange(0, 0.22, 0.02))

    ax6.legend(loc='upper left', fontsize = 8) 
    ax6.set_title(" " , fontweight='bold', fontsize=25) 
    ax6.tick_params(axis='y', labelsize=18) 
    ax6.tick_params(axis='x', labelsize=18) 
    ax6.set_ylabel('Real Permitivitty [-]', fontsize = 16) 
    ax6.set_xlabel('Correctec Bulk Conductivity [mS/m]', fontsize = 18) 
    ax6.grid(True) 
    ax6.set_ylim(0, 35) 
    ax6.set_xlim(0, 50)
    
    ax7.legend(loc='upper left', fontsize = 8) 
    #ax7.set_title("Field calibration curve " , fontweight='bold', fontsize=25) 
    ax7.tick_params(axis='y', labelsize=18) 
    ax7.tick_params(axis='x', labelsize=18) 
    ax7.set_ylabel('Real Rel. Permittivity', fontsize = 16) 
    ax7.set_xlabel('Volumetric water contect [m3/m3]', fontsize = 18) 
    ax7.grid(True) 
    ax7.set_xlim(0.00, 70) 
    ax7.set_ylim(5, 90)
    #ax7.set_yticks(np.arange(0, 0.25, 0.02))
    #ax7.set_xticks(np.arange(0, 0.22, 0.02))

    ax8.legend(loc='lower right', fontsize = 8) 
    #ax8.set_title("Field calibration curve " , fontweight='bold', fontsize=25) 
    ax8.tick_params(axis='y', labelsize=18) 
    ax8.tick_params(axis='x', labelsize=18) 
    ax8.set_ylabel('Real Rel. Permittivity', fontsize = 16) 
    ax8.set_xlabel('Volumetric water contect [m3/m3]', fontsize = 18) 
    ax8.grid(True) 
    #ax8.set_xlim(0.001, 40) 
    #ax8.set_ylim(5, 45)
    #ax8.set_yticks(np.arange(0, 0.25, 0.02))
    #ax8.set_xticks(np.arange(0, 0.22, 0.02))

def moist_curve(axa, axb, axc, axd, axe, axf, axg, axh, plt):
    import numpy as np
    xlim = 50
    axa.legend(loc='upper left', fontsize = 14)
    axa.set_title("Water content vs Real Permittivity " , fontweight='bold', fontsize=25) 
    axa.tick_params(axis='y', labelsize=20) 
    axa.tick_params(axis='x', labelsize=20) 
    axa.set_xlabel('Volumetric water content [%]', fontsize = 22) 
    axa.set_ylabel('Real Permittivity', fontsize = 22) 
    axa.set_yticks(np.arange(0, 50, 5))
    axa.set_xticks(np.arange(0, xlim, 5))
    axa.grid(True) 
    axa.set_ylim(0, 50)  
    axa.set_xlim(0, xlim) 

    axb.legend(loc='upper left', fontsize = 14)
    axb.set_title("Water content vs Im. Permittivity " , fontweight='bold', fontsize=25) 
    axb.tick_params(axis='y', labelsize=20) 
    axb.tick_params(axis='x', labelsize=20) 
    axb.set_xlabel('Volumetric water content [%]', fontsize = 22) 
    axb.set_ylabel('Imaginary Permittivity', fontsize = 22) 
    axb.set_yticks(np.arange(0, 80, 10))
    axb.set_xticks(np.arange(0, xlim, 5))
    axb.grid(True) 
    axb.set_ylim(0, 80)  
    axb.set_xlim(0, xlim) 

    axc.legend(loc='upper left', fontsize = 14)
    axc.set_title("Water content vs EC" , fontweight='bold', fontsize=25) 
    axc.tick_params(axis='y', labelsize=20) 
    axc.tick_params(axis='x', labelsize=20) 
    axc.set_xlabel('Volumetric water content [%]', fontsize = 22) 
    axc.set_ylabel('EC [mS/m]', fontsize = 22) 
    axc.set_yticks(np.arange(0, 500, 50))
    axc.set_xticks(np.arange(0, xlim, 5))
    axc.grid(True) 
    axc.set_ylim(0, 500)  
    axc.set_xlim(0, xlim) 

    axd.legend(loc='upper left', fontsize = 14)
    axd.set_title("Water content vs EC" , fontweight='bold', fontsize=25) 
    axd.tick_params(axis='y', labelsize=20) 
    axd.tick_params(axis='x', labelsize=20) 
    axd.set_xlabel('Volumetric water content [%]', fontsize = 22) 
    axd.set_ylabel('EC [mS/m]', fontsize = 22) 
    axd.set_yticks(np.arange(0, 85, 10))
    axd.set_xticks(np.arange(0, xlim, 5))
    axd.grid(True) 
    axd.set_ylim(0, 85)  
    axd.set_xlim(0, xlim) 
    
    axe.legend(loc='upper left', fontsize = 14)
    axe.set_title("Water content vs Relaxation component " , fontweight='bold', fontsize=25) 
    axe.tick_params(axis='y', labelsize=20) 
    axe.tick_params(axis='x', labelsize=20) 
    axe.set_xlabel('Volumetric water content [%]', fontsize = 22) 
    axe.set_ylabel('Relaxation component [-]', fontsize = 22) 
    #axe.set_yticks(np.arange(0, 55, 5))
    axe.set_xticks(np.arange(0, xlim, 5))
    axe.grid(True) 
    #axe.set_ylim(0, 55)  
    axe.set_xlim(0, xlim) 
    
    axf.legend(loc='lower right', fontsize = 14)
    axf.set_title("EC vs Real permittivity" , fontweight='bold', fontsize=25) 
    axf.tick_params(axis='y', labelsize=18) 
    axf.tick_params(axis='x', labelsize=18) 
    axf.set_xlabel('EC [mS/m]', fontsize = 18) 
    axf.set_ylabel('Real Permittivity', fontsize = 18) 
    axf.set_yticks(np.arange(0, 50, 5))
    #axf.set_xticks(np.arange(0, 50, 5))
    axf.grid(True) 
    axf.set_ylim(0, 50)  
    #axf.set_xlim(0, 50) 
    
    axg.legend(loc='upper left', fontsize = 14)
    axg.set_title("Water content vs Real Permittivity " , fontweight='bold', fontsize=25) 
    axg.tick_params(axis='y', labelsize=18) 
    axg.tick_params(axis='x', labelsize=18) 
    axg.set_xlabel('Volumetric water content [%]', fontsize = 18) 
    axg.set_ylabel('Real Permittivity', fontsize = 18) 
    #axg.set_yticks(np.arange(0, 300, 20))
    axg.set_xticks(np.arange(0, 50, 5))
    axg.grid(True) 
    #axg.set_ylim(0, 80)  
    axg.set_xlim(0, 50) 
    
    #axh.legend(loc='upper left', fontsize = 14)
    #axh.set_title("Water content vs Real Permittivity " , fontweight='bold', fontsize=25) 
    #axh.tick_params(axis='y', labelsize=18) 
    #axh.tick_params(axis='x', labelsize=18) 
    #axh.set_xlabel('Volumetric water content [%]', fontsize = 18) 
    #axh.set_ylabel('Real Permittivity', fontsize = 18) 
    #axh.set_yticks(np.arange(0, 50, 5))
    #axh.set_xticks(np.arange(0, 50, 5))
    #axh.grid(True) 
    #axh.set_ylim(0, 50)  
    #axh.set_xlim(0, 50) 


def pltmsc(ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, lw, yc, plt):
    ax1.legend(loc='upper right', fontsize = 10)
    ax1.set_title("Susceptibility vs Sand" , fontweight='bold', fontsize=25) 
    ax1.tick_params(axis='y', labelsize=12) 
    ax1.tick_params(axis='x', labelsize=12) 
    ax1.set_xlabel('Sand [%]', fontsize = 16) 
    ax1.set_ylabel('Susceptibility [10−8 m3/kg]', fontsize = 16) 
    ax1.grid(True) 
    ax1.set_ylim(0, yc)  
    #ax1.set_xlim(1, 50) 
    ax1.legend(loc='upper right', fontsize = 10)

    ax2.legend(loc='upper left', fontsize = 10) 
    ax2.set_title("Susceptibility vs Organic matter" , fontweight='bold', fontsize=25) 
    ax2.tick_params(axis='y', labelsize=12) 
    ax2.tick_params(axis='x', labelsize=12) 
    ax2.set_ylabel('Susceptibility [10−8 m3/kg]', fontsize = 16) 
    ax2.set_xlabel('Organic matter [%]', fontsize = 16) 
    ax2.grid(True) 
    ax2.set_ylim(0, yc) 
    #ax2.set_xlim(10, 100000)
    ax2.legend(loc='upper left', fontsize = 10) 

    ax3.grid(True) 
    ax3.set_ylabel('Susceptibility [10−8 m3/kg]', fontsize = 16) 
    ax3.set_xlabel("Cobalt [mg/Kg]" , fontsize = 16) 
    ax3.legend(loc='upper left', fontsize = 10) 
    ax3.set_title("Susceptibility vs Cobalt" , fontweight='bold', fontsize=25) 
    ax3.tick_params(axis='y', labelsize=12) 
    ax3.tick_params(axis='x', labelsize=12) 
    ax3.set_ylim(0, yc) 
    #ax3.set_xlim(0, 60) 
    ax3.legend(loc='upper left', fontsize = 10) 

    ax12.legend(loc='upper right', fontsize = 10) 
    ax12.set_title("Susceptibility vs Copper" , fontweight='bold', fontsize=25) 
    ax12.tick_params(axis='y', labelsize=12) 
    ax12.tick_params(axis='x', labelsize=12) 
    ax12.set_xlabel('Copper [mg/Kg]', fontsize = 16) 
    ax12.set_ylabel('Susceptibility [10−8 m3/kg]', fontsize = 16) 
    ax12.grid(True) 
    ax12.set_ylim(0, yc) 
    #ax12.set_xlim(0.1, 0.8) 
    ax12.legend(loc='upper right', fontsize = 10) 

    ax5.legend(loc='upper right', fontsize = 10) 
    ax5.set_title("Susceptibility vs Iron" , fontweight='bold', fontsize=25) 
    ax5.tick_params(axis='y', labelsize=12) 
    ax5.tick_params(axis='x', labelsize=12) 
    ax5.set_xlabel('Iron [mg/Kg]', fontsize = 16) 
    ax5.set_ylabel('Susceptibility [10−8 m3/kg]', fontsize = 16) 
    ax5.grid(True) 
    ax5.set_ylim(0, yc) 
    #ax5.set_xlim(0, 25) 
    ax5.legend(loc='upper right', fontsize = 10) 

    ax6.set_title("Susceptibility vs Zinc" , fontweight='bold', fontsize=25) 
    ax6.tick_params(axis='y', labelsize=12) 
    ax6.tick_params(axis='x', labelsize=12) 
    ax6.set_ylabel('Susceptibility [10−8 m3/kg]', fontsize = 16) 
    ax6.set_xlabel('Zinc [mg/Kg]', fontsize = 16) 
    ax6.grid(True) 
    #ax6.set_xlim(0, 65) 
    ax6.set_ylim(0, yc)
    ax6.legend(loc='upper right', fontsize = 10) 
    
    ax7.set_title("Susceptibility vs pH" , fontweight='bold', fontsize=25) 
    ax7.tick_params(axis='y', labelsize=12) 
    ax7.tick_params(axis='x', labelsize=12) 
    ax7.set_xlabel('pH', fontsize = 16) 
    ax7.set_ylabel('Susceptibility [10−8 m3/kg]', fontsize = 16) 
    ax7.grid(True) 
    ax7.set_ylim(0, yc) 
    #ax7.set_xlim(0, 25) 
    ax7.legend(loc='upper right', fontsize = 10) 

    ax8.set_title("LF Susceptibility vs PLI" , fontweight='bold', fontsize=25) 
    ax8.tick_params(axis='y', labelsize=12) 
    ax8.tick_params(axis='x', labelsize=12) 
    ax8.set_ylabel('LF Susceptibility [10−8 m3/kg]', fontsize = 16) 
    ax8.set_xlabel("Pollution Load Index", fontsize = 16) 
    ax8.grid(True) 
    #ax8.set_xlim(0, 65) 
    #ax8.set_ylim(0, yc)
    ax8.legend(loc='upper right', fontsize = 10) 
    
    ax9.legend(loc='upper right', fontsize = 10) 
    ax9.set_title("Susceptibility vs Chrome" , fontweight='bold', fontsize=25) 
    ax9.tick_params(axis='y', labelsize=12) 
    ax9.tick_params(axis='x', labelsize=12) 
    ax9.set_xlabel('Chrome [mg/Kg]', fontsize = 16) 
    ax9.set_ylabel('Susceptibility [10−8 m3/kg]', fontsize = 16) 
    ax9.grid(True) 
    ax9.set_ylim(0, yc) 
    #ax9.set_xlim(0, 25) 
    ax9.legend(loc='upper right', fontsize = 10) 

    ax10.set_title("Susceptibility vs Plumb" , fontweight='bold', fontsize=25) 
    ax10.tick_params(axis='y', labelsize=12) 
    ax10.tick_params(axis='x', labelsize=12) 
    ax10.set_ylabel('Susceptibility [10−8 m3/kg]', fontsize = 16) 
    ax10.set_xlabel('Plumb [mg/Kg]', fontsize = 16) 
    ax10.grid(True) 
    #ax10.set_xlim(0, 65) 
    ax10.set_ylim(0, yc)
    ax10.legend(loc='upper right', fontsize = 10) 
    
    ax11.set_title("Susceptibility vs Nickel" , fontweight='bold', fontsize=25) 
    ax11.tick_params(axis='y', labelsize=12) 
    ax11.tick_params(axis='x', labelsize=12) 
    ax11.set_xlabel('Nickel [mg/Kg]', fontsize = 16) 
    ax11.set_ylabel('Susceptibility [10−8 m3/kg]', fontsize = 16) 
    ax11.grid(True) 
    ax11.set_ylim(0, yc) 
    #ax11.set_xlim(0, 25) 
    ax11.legend(loc='upper right', fontsize = 10) 

def sets(axc1, axc2, axc3):
    wide = 90
    axc1.legend(loc='lower right', fontsize = 10)
    axc1.set_title("Field 1", fontweight='bold', fontsize=25)
    axc1.tick_params(axis='y', labelsize=12)
    axc1.tick_params(axis='x', labelsize=12)
    axc1.set_xlabel('Magnetic susceptibility [e-5]', fontweight='bold', fontsize = 16)
    axc1.set_ylabel('Depth [cm]', fontweight='bold', fontsize = 16)
    axc1.grid(True)
    #axc1.set_ylim(0, 65) 
    axc1.set_xlim(1, wide)

    axc2.legend(loc='lower right', fontsize = 10)
    axc2.set_title("Field 2", fontweight='bold', fontsize=25)
    axc2.tick_params(axis='y', labelsize=12)
    axc2.tick_params(axis='x', labelsize=12)
    axc2.set_xlabel('Magnetic susceptibility [e-5]', fontweight='bold', fontsize = 16)
    #axc2.set_ylabel('Depth [cm]', fontsize = 16)
    axc2.grid(True)
    #axc2.set_ylim(0, 65)
    axc2.set_xlim(1, wide)

    axc3.grid(True)
    axc3.set_xlabel('Magnetic susceptibility [e-5]', fontweight='bold', fontsize = 16)
    #axc3.set_ylabel('Depth [cm]', fontsize = 16)
    axc3.legend(loc='lower right', fontsize = 10)
    axc3.set_title("Field 3", fontweight='bold', fontsize=25)
    axc3.tick_params(axis='y', labelsize=12)
    axc3.tick_params(axis='x', labelsize=12)
    #axc3.set_ylim(0, 65)
    axc3.set_xlim(1, wide)
    
def camargoplots(axp1, axp2):
    axp1.legend(loc='upper right', fontsize = 10)
    axp1.set_title("Susceptibility vs Pads" , fontweight='bold', fontsize=25) 
    axp1.tick_params(axis='y', labelsize=12) 
    axp1.tick_params(axis='x', labelsize=12) 
    axp1.set_xlabel('Phosphate adsorved [mg/Kg]', fontsize = 16) 
    axp1.set_ylabel('Susceptibility [10−6 m3/kg]', fontsize = 16) 
    axp1.grid(True) 
    #axp1.set_ylim(0, yc)  
    #axp1.set_xlim(1, 50) 
    axp1.legend(loc='upper right', fontsize = 10)

    axp2.legend(loc='upper right', fontsize = 10)
    axp2.set_title("Fe vs Pads" , fontweight='bold', fontsize=25) 
    axp2.tick_params(axis='y', labelsize=12) 
    axp2.tick_params(axis='x', labelsize=12) 
    axp2.set_xlabel('Phosphate adsorved [mg/Kg]', fontsize = 16) 
    axp2.set_ylabel('Fe [mg/Kg]', fontsize = 16) 
    axp2.grid(True) 
    #axp2.set_ylim(0, yc)  
    #axp2.set_xlim(1, 50) 
    axp2.legend(loc='upper right', fontsize = 10)
    
def pltgamma(ax1, ax2, ax3, ax4, ax5, ax6, lw, yc, plt):
    
    ax1.set_title("Sand vs Th/K" , fontweight='bold', fontsize=25) 
    ax1.set_ylabel('Th / K ratio', fontsize = 16) 
    ax1.set_xlabel('Sand [%]', fontsize = 16) 
    ax1.grid(True) 
    ax1.set_ylim(0, yc)  
    #ax1.set_xlim(1, 50) 
    ax1.legend(loc='upper right', fontsize = 10)

    ax2.set_title("Silt vs Th/K" , fontweight='bold', fontsize=25) 
    ax2.set_xlabel('Silt [%]', fontsize = 16) 
    ax2.set_ylabel('Th / K ratio', fontsize = 16) 
    ax2.legend(loc='upper left', fontsize = 10) 
    ax2.tick_params(axis='y', labelsize=12) 
    ax2.tick_params(axis='x', labelsize=12) 
    ax2.grid(True) 
    ax2.set_ylim(0, yc) 
    #ax2.set_xlim(10, 100000)
    ax2.legend(loc='upper left', fontsize = 10) 

    ax3.set_xlabel('Clay [%]', fontsize = 16) 
    ax3.set_ylabel("Th / K ratio" , fontsize = 16) 
    ax3.set_title("Clay vs Th/K" , fontweight='bold', fontsize=25) 
    ax3.grid(True) 
    ax3.legend(loc='upper left', fontsize = 10) 
    ax3.tick_params(axis='y', labelsize=12) 
    ax3.tick_params(axis='x', labelsize=12) 
    ax3.set_ylim(0, yc) 
    #ax3.set_xlim(0, 60) 
    ax3.legend(loc='upper left', fontsize = 10) 

    ax4.legend(loc='upper right', fontsize = 10) 
    ax4.set_title("CEC vs Th/K" , fontweight='bold', fontsize=25) 
    ax4.tick_params(axis='y', labelsize=12) 
    ax4.tick_params(axis='x', labelsize=12) 
    ax4.set_ylabel('Th / K ratio', fontsize = 16) 
    ax4.set_xlabel('Cation Exchange Capacity [cmol/Kg]', fontsize = 16) 
    ax4.grid(True) 
    ax4.set_ylim(0, yc) 
    #ax4.set_xlim(0.1, 0.8) 
    ax4.legend(loc='upper right', fontsize = 10) 

    ax5.legend(loc='upper right', fontsize = 10) 
    ax5.set_title("pH vs Th/K" , fontweight='bold', fontsize=25) 
    ax5.tick_params(axis='y', labelsize=12) 
    ax5.tick_params(axis='x', labelsize=12) 
    ax5.set_ylabel('Th / K ratio', fontsize = 16) 
    ax5.set_xlabel('pH', fontsize = 16) 
    ax5.grid(True) 
    ax5.set_ylim(0, yc) 
    #ax5.set_xlim(0, 25) 
    ax5.legend(loc='upper right', fontsize = 10) 

    ax6.set_title("Organic content vs Th/K" , fontweight='bold', fontsize=25) 
    ax6.tick_params(axis='y', labelsize=12) 
    ax6.tick_params(axis='x', labelsize=12) 
    ax6.set_xlabel('Organic content [%]', fontsize = 16) 
    ax6.set_ylabel('Th / K ratio', fontsize = 16) 
    ax6.grid(True) 
    #ax6.set_xlim(0, 65) 
    ax6.set_ylim(0, yc)
    ax6.legend(loc='upper right', fontsize = 10) 


# Enhanced 3D plotting function with axis labels
def plot_3d(df, x, y, z, X, Y, Z, elev=30, azim=30):
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


def fit_and_plot(df, x_cols, y_col, degree, mapping, ss=60, lw=0):
    x = df[x_cols]
    y = df[y_col]
    
    # Generate polynomial features
    poly = PolynomialFeatures(degree)
    x_poly = poly.fit_transform(x)
    
    # Fit a linear model
    model = LinearRegression()
    model.fit(x_poly, y)
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
        
        # 3D surface plot
        fig = go.Figure(data=[go.Scatter3d(x=x[x_cols[0]], y=x[x_cols[1]], z=y, mode='markers', marker=dict(size=5), name='Data points'),
                              go.Surface(x=x0_range, y=x1_range, z=y_grid, name='Polynomial Surface')])
        fig.update_layout(title=f'3D Polynomial Fit (R² = {r2:.2f})', scene=dict(xaxis_title=x_cols[0], yaxis_title=x_cols[1], zaxis_title=y_col))
        fig.show()
        
        # 2D plots for each predictor
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for i, col in enumerate(x_cols):
            x_plot = np.linspace(x[col].min(), x[col].max(), 300)
            x_plot_2d = np.column_stack([x_plot if i == j else np.full_like(x_plot, x[x_cols[j]].mean()) for j in range(2)])
            x_plot_poly = poly.transform(x_plot_2d)
            y_plot = model.predict(x_plot_poly)
            
            for start_str, (color, marker) in mapping.items():
                mask = df['SAMPLE'].str.startswith(start_str)
                filtered_df = df[mask]
                if not filtered_df.empty:  # Ensure there are points to plot and fit
                    axes[i].scatter(filtered_df[col], filtered_df[y_col], s=ss, linewidth=lw, c=color, marker=marker, label=f"{start_str} Site")


            axes[i].plot(x_plot, y_plot, color='red', label=f'Polynomial fit degree {degree}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel(y_col)
            axes[i].set_ylim(bottom=0)  # Ensure y-axis starts from 0
            axes[i].legend()
            axes[i].grid(True)

            #if x[i] == 'MS_field':
            #    axes[i].set_yscale('log')

        
        plt.show()


def res_plot(df, var1, var2, cov):# Create a figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))

    # First plot: var1 vs Fe with linear regression and residuals
    sns.regplot(x=cov, y=var1, data=df, ax=axs[0])
    axs[0].set_title(var1+' vs '+cov+' with Linear Regression')

    # Calculate residuals for var1 vs Cov
    X_var1 = df[var1].values.reshape(-1, 1)
    y_cov = df[cov].values
    reg_var1 = LinearRegression().fit(X_var1, y_cov)
    y_pred_var1 = reg_var1.predict(X_var1)
    residuals_var1 = y_cov - y_pred_var1

    # Second plot: CEC vs Fe with linear regression and residuals
    sns.regplot(x=var2, y=cov, data=df, ax=axs[1])
    axs[1].set_title(var2+' vs '+cov+' with Linear Regression')

    # Calculate residuals for var2 vs Fe
    X_var2 = df[var2].values.reshape(-1, 1)
    y_cov = df[cov].values
    reg_var2 = LinearRegression().fit(X_var2, y_cov)
    y_pred_var2 = reg_var2.predict(X_var2)
    residuals_var2 = y_cov - y_pred_var2

    # Third plot: Residuals of both linear regressions
    corr1 = np.corrcoef(residuals_var2, residuals_var1)
    axs[2].scatter(residuals_var1, residuals_var2, label='Residuals of '+var1+' vs '+cov+str(corr1), color='blue')
    axs[2].axhline(y=0, color='black', linestyle='--')
    axs[2].set_title('Residuals of Linear Regressions')
    axs[2].set_xlabel('Res '+var1)
    axs[2].set_ylabel('Res '+var2)
    axs[2].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()


def partial_correlation(df, x_col, y_col, control_cols):
    """
    Calculate the partial correlation between two columns of a dataframe controlling for two other columns.
    
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
    for i, control_col in enumerate(control_cols):
        sns.regplot(x=df[x_col], y=df[control_col], ax=axs[i])
        axs[i].set_title(f'{x_col} vs {control_col}')
    sns.scatterplot(x=x_residuals, y=y_residuals, ax=axs[-1])
    axs[-1].set_title(f'Residuals: {x_col} vs {y_col}\nPartial Corr: {partial_corr:.2f}')
    plt.tight_layout()
    plt.show()

    return partial_corr