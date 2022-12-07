""" Helper functions for all the visualization tasks. """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .io import bprint
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection


def plot_violin(data_analysis: dict, save_folder: str, mode: str, config_name: str, 
                figsize: tuple=(15, 10), dpi: int=200, v: bool=False) -> None:
    """
        Plot and save the violin plot of the error distribution.

        Input:
            data_analysis: dict, the data analysis dictionary.
            save_folder: str, the save folder.
            mode: str ('none', 'shift', 'homography'), the mode.
            config_name: str, the config name, used for saving the figure.
            figsize: tuple, the figure size.
            dpi: int, the figure dpi.
            v: bool, verbose.
    """
    data_to_plot = [data_analysis['global']['arr'], data_analysis['best']['arr'], data_analysis['worst']['arr']]
    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    ax.violinplot(data_to_plot, showmeans=True)
    plt.ylim([0, max(69, max(data_analysis['global']['arr'].max(),
                             data_analysis['best']['arr'].max(),
                             data_analysis['worst']['arr'].max()))+1])
    ax.set_ylabel("Reprojection Error (pixels)")
    ax.set_xticks([1, 2, 3], ["Global", "Best", "Worst"])
    ax.set_title("Reprojection Error Distribution")
    plt.tight_layout()
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k')
    plt.grid(b=True, which='minor', color='k', alpha=0.1)

    save_path = f'{save_folder}violin_{config_name.split(".")[0]}_{mode}.png'
    plt.savefig(save_path, dpi=dpi)
    if v:
        bprint(f"Saved violin plot to {save_path}")


def plot_polar(dict_data: dict, dict_angle: dict, vargeo: int, polar_plot_info: list, save_folder: str, 
               mode: str, config_name:str, figsize: tuple=(15, 10), dpi: int=200, v: bool=False) -> None:
    """
        Plot and save the polar plot of the error distribution (only possible for captures with ranges).
        ref: https://towardsdatascience.com/polar-heatmaps-in-python-with-matplotlib-d2a09610bc55

        Input:
            dict_data: dict, dictionary containing the error data for the entire capture.
            dict_angle: dict, dictionary containing the angle data for the entire capture.
            vargeo: int, the vargeo for the plot.
            polar_plot_info: list [#thetas, #phis], list containing the information for the polar plot.
            save_folder: str, the save folder.
            mode: str ('none', 'shift', 'homography'), the mode.
            config_name: str, the config name, used for saving the figure.
            figsize: tuple, the figure size.
            dpi: int, the figure dpi.
            v: bool, verbose.
    """
    # Assembling the dataframe with required values
    thetas = []
    phis = []
    plot_info = []
    for key, losses in dict_data.items():
        angles = dict_angle[key]
        theta_i, phi_i, theta_o, phi_o, varg = angles
        if varg != vargeo:
            continue
        thetas.append(np.sin(np.deg2rad(theta_o)))
        phis.append(phi_o)
        plot_info.append(losses.mean())
    df = pd.DataFrame(list(zip(thetas, phis, plot_info)),columns =['theta','phi','errs'])
    
    # Filling up the patches
    thetas = np.array(thetas)
    phis = np.array(phis)
    plot_info = np.array(plot_info)

    n_phi = polar_plot_info[1]
    dtheta = 360 / n_phi
    plot_phis = np.unique(phis)
    # Considering radial distortion
    n_theta = polar_plot_info[0]
    rad_diffs = np.zeros(n_theta+1)
    rad_diffs[1:] = np.unique(thetas)

    cm = plt.cm.get_cmap('viridis', 1000)
    cm.set_bad(color='white')

    patches = []; avg_errs = []
    for nr in range(n_theta,0,-1):
        start_r = rad_diffs[nr-1]
        end_r = rad_diffs[nr]
        for nt in range(0, n_phi):
            start_t = plot_phis[nt%n_phi]
            end_t = plot_phis[(nt+1)%n_phi]
            if end_t < start_t:
                end_t += 360
            stripped = df[(df['theta']>start_r) & (df['theta']<=end_r) &          
                (df['phi']>=start_t) & (df['phi']<end_t)]
        
            avg_errs.append(stripped['errs'].mean())
            wedge = mpatches.Wedge(0,end_r, start_t, end_t)
            patches.append(wedge)

    # Normalizing the error values of the colorbar
    avg_errs = np.array(avg_errs)
    avg_errs = (avg_errs - min(avg_errs)) / (max(avg_errs) - min(avg_errs))

    collection = PatchCollection(patches,linewidth=0.5,
        edgecolor=['#111' for x in avg_errs], 
        facecolor=cm([x for x in avg_errs]),
        cmap=cm)
    
    # Saving the figure
    fig = plt.figure(figsize=figsize, dpi=dpi, edgecolor='w',facecolor='w')
    ax = fig.add_subplot()
    mesh = ax.add_collection(collection)
    ax.autoscale_view()
    # Clean up the image canvas and save the figure
    plt.axis('equal')
    plt.axis('off')
    mesh.set_clim(0, max(plot_info))
    plt.title(f'Vargeo: {vargeo}')
    plt.colorbar(collection, location='bottom', pad=0, shrink=0.5)

    save_path = f'{save_folder}polar_{config_name.split(".")[0]}_{mode}_var{vargeo}.png'
    plt.savefig(save_path, dpi=dpi)
    if v:
        bprint(f"Saved error plot to {save_path}")