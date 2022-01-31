from itertools import product, combinations

import os
import numpy as np
from matplotlib import pyplot as plt

from src.simulation.utils import create_grid_spherical, vec_to_rir


def plot_3d_planes(fun, slices, other_bounds, N, axis=0):
    """Plot a 3d function on the planes contained by a cartesian grid.
    Args:
        -fun (callable) : a function of 3 arguments implemented as a single argument callable
        -slices (list) : list containing the values defining each slicing plane
        -other_bounds (tuple) : a tuple or list containing the bounds for the remaining variables, e.g
    (xmin, xmax, ymin, ymax) if slicing along the z axis
        -N (int) : number of points for the linspace in the non slicing directions
        -axis (int) : axis orthogonal to the slicing plane
    """

    ax1 = np.linspace(other_bounds[0], other_bounds[1], N)
    ax2 = np.linspace(other_bounds[2], other_bounds[3], N)
    ax1, ax2 = np.meshgrid(ax1, ax2)
    ax1, ax2 = ax1.reshape(-1, 1), ax2.reshape(-1, 1)
    m, mm = np.inf, -np.inf
    if axis == 0:
        all_ax = [np.zeros([N*N, 1]), ax1, ax2]
        xlabel, ylabel, zlabel = 'y', 'z', 'x'

    elif axis == 1:
        all_ax = [ax1, np.zeros([N*N, 1]), ax2]
        xlabel, ylabel, zlabel = 'x', 'z', 'y'
    else:
        all_ax = [ax1, ax2, np.zeros([N*N, 1])]
        xlabel, ylabel, zlabel = 'x', 'y', 'z'
    all_ax = np.concatenate(all_ax, axis=1)

    N_slice = len(slices)
    for i in range(N_slice):

        all_ax[:, axis] = slices[i]
        res = np.apply_along_axis(fun, 1, all_ax)
        m = np.minimum(res, m)
        mm = np.maximum(mm, res)
        plt.imshow(res.reshape(N, N), extent=other_bounds, origin='lower')
        plt.xticks()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(zlabel + " = " + str(slices[i]))
        plt.colorbar()
        plt.show()
    print("Maximum on each slice : {} \n Minimum on each slice : {}".format(m, mm))


def plot_3d_sphere(fun, radius, dtheta, dphi):
    """
    Plot the function fun on 3d spheres indicated by the list of radiuses "radius".
    dtheta, dphi give the angular discretizations for each sphere.
    """

    for i in range(len(radius)):
        r = radius[i]
        grid, _, _ = create_grid_spherical(r, r, 1, dtheta, dphi)
        res = np.apply_along_axis(fun, 1, grid)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        sc = ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2], marker='o', c=res)
        plt.title('r = {} '.format(r))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.colorbar(sc)
        plt.show()


def plot_room(mic, src, ampl, reconstr_src, orders=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    src_ind = np.argmax(ampl)
    ax.scatter(*src[src_ind], label='source', marker='x')
    sec_src_ind = ampl < np.max(ampl)

    if orders is not None:
        order1 = (orders == 1)
        colors_true = orders[sec_src_ind]
    else:
        order1 = sec_src_ind
        colors_true = ['blue']

    wall_intersect = (src[src_ind] + src[order1]) / 2
    xmin, ymin, zmin = np.min(wall_intersect, axis=0)
    xmax, ymax, zmax = np.max(wall_intersect, axis=0)

    vertices = product([xmin, xmax], [ymin, ymax], [zmin, zmax])
    edges_plus = combinations(vertices, 2)
    for edge in edges_plus:
        x1, x2, y1, y2, z1, z2 = edge[0][0], edge[1][0], edge[0][1], edge[1][1], edge[0][2], edge[1][2]
        if len(np.unique([x1, x2])) + len(np.unique([y1, y2])) + len(np.unique([z1, z2])) == 4:
            ax.plot3D([x1, x2], [y1, y2], [z1, z2], color='k')

    ax.scatter(mic[:, 0], mic[:, 1], mic[:, 2], label='microphones', marker='+')
    ax.scatter(reconstr_src[:, 0], reconstr_src[:, 1], reconstr_src[:, 2], label='reconstructed sources',
               marker='o',  s=50, alpha=0.3, edgecolor='k', color='red')
    ax.scatter(src[sec_src_ind, 0], src[sec_src_ind, 1], src[sec_src_ind, 2], label='image sources',
               marker='D', alpha=1, s=5, c=colors_true)
    plt.legend()
    plt.show()


def save_rir(rir_vec: np.ndarray, N: int, fs: float, directory: str, name1="RIR", rir_vec2=None, name2="RIR2"):
    """Save the RIR from each microphone in a separate .eps file. If two RIR are specified, plot both for comparison
     in each file.
     Args:
         -rir_vec : the flat array containing the RIR.
         -N : number of time samples in each RIR
         -fs : sampling frequency
         -directory : path to the directory where the files are saved
         -name1 : name for the resulting file (will be postfixed by the indexes of the microphones) and the legend
         -rir_vec2 (optional) : a second rir array to plot alongside the first
     """

    M = len(rir_vec) // N
    x = np.arange(N) / fs
    for m in range(M):
        if rir_vec2 is not None:
            rir2 = vec_to_rir(rir_vec2, m, N)
            plt.plot(x, rir2, '-', label=name2)

        rir1 = vec_to_rir(rir_vec, m, N)
        plt.plot(x, rir1, '-.', label=name1)
        plt.legend()
        plt.savefig(os.path.join(directory, name1 + "_{}.pdf".format(m)))
        plt.clf()
