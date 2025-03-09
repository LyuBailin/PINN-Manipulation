import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.animation import FuncAnimation

FIGURES_PATH = os.path.join('../figures')
FIGURE_WIDTH_PT = 410


def fig_size(width_pt=FIGURE_WIDTH_PT, fraction=1, ratio=(5 ** .5 - 1) / 2, subplots=(1, 1)):
    """
    Returns the width and heights in inches for a matplotlib figure.

    :param float width_pt: document width in points, in latex can be determined with \showthe\linewidth
    :param float fraction: fraction of the width with which the figure will occupy
    :param float ratio: ratio of the figure, default is the golden ratio
    :param tuple subplots: the shape of subplots
    :return: float fig_width_in: width in inches of the figure, float fig_height_in: height in inches of the figure
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = ratio

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in


def new_fig(width_pt=410, fraction=1, ratio=(5 ** .5 - 1) / 2, subplots=(1, 1)):
    """
    Creates new instance of a `matplotlib.pyplot.figure` fig by using the `fig_size` function.

    :param float width_pt: document width in points, in latex can be determined with \showthe\textwidth
    :param float fraction: fraction of the width with which the figure will occupy
    :param float ratio: ratio of the figure, default is the golden ratio
    :param tuple subplots: the shape of subplots
    :return: matplotlib.pyplot.figure fig: instance of a `matplotlib.pyplot.figure` with desired width and height
    """
    fig = plt.figure(figsize=fig_size(width_pt, fraction, ratio, subplots))
    return fig


def save_fig(fig, name, path=None, tight_layout=True):
    """
    Saves a `matplotlib.pyplot.figure` as pdf file.

    :param matplotlib.pyplot.figure fig: instance of a `matplotlib.pyplot.figure` to save
    :param str name: filename without extension
    :param str path: path where the figure is saved, if None the figure is saved at the results directory
    :param bool crop: bool if the figure is cropped before saving
    """
    if tight_layout:
        fig.tight_layout()

    if path is None:
        path = FIGURES_PATH

    if not os.path.exists(path):
        os.makedirs(path)

    fig.savefig(os.path.join(path, f'{name}.pdf'), transparent=True)


def animate(Y_test, Y_preds, labels, fraction=1, fps=100, save_ani=False):
    """
    Creates a animation of robot along the time.

    :param np.ndarray Y_test: reference solution
    :param list Y_preds: list of predicted solutions
    :param list labels: labels of the predicted solutions
    :param fraction: fraction of the width with which the figure will occupy
    :param str file_basename: basename of the file, where the animation is saved
    """

    def q_2_polygons(q, link_width=0.08):
        """
        Calculates data of polygons for the fancy visualization of the robot.

        :param np.ndarray q: vector of generalized coordinates
        :param float link_width: width of the links
        :return: np.ndarray x_data: vector of coordinates in x direction for the polygons,
        np.ndarray x_data: vector of coordinates in y direction for the polygons,
        """
        l0 = 0.2236
        l1 = 0.18
        l2 = 0.5586

        alpha0 = 0
        alpha = q[0]
        beta = q[1]
        angels = [alpha0, alpha + alpha0, beta + alpha + alpha0, beta + alpha + alpha0]

        x_data = [0, 0, -np.sin(alpha) * l1, -np.sin(alpha) * l1 - np.sin(alpha + beta) * (l2 - link_width),
                  -np.sin(alpha) * l1 - np.sin(alpha + beta) * l2]
        y_data = [0, l0, l0 + np.cos(alpha) * l1, l0 + np.cos(alpha) * l1 + np.cos(alpha + beta) * (l2 - link_width),
                  l0 + np.cos(alpha) * l1 + np.cos(alpha + beta) * l2]

        polygons = []

        for i, angle in enumerate(angels):
            ln = np.array([[x_data[i], y_data[i]], [x_data[i + 1], y_data[i + 1]]])
            if i == 0:
                ln = np.array([
                    ln[0],
                    ln[1] + np.array([- np.sin(angle) * link_width / 2, np.cos(angle) * link_width / 2]),
                ])
            elif i == 2:
                ln = np.array([
                    ln[0] - np.array([- np.sin(angle) * link_width / 2, np.cos(angle) * link_width / 2]),
                    ln[1],
                ])
            elif i == 3:
                ln = np.array([
                    ln[0],
                    ln[1],
                ])
            else:
                ln = np.array([
                    ln[0] - np.array([- np.sin(angle) * link_width / 2, np.cos(angle) * link_width / 2]),
                    ln[1] + np.array([- np.sin(angle) * link_width / 2, np.cos(angle) * link_width / 2]),
                ])

            if i == 3:
                link = np.array([
                    ln[0] + np.array([np.cos(angle) * link_width / 4, np.sin(angle) * link_width / 4]),
                    ln[0] - np.array([np.cos(angle) * link_width / 4, np.sin(angle) * link_width / 4]),
                    ln[1] - np.array([np.cos(angle) * link_width / 4, np.sin(angle) * link_width / 4]),
                    ln[1] + np.array([np.cos(angle) * link_width / 4, np.sin(angle) * link_width / 4]),
                ])
            else:
                link = np.array([
                    ln[0] + np.array([np.cos(angle) * link_width / 2, np.sin(angle) * link_width / 2]),
                    ln[0] - np.array([np.cos(angle) * link_width / 2, np.sin(angle) * link_width / 2]),
                    ln[1] - np.array([np.cos(angle) * link_width / 2, np.sin(angle) * link_width / 2]),
                    ln[1] + np.array([np.cos(angle) * link_width / 2, np.sin(angle) * link_width / 2]),
                ])
            polygons.append(link)

        return polygons

    fig = new_fig(fraction=fraction)
    ax = fig.add_subplot(111)
    ax.set_xlabel('$x$ in m')
    ax.set_ylabel('$y$ in m')
    title = ax.set_title('Time $t=0$ s')
    ax.set(xlim=[-1.25, 1.25], ylim=[-0.25, 1.25])

    polygons_coords = q_2_polygons(np.array([-0.5, 0.5]))

    polygon_list_test = []
    fc_list = ['grey', 'darkslategrey', 'lightgrey', 'lightgrey']
    plot_order = [1, 0, 2, 3]
    for i in plot_order:
        if i == 0:
            polygon = patches.Polygon(polygons_coords[i], closed=True, fc=fc_list[i], ec='k', label='Ref')
        else:
            polygon = patches.Polygon(polygons_coords[i], closed=True, fc=fc_list[i], ec='k')
        ax.add_patch(polygon)

        polygon_list_test.append(polygon)

    ploygon_lists_preds = []
    fc_list = [(0, 0, 1, 0.3), (1.0, 0.5, 0.25, 0.3)]
    ec_list = [(0, 0, 1, 1), (1.0, 0.5, 0.25, 1)]
    for i in range(len(Y_preds)):
        polygon_list = []
        for j in plot_order:
            if j == 0:
                polygon = patches.Polygon(
                    polygons_coords[j], closed=True, fc=fc_list[i], ec=ec_list[i], label=labels[i])
            else:
                polygon = patches.Polygon(
                    polygons_coords[j], closed=True, fc=fc_list[i], ec=ec_list[i])
            ax.add_patch(polygon)
            polygon_list.append(polygon)
        ploygon_lists_preds.append(polygon_list)

    stand_polygon = patches.Polygon(np.array([[0.25, -0.025], [-0.25, -0.025], [-0.25, 0.0], [0.25, 0.0]]),
                                    closed=True, fc='grey', ec='k')
    ax.add_patch(stand_polygon)

    ax.legend(loc='upper right')

    def update_plot(frame, title):
        title.set_text(f'Time $t={frame / fps:.2f}$ s')
        polygons_coords = q_2_polygons(Y_test[frame])
        for i, j in enumerate(plot_order):
            polygon_list_test[i].set_xy(polygons_coords[j])

        for index, Y_pred in enumerate(Y_preds):
            polygons_coords = q_2_polygons(Y_pred[frame])
            for i, j in enumerate(plot_order):
                ploygon_lists_preds[index][i].set_xy(polygons_coords[j])

    ani = FuncAnimation(fig, update_plot, frames=len(Y_test), fargs=(title,),
                        interval=1000 / fps)
    if save_ani:
        if not os.path.exists(FIGURES_PATH):
            os.makedirs(FIGURES_PATH)
        # ani.save(os.path.join(FIGURES_PATH, 'ani.mp4'), writer='ffmpeg', fps=fps)
        ani.save(os.path.join(FIGURES_PATH, 'ani.gif'), writer='imagemagick', fps=fps)

    fig.tight_layout()
    plt.show()


def plot_input_sequence(T, U, filename=None):
    M = U.shape[1]
    linewidth = 2
    fig = new_fig()
    ax = fig.add_subplot(111)
    ax.set(xlim=[np.min(T), np.max(T)])
    ax.set(xlabel='Time $t$ (s)', ylabel=fr'Input $u$ (A)')
    colors = ['tab:blue', 'tab:red']
    for i in range(M):
        ax.step(T, U[:, i], where='post', linewidth=linewidth, label=fr'$u_{i + 1}$', c=colors[i])
    ax.legend(loc='best')
    ax.grid('on')
    fig.tight_layout()
    if filename is not None:
        save_fig(fig, filename)
    plt.show()


def plot_states(T, Z_ref, Z_pred=None, Z_mpc=None, filename=None):
    linewidth = 2
    fig = new_fig()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel='Time $t$ (s)', ylabel=r'Angle $\alpha/\beta$ (rad)')
    ax.set(xlim=[np.min(T), np.max(T)])
    colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']
    for i, angle in enumerate(['alpha', 'beta']):
        ax.plot(T, Z_ref[:, i], linewidth=linewidth, label=f'$\{angle}' + r'^{\mathrm{ref}}$', c=colors[i * 2])
        if Z_pred is not None:
            ax.plot(T, Z_pred[:, i], linestyle='--', linewidth=linewidth, label=r'$\widehat{' + f'\{angle}' + '}$',
                    c=colors[i * 2 + 1])
        if Z_mpc is not None:
            ax.plot(T, Z_mpc[:, i], linestyle='--', linewidth=linewidth, label=f'$\{angle}' + r'^{\mathrm{MPC}}$',
                    c=colors[i * 2 + 1])

    ax.grid('on')
    ax.legend(loc='best')
    fig.tight_layout()
    if filename is not None:
        save_fig(fig, filename)
    plt.show()


def plot_absolute_error(T, Z_ref, Z_pred=None, Z_mpc=None, filename=None):
    states = [r'\alpha', r'\beta']
    colors = ['tab:blue', 'tab:red']
    fig = new_fig()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel='Time $t$ (s)', ylabel=r'Absolute Error (rad)')
    ax.set_yscale('log')
    ax.grid(True, which='both')
    ax.set(xlim=[np.min(T), np.max(T)])

    for i, state in enumerate(states):
        abs_errors_list = []
        if Z_pred is not None:
            abs_errors = np.abs(Z_ref[:, i] - Z_pred[:, i])
            label = r'$|' + state + r'^{\mathrm{ref}} - ' + r'\widehat{' + state + r'}|$'
        else:
            abs_errors = np.abs(Z_ref[:, i] - Z_mpc[:, i])
            label = r'$|' + state + r'^{\mathrm{ref}} - ' + state + r'^{\mathrm{MPC}}|$'

        abs_errors_list.append(abs_errors)

        mae = abs_errors.mean()
        maxe = abs_errors.max()
        ax.plot(T, abs_errors, linewidth=2,
                label=label, c=colors[i])
        logging.info(label + f', MAE: {mae:.2e}, MaxE: {maxe:.2e}')

    ax.legend(loc='best')
    if filename is not None:
        save_fig(fig, filename)
    fig.tight_layout()
    plt.show()


def visualize_loaded_data(lb, ub, X_test, Y_test, X_star, Y_star, input_dim, output_dim):
    """
    Visualize the loaded data from data.npz
    
    Args:
        lb: Lower bounds
        ub: Upper bounds
        X_test: Test input data
        Y_test: Test output data
        X_star: Validation input data
        Y_star: Validation output data
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Bounds
    ax1 = fig.add_subplot(321)
    bounds_index = range(len(lb))
    ax1.bar(bounds_index, ub, alpha=0.3, label='Upper bound')
    ax1.bar(bounds_index, lb, alpha=0.3, label='Lower bound')
    ax1.set_title('Input Bounds')
    ax1.set_xlabel('Input dimension')
    ax1.set_ylabel('Value')
    ax1.legend()
    
    # Plot 2: Test Output
    ax2 = fig.add_subplot(322)
    for i in range(Y_test.shape[1]):
        ax2.plot(Y_test[:, i], label=f'Output {i+1}')
    ax2.set_title('Test Output Sequence')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Value')
    ax2.legend()
    
    # Plot 3: Validation Output
    ax3 = fig.add_subplot(323)
    for i in range(Y_star.shape[1]):
        ax3.plot(Y_star[:400, i], label=f'Output {i+1}')
    ax3.set_title('Validation Output Sequence')
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Value')
    ax3.legend()
    
    # Plot 4: Test Input Data
    ax4 = fig.add_subplot(324)
    for i in range(X_test.shape[1]):
        ax4.plot(X_test[:, i], label=f'Input {i+1}')
    ax4.set_title('Test Input Sequence')
    ax4.set_xlabel('Time step')
    ax4.set_ylabel('Value')
    ax4.legend()
    
    # Plot 5: Validation Input Data
    ax5 = fig.add_subplot(325)
    for i in range(X_star.shape[1]):
        ax5.plot(X_star[:400, i], label=f'Input {i+1}')
    ax5.set_title('Validation Input Sequence')
    ax5.set_xlabel('Time step')
    ax5.set_ylabel('Value')
    ax5.legend()
    
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()

    # Create a new figure for the test data distribution
    fig_dist = plt.figure(figsize=(20, 15))
    num_features = X_test.shape[1]
    
    for i in range(num_features):
        ax = fig_dist.add_subplot((num_features + 1) // 2, 2, i + 1)
        ax.hist(X_test[:, i], bins=300, alpha=0.3, label=f'Input {i+1}')
        ax.set_title(f'Test Input Distribution for Feature {i+1}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()
    
    # Print data dimensions
    print("\nData Dimensions:")
    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_test shape: {Y_test.shape}")
    print(f"X_star shape: {X_star.shape}")
    print(f"Y_star shape: {Y_star.shape}")