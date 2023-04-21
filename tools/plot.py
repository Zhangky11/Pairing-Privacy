import os

# import ML libs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# import internal libs
from config import MAX_NUM_LINE
from logger import logger
import seaborn as sns
from scipy.ndimage import gaussian_filter


def plot_prob(args, vals, labels, name, xlbl, ylbl):
    """plot hist for the array.

    Args:
        args (dict): set containing all program arguments
        val (np.array): (length, ) contain numbers. 
        labels (list): contain the label for each number.
        name (str): the name of the figure
        xlbl (str)
        ylbl (str)
    """
    fig, ax = plt.subplots()
    ax.bar(labels, vals)
    ax.set(xlabel = xlbl, ylabel = ylbl, title = name)
    # define the path & save the fig
    path = os.path.join(args.save_path, "{}.png".format(name))
    fig.savefig(path, dpi = 500)
    plt.close()


def plot_hist(args, arr, key, labels = None, if_density = True):
    """plot hist for the array.

    Args:
        args (dict): set containing all program arguments
        arr (np.array): (length, n) contain numbers. 
        labels (list): the list of label whose size corresponds to n.
        key (str): what the values stands for
        if_density (boolen): if use density.
    """
    fig, ax = plt.subplots()
    if labels == None:
        if if_density:
            ax.hist(arr, histtype='bar', density=True)
        else:
            ax.hist(arr, histtype='bar', density=False)
    else:
        if if_density:
            ax.hist(arr, histtype='bar', label=labels, density=True)
        else:
            ax.hist(arr, histtype='bar', label=labels, density=False)
        ax.legend(prop={'size': 10})
    ax.set(xlabel = key, title = '{}\'s distribution'.format(key))
    # define the path & save the fig
    path = os.path.join(args.save_path, "{}-hist.png".format(key))
    fig.savefig(path)
    plt.close()


def plot_curves(args, save_folder, res_dict):
    """plot curves for each key in dictionary.

    Args:
        args (dict): set containing all program arguments
        res_dict (dict): dictionary containing pairs of key and val(list)
    """
    for key in res_dict.keys():   
        # define the path
        path = os.path.join(save_folder, "{}-curve.png".format(key))
        # plot the fig
        if "eigen" in key:
            fig, ax = plt.subplots()
            ax.plot(np.arange(0, len(res_dict[key]) * args.hessian_interval, args.hessian_interval), res_dict[key])
            ax.set(xlabel='epoch', ylabel=key,
                   title='{}\'s curve'.format(key))
            ax.grid()
            fig.savefig(path)
        else:
            fig, ax = plt.subplots()
            ax.plot(np.arange(len(res_dict[key])) + 1, res_dict[key])
            ax.set(xlabel = 'epoch', ylabel = key,
                title = '{}\'s curve'.format(key))
            ax.grid()
            fig.savefig(path)
        plt.close()


def plot_curves_numpy(args, data, name):
    """plot curves for each key in dictionary.

    Args:
        args (dict): set containing all program arguments
        res_dict (dict): dictionary containing pairs of key and val(list)
    """
    # define the path
    path = os.path.join(args.save_path, "{}-curve.png".format(name))
    # plot the fig
    fig, ax = plt.subplots()
    ax.plot(np.arange(data.shape[0]) + 1, data)
    ax.set(xlabel='epoch', ylabel=name,
           title='{}\'s curve'.format(name))
    ax.grid()
    fig.savefig(path)
    plt.close()


def plot_multiple_curves(args, res_dict, name, bold_keys):
    """plot curves in one figure for each key in dictionary.

    Args:
        args (dict): set containing all program arguments
        res_dict (dict): dictionary containing pairs of key and val(list)
        name (str): the name of the plot.
        bold_keys (list): the list of keys whose corresponding lines should be bold
    """
    if len(res_dict.keys()) > MAX_NUM_LINE:
        logger.error("the number of lines exceed the capacity of one graph. Terminate the program with code -1.")
        exit(-1)
    
    # Initialise the figure and axes.
    fig, ax = plt.subplots(figsize=(10, 5))
    # Draw all the lines in the same plot, assigning a label for each one to be
    # shown in the legend.
    for key in res_dict.keys():
        if key in bold_keys:
            ax.plot(np.arange(len(res_dict[key])), np.asarray(res_dict[key]), label=key, linewidth=4.)
        else:
            ax.plot(np.arange(len(res_dict[key])), np.asarray(res_dict[key]), label=key)
    ax.grid()
    ax.set(xlabel = 'epoch', title = name)
    # Add a legend, and position it on the lower right (with no box)
    plt.legend(frameon=False, prop={'size': 6.5}, bbox_to_anchor=(1.0001, 1), loc='upper left')
    # save the fig
    path = os.path.join(args.save_path, "{}.png".format(name))
    fig.savefig(path)
    plt.close()


def plot_heatmap(args, mat, name, col = None, if_balance = True):
    """plot heatmap
    Args:
        args (dict): set containing all program arguments
        mat: the mat used to plot heatmap numpy 2darray
        name (str): the name of the plot.
        col (str): name of color
        if_balance (boolen): if balance the color bar
    """
    mat = np.round(mat, 5)
    fig, ax = plt.subplots(figsize=(15, 15))
    if if_balance:
        max_magn = np.max(np.abs(mat))
        if col == "coolwarm":
            ax = sns.heatmap(mat, annot=True, cmap = col, vmin= - max_magn, vmax= max_magn)
        else:
            ax = sns.heatmap(mat, annot=True, vmin= - max_magn, vmax= max_magn)
    else:
        if col == "coolwarm":
            ax = sns.heatmap(mat, annot=True, cmap = col)
        else:
            ax = sns.heatmap(mat, annot=True)
    ax.set_title(name)
    fig.tight_layout()
    path = os.path.join(args.save_path, "{}.png".format(name))
    fig.savefig(path) 
    plt.close()
    

def plot_heatmap_with_feature(args, mat, name, feature, col = None, if_balance = True, contexts = None):
    """plot heatmap plot rectangle on the feature.
    Args:
        args (dict): set containing all program arguments
        mat: the mat used to plot heatmap numpy 2darray
        name (str): the name of the plot.
        feature (scalar): the point which should be pointed out on figure.
        col (str): name of color
        if_balance (boolen): if balance the color bar
        contexts (list): a list of points which should be highlighted.
    """
    image_size = mat.shape[-1]
    mat = np.round(mat, 5)
    fig, ax = plt.subplots(figsize=(15, 15))
    if if_balance:
        max_magn = np.max(np.abs(mat))
        if col == "coolwarm":
            ax = sns.heatmap(mat, annot=True, cmap = col, vmin= - max_magn, vmax= max_magn)
        else:
            ax = sns.heatmap(mat, annot=True, vmin= - max_magn, vmax= max_magn)
    else:
        if col == "coolwarm":
            ax = sns.heatmap(mat, annot=True, cmap = col)
        else:
            ax = sns.heatmap(mat, annot=True)
    ax.add_patch(patches.Rectangle(( feature % image_size, feature // image_size), 30 / image_size, 30 / image_size, fill=False, edgecolor='green', linewidth=4))
    if contexts is not None:
        for i in contexts:
            ax.add_patch(patches.Rectangle((i % image_size, i // image_size), 30 / image_size, 30 / image_size, fill=False, edgecolor='red', linewidth=4))
    ax.set_title(name)
    fig.tight_layout()
    path = os.path.join(args.save_path, "{}.png".format(name))
    fig.savefig(path) 
    plt.close()


def plot_heatmap_with_gaussian(args, mat, name, feature, sigma = 1, col = None, if_balance = True):
    """plot heatmap
    Args:
        args (dict): set containing all program arguments
        mat: the mat used to plot heatmap numpy 2darray
        name (str): the name of the plot.
        feature (scalar): the point which should be pointed out on figure.
        sigma (int): the parameter of gaussian blur.
        col (str): name of color
        if_balance (boolen): if balance the color bar
    """
    image_size = mat.shape[-1]
    mat = gaussian_filter(mat, sigma)
    fig, ax = plt.subplots(figsize=(15, 15))
    if if_balance:
        max_magn = np.max(np.abs(mat))
        if col == "coolwarm":
            ax = sns.heatmap(mat, annot=False, cmap = col, vmin= - max_magn, vmax= max_magn)
        else:
            ax = sns.heatmap(mat, annot=False, vmin= - max_magn, vmax= max_magn)
    else:
        if col == "coolwarm":
            ax = sns.heatmap(mat, annot=False, cmap = col)
        else:
            ax = sns.heatmap(mat, annot=False)
    ax.add_patch(patches.Rectangle((feature % image_size, feature // image_size), 30 / image_size, 30 / image_size, fill=False, edgecolor='green', linewidth=4))
    ax.set_title(name)
    fig.tight_layout()
    path = os.path.join(args.save_path, "{}.png".format(name))
    fig.savefig(path) 
    plt.close()


def plot_image(args, img, name):
    """plot RGB image
    Args:
        args (dict): set containing all program arguments
        img: the mat used to plot heatmap numpy 3darray (3, image_size, image_size) or (image_size, image_size)
        name (str): the name of the plot.
    """
    image_size = img.shape[-1]
    if len(img.shape) == 3:
        img = img.transpose((1,2,0))
    # plot figure
    fig, ax = plt.subplots(figsize=(round(image_size/3), round(image_size/3)))
    imgplot = plt.imshow(img)
    ax.set_title(name)
    # save the figure
    path = os.path.join(args.save_path, "{}.png".format(name))
    fig.savefig(path)
    plt.close()