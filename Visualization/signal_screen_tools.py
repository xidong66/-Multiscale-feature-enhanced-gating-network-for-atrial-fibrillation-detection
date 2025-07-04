"""
Helper functions for manipulation with models and data
"""

import os
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def plot_with_gradient(y: np.array, gradient: np.array, ax=None, x=None,
                       colormap="plasma", color_bar=True,
                       normalise=False, **kwargs):
    """
    Uses template to show gradient of another variable
    :param ax: axis of the plot
    :param x: x data
    :param y: y data
    :param gradient: creates gradient on the plot
    :param colormap: viridis/plasma/...
    :param color_bar: add colour bar
    :param normalise: boolean - 0-1 normalisation
    :return:
    """

    if x is None:
        x = range(y.ravel().shape[0])

    if len(gradient.shape) > 2:
        raise Exception("Gradient is not 1D vector, but has {}".format(len(gradient.shape)))

    if len(gradient.shape) == 2:
        gradient = gradient.ravel()

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(18.5, 10.5)

    points = np.array([x, y]).T.reshape(-1, 1, 2)  # create gradient colour
    segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)

    if normalise:
        lc = LineCollection(segments, cmap=colormap, linewidth=4, norm=Normalize(vmin=0, vmax=1))
    else:
        lc = LineCollection(segments, cmap=colormap, linewidth=4)

    lc.set_array(gradient)  # set up gradient
    line = ax.add_collection(lc)
    ax.autoscale()

    if color_bar:
        plt.colorbar(line, ax=ax)

    if "title" in kwargs:
        ax.set_title(kwargs["title"])

    if fig is not None:
        return fig, ax
    return ax


def add_zeros(array: np.ndarray, index: int, n_zeros: int) -> np.ndarray:
    """
    Adds zeros centered around the required index. So if signal is 1 2 3 4 5 and central index is 2 with number of
    zeros to put is 3 -> result is 1 0 0 0 5

    :param array: array where to put zeros
    :param index: where is central index
    :param n_zeros: number of zeros around it.
    :return: array with added zeros
    """

    half = int((n_zeros - 1) / 2)

    if len(array.shape) > 1:
        if index <= half:
            array[:, :half + index + 1, ...] = 0
        elif array.shape[1] <= index + half:
            array[:, index - half:, ...] = 0
        else:
            array[:, index - half: index + half + 1, ...] = 0
    else:
        if index <= half:
            array[:half + index + 1] = 0
        elif array.shape[1] <= index + half:
            array[index - half:] = 0
        else:
            array[index - half: index + half + 1] = 0

    return array


def normalise_array(array: np.ndarray) -> np.ndarray:
    """
    normalise array to range 0 - 1
    :param array: numpy array
    :return: numpy array with range 0 - 1
    """
    arr_min, arr_max = np.min(array), np.max(array)
    return (array - arr_min) / (arr_max - arr_min + tf.keras.backend.epsilon())


def clone_model(model: Model) -> Model:
    """
    Creates clone of the Keras model
    :param model: model to use in library
    :return: cloned keras modelModel
    """
    model_clone = tf.keras.models.clone_model(model)
    model_clone.set_weights(model.get_weights())
    model_clone.compile()
    return model_clone


def change_last_activation_to_linear(model: Model, dependencies=None) -> Model:
    """
    Changes last activation function to linear, if it already is linear - returns same model
    :param dependencies: necessary objects for custom functions in keras
    :param model: Keras model
    :return: Keras model with linear activation function at the end
    """

    cloned_model = clone_model(model)  # create copy of model to modify

    if model.layers[-1].activation == tf.keras.activations.linear:
        return cloned_model

    config = cloned_model.layers[-1].get_config()  # store config of last layer
    weights = cloned_model.layers[-1].get_weights()
    config['activation'] = tf.keras.activations.linear
    config['name'] = 'predictions'

    # add new layer
    new_layer = tf.keras.layers.Dense(**config)(cloned_model.layers[-2].output)
    new_model = tf.keras.Model(inputs=cloned_model.inputs, outputs=[new_layer])
    new_model.layers[-1].set_weights(weights)

    assert new_model.layers[-1].activation == tf.keras.activations.linear

    # store and load new model to prevent some ambiguity
    try:
        new_model.save("temp_model.h5")
        new_model = tf.keras.models.load_model("temp_model.h5", dependencies, compile=False)
        return new_model

    finally:
        if os.path.exists("temp_model.h5"):
            os.remove("temp_model.h5")
