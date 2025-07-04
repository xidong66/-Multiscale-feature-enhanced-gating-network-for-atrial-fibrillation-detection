"""
Functions to calculate different visualisations for 1D models
"""

from tensorflow.keras.models import Model
from signal_screen_tools import add_zeros, change_last_activation_to_linear, normalise_array

import numpy as np
import tensorflow as tf


def calculate_occlusion_sensitivity(model: Model, data: np.ndarray, c: int, number_of_zeros=None):
    """
    https://arxiv.org/abs/1311.2901
    Places zeros through the all measurements and process them through model. Changes are relative to reference created
    by unchanged data. New point is calculated by subtraction of newly generated data from reference.
    :param model: Keras Sequential model
    :param data: array of signals
    :param c: index of class
    :param number_of_zeros: number of zeros placed in the signal - list is iterated e.g. [5, 9, 15]
    :return: data, list of all differences based on number of zeros placed, [number of used zeros]
    """

    if number_of_zeros is None:
        number_of_zeros = [15]  # [3, 5, 9] no range is defined

    # creation of reference - to which the difference is calculated
    reference = model.predict(data)[..., c]

    sensitivities = []

    # putting zeros into signals
    for zeros in number_of_zeros:  # calculated for picked ranges

        sensitivity_temp = np.empty([data.shape[0], data.shape[1]])  # temporal memory for actual sensitivity

        # create iteration object with tqdm library
        try:
            from tqdm import tqdm
            to_iterate = tqdm(range(data.shape[1]),
                              desc="Occlusion sensitivity for {} samples and class {}".format(zeros, c))
        except ModuleNotFoundError:
            to_iterate = range(data.shape[1])

        # iterate through signal
        for index in to_iterate:
            prediction = model.predict(add_zeros(data.copy(), index, zeros))[..., c]  # get prediction
            sensitivity_temp[:, index] = reference - prediction

        # save all configurations of zeros
        sensitivities.append(sensitivity_temp)  # stacking for all zero frames

    return sensitivities, number_of_zeros


def calculate_grad_cam(model: Model, data: np.ndarray, c: int,
                       name_of_conv_layer: str = None, index_of_conv_layer: int = None,
                       dependencies=None, use_relu=True, normalise=True, upscale_to_input=True) -> np.ndarray:
    """
    https://arxiv.org/abs/1610.02391 - gradCAM
    Calculates gradCAM for feedforward neural network.
    Modified code from:
    :param c: index of class
    :param dependencies: custom dependencies for neural network
    :param model: model for gradCAM
    :param data: data to visualise
    :param index_of_conv_layer: index of conv layer
    :param name_of_conv_layer: string of conv layer
    :param use_relu: raw gradients can be more interpretative then only positive gradients
    :param upscale_to_input: result will be the same size as
    :param normalise: to put values between 0-1
    :return: np.ndarray with saliency map
    """

    model_modified: Model = change_last_activation_to_linear(model, dependencies)

    # find last layer of CNN if not defined
    if name_of_conv_layer is None and index_of_conv_layer is None:
        for i, layer in enumerate(model_modified.layers):
            if isinstance(layer, tf.keras.layers.Conv1D):
                index_of_conv_layer = i

    if index_of_conv_layer:
        layer = model_modified.get_layer(index=index_of_conv_layer)
        if not isinstance(layer, tf.keras.layers.Conv1D):
            raise Exception("Convolution layer is not correctly defined")
        model_with_conv_output = Model([model_modified.input],
                                       [model_modified.output,
                                        model_modified.get_layer(index=index_of_conv_layer).output])
    elif name_of_conv_layer:
        layer = model_modified.get_layer(name=name_of_conv_layer)
        if not isinstance(layer, tf.keras.layers.Conv1D):
            raise Exception("Convolution layer is not correctly defined")
        model_with_conv_output = Model([model_modified.input],
                                       [model_modified.output,
                                        model_modified.get_layer(name=name_of_conv_layer).output])
    else:
        raise Exception("Convolution layer is missing / is not correctly defined")

    # calculate gradient
    with tf.GradientTape() as g:
        data = tf.convert_to_tensor(data)
        y_A = model_with_conv_output(data)
        dy_dA = g.gradient(y_A[0][:, c], y_A[1])

    # weights for the masks
    weights = tf.reduce_mean(dy_dA, axis=(0, 1))
    
    grad_CAM = tf.reduce_sum(tf.multiply(weights, y_A[1]), axis=-1)  # multiplication of the masks
    # reducing to one projection
    # _grad_CAM = tf.reduce_mean(tf.reduce_sum(tf.multiply(weights, y_A[1]), axis=-1), axis=0)

    # use relu to get only positive gradients
    if use_relu:
        grad_CAM = tf.nn.relu(grad_CAM)

    if upscale_to_input:
        from scipy.ndimage.interpolation import zoom
        if len(grad_CAM.shape) == 1:
            scale_factor = data.shape[1] / grad_CAM.shape[0]
        else:
            scale_factor = data.shape[1] / grad_CAM.shape[1]  # expand to shape of input
        grad_CAM = zoom(grad_CAM, scale_factor)

    if normalise:
        grad_CAM = normalise_array(grad_CAM)

    if not isinstance(grad_CAM, np.ndarray):
        return grad_CAM.numpy()

    return grad_CAM


def calculate_saliency_map(model: Model, data: np.ndarray, c: int, dependencies=None, normalise=True) -> np.ndarray:
    """
    https://arxiv.org/abs/1312.6034 - saliency maps
    https://arxiv.org/abs/1706.03825 - smoothed version
    Calculates saliency map / vanilla gradient for feedforward neural network. Calculated from all the data.
    Modified code from:
    :param c: class index
    :param normalise: output will be in range 0-1
    :param dependencies: custom dependencies for neural network
    :param model: model for saliency calculation
    :param data: data for
    :return: np.ndarray with saliency map
    """

    model_modified = change_last_activation_to_linear(model, dependencies)

    # calculate gradient d y/d input
    with tf.GradientTape() as g:
        data = tf.convert_to_tensor(data)
        g.watch(data)
        loss = model_modified(data)[:, c]
        d_loss_d_data = g.gradient(loss, data)

    grads = np.abs(d_loss_d_data)

    if normalise:
        grads = normalise_array(grads)

    grads = tf.reshape(grads, [grads.shape[0], grads.shape[1]])

    return grads.numpy()
