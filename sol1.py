################ imports ###############
import numpy as np
from skimage.color import rgb2gray
from scipy.misc import imread
import matplotlib.pyplot as plt

############### constants ###############
RGB_SHAPE = 3
GRAY_MAX_VALUE = 255

COLOR_CONVERSION_MATRIX = np.array([[0.299, 0.587, 0.114],
                                    [0.596, -0.275, -0.321],
                                    [0.212, -0.523, 0.311]])

INV_COLOR_CONVERSION_MATRIX = np.linalg.inv(np.transpose(COLOR_CONVERSION_MATRIX))


############## functions ################


def read_image(filename, representation):
    """
    3.1 Reading an image into a given representation.
    :param filename: read_image(filename, representation).
    :param representation: representation code, either 1 or 2 defining
    whether the output should be a grayscale image (1) or an RGB image (2).
    If the input image is grayscale, we won’t call it with representation = 2.
    :return: This function returns an image, make sure the output image
    is represented by a matrix of type np.float64 with intensities
    (either grayscale or RGB channel intensities)
    normalized to the range [0, 1].
    """

    if representation == 1:

        return rgb2gray(imread(filename)).astype(np.float64)

    elif representation == 2:

        return ((imread(filename)) / GRAY_MAX_VALUE).astype(np.float64)


def imdisplay(filename, representation):
    """

    :param filename:
    :param representation:
    :return:
        """

    if representation == 1:

        plt.imshow((read_image(filename, representation)), cmap=plt.cm.gray)

    elif representation == 2:

        plt.imshow(read_image(filename, representation))

    plt.show()


def rgb2yiq(imRGB):
    """
    This function transform an image from rgb format to yiq format
    :param imRGB: the rgb image to transform
    :return:
    """
    return imRGB.dot(np.transpose(COLOR_CONVERSION_MATRIX)).clip(min=-1, max = 1)


def yiq2rgb(imYIQ):
    """
    This function transform an image from yiq format to rgb format
    :param imYIQ: the image at the yiq format
    :return:
    """
    return imYIQ.dot(INV_COLOR_CONVERSION_MATRIX)


def histogram_equalize(im_orig):
    """
    This function transform an image from rgb format to yiq format
    :param im_orig: is the input grayscale or RGB
    float64 image with values in [0, 1].
    :return: The function returns a list [im_eq, hist_orig, hist_eq] where
    im_eq - is the equalized image. grayscale or RGB float64
    image with values in [0, 1].
    hist_orig - is a 256 bin histogram of
    the original image (array with shape (256,) ).
    hist_eq - is a 256 bin histogram
    of the equalized image (array with shape (256,) ).
    """
    rgb_flag = False
    temp = []
    im_copy = im_orig.copy().astype(np.float64) \
        if im_orig.dtype != np.float64 else im_orig.copy()
    if len(im_copy.shape) == RGB_SHAPE:
        # if the image is rgb
        temp = rgb2yiq(im_copy)
        im_copy = (temp)[:, :, 0]
        rgb_flag = True

    im_eq, hist_orig, hist_eq = calculate_histograms(im_copy)

    if rgb_flag:
        im_eq = yiq2rgb(np.stack([im_eq, temp[:,:, 1], temp[:, :, 2]], axis=2))

    return [im_eq, hist_orig, hist_eq]



def calculate_histograms(im_copy):
    """
    This function calculates the histograms of the image
    :param im_copy: the copy of the original image.
    :return:
    """
    im_copy_255 = (im_copy * 255).astype(np.uint8)
    hist_orig, bins = np.histogram(im_copy_255, range=(0, 256), bins=256)
    # plt.bar(bins[:-1], hist_orig, width=0.5)
    # plt.xlim(min(bins), max(bins))
    # plt.show()
    cum_hist = hist_orig.cumsum()
    # plt.bar(bins[:-1], cum_hist, width=0.5)
    # plt.xlim(min(bins), max(bins))
    # plt.show()
    im_eq, hist_eq = normalized_cumulative_histogram(cum_hist, bins, im_copy_255)
    return im_eq, hist_orig, hist_eq

def normalized_cumulative_histogram(cumsum_hist, bins, im_orig):
    """
    This function computes the normalized cumulative histogram of the image.
    :param cumsum_hist:
    :param img_orig:  is the input grayscale or RGB
    float64 image with values in [0, 1].
    :return:
    """
    m = np.argmax(cumsum_hist > 0)

    def T(k):
        """
        T(k) = round{ [C(k)-C(m)] / [C(255)-C(m)] × 255}
        :param index:
        :param m:
        :param cumsum_hist:
        :return:
        """
        temp = cumsum_hist[k] - cumsum_hist[m]
        temp2 = cumsum_hist[255] - cumsum_hist[m]
        return ((temp / temp2) * 255).round().astype(np.uint8)

    look_up_table = np.vectorize(T)
    im_eq = look_up_table(im_orig)

    hist_eq = np.histogram(im_eq, range=(0, 256), bins=256)[0]
    # plt.bar(bins[:-1], hist_eq, width=0.5)
    # plt.xlim(min(bins), max(bins))
    # plt.show()

    return (im_eq / GRAY_MAX_VALUE) , hist_eq

def quantize (im_orig, n_quant, n_iter):
    """
    Write a function that performs optimal quantization of a given
    grayscale or RGB image.
    :param im_orig: is the input grayscale or RGB image to be quantized
        (float64 image with values in [0, 1]).
    :param n_quant: is the number of intensities the output
        im_quant image should have.
    :param n_iter: is the maximum number of iterations of the
        optimization procedure (may converge earlier.)
    :return: a list [im_quant, error] where:
        im_quant - is the quantized output image.
        error - is an array with shape (n_iter,) (or less)
        of the total intensities error for each iteration of
        the quantization procedure.
    """
    error = np.array(list())
    temp = []
    rgb_flag = False
    im_copy = im_orig.copy().astype(np.float64) \
        if im_orig.dtype != np.float64 else im_orig.copy()
    if len(im_copy.shape) == RGB_SHAPE:
        # if the image is rgb
        temp = rgb2yiq(im_copy)
        im_copy = (temp)[:, :, 0]
        rgb_flag = True

    im_copy_255 = (im_copy * GRAY_MAX_VALUE).astype(np.uint8)

    hist_orig, bins = np.histogram(im_copy_255, bins=range(0, 257))

    # calculating the first segments
    new_z = np.rint(np.quantile(im_copy_255, np.linspace(0, 1, n_quant + 1))).astype(np.uint8)
    new_z[0] = 0
    new_z[-1] = GRAY_MAX_VALUE

    z = new_z.copy()

    quantums = np.zeros(n_quant, dtype= np.float64)

    for i in range(n_iter):

        quantums = calculate_quantum_values(hist_orig, new_z, quantums)

        new_z = calculate_segments(new_z, quantums)

        if np.array_equal(z ,new_z):

            break

        error = np.append(error, calculate_error(new_z, hist_orig, quantums, n_quant))

        z = new_z.copy()

    digitized_map = np.digitize(range(0, 256), new_z, True) -1
    digitized_map[0] = 0

    def look_up_table(digitized_pixel):
        """
        This function is the lookup table needed to convert all the pixels
        at the photo to their new bins.
        :param n_quant: is the number of intensities the output
        im_quant image should have.
        :param quantums: The current quantums array
        :return: The new value of the pixel
        """
        return quantums[digitized_map[digitized_pixel]]


    apply_look_up_table = np.vectorize(look_up_table)

    im_quant = apply_look_up_table(im_copy_255) / GRAY_MAX_VALUE

    if rgb_flag:

        im_quant = yiq2rgb(

            np.stack([im_quant, temp[:, :, 1], temp[:, :, 2]], axis=2))

    return [im_quant, np.array(error)]


def calculate_quantum_values(hist_orig, z, quantums):
    """
    This function calculates the new values of the quantums
     array according to the current z bins.
    :param hist_orig:
    :param z: The bins array
    :return: The new quantums array (float64)
    """

    for i in range(len(quantums)):

        z_range = np.arange(z[i] + 1, z[i + 1] + 1)
        hist_z_range = hist_orig[z_range]
        hist_z_range_sum = hist_z_range.sum()

        quantums[i] = (z_range.dot(hist_z_range)) / hist_z_range_sum

    return  np.round(quantums)

def calculate_error(z, hist_orig, quantums, n_quants):
    """
    This function calculates the error for a given z and q arrays.
    :param z: the segments array
    :param hist_orig: the image histogram
    :param quantums: The current q values array
    :return: the current error calculated according to the given z and q
    values.
    """
    err = 0

    for i in range(n_quants):

        for j in range(z[i] + 1, z[i + 1] + 1):

            err += ((quantums[i] - j) ** 2) * hist_orig[j]

    err += (quantums[-1] - GRAY_MAX_VALUE) ** 2 * hist_orig[GRAY_MAX_VALUE]

    return err


def calculate_segments(z, quantums):
    """
    This function calculates the new bins according to the current quantums.
    :param n_quant: is the number of intensities the output
        im_quant image should have.
    :param quantums: The quantums array.
    :param old_z: the old bins array.
    :return: The new bins array (int8)
    """

    for i in range(1, len(z) - 1):

        z[i] = (quantums[i-1] + quantums[i]) / 2

    return np.round(z)
