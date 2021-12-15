import os
import cv2
import numpy as np
from math import exp
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

# https://pyradar-tools.readthedocs.io/en/latest/_modules/pyradar/filters/lee.html#lee_filter
K_DEFAULT = 1.0
CU_DEFAULT = 0.523
CMAX_DEFAULT = 1.73


def weighting(pix_value, window, k=K_DEFAULT,
              cu=CU_DEFAULT, cmax=CMAX_DEFAULT):
    """
    Computes the weighthing function for Lee filter using cu as the noise
    coefficient.
    """

    # cu is the noise variation coefficient

    # ci is the variation coefficient in the window
    window_mean = window.mean()
    window_std = window.std()
    ci = window_std / window_mean

    if ci <= cu:  # use the mean value
        w_t = 1.0
    elif cu < ci < cmax:  # use the filter
        w_t = exp((-k * (ci - cu)) / (cmax - ci))
    elif ci >= cmax:  # preserve the original value
        w_t = 0.0

    return w_t

def assert_window_size(win_size):
    """
    Asserts invalid window size.
    Window size must be odd and bigger than 3.
    """
    assert win_size >= 3, 'ERROR: win size must be at least 3'

    if win_size % 2 == 0:
        print('It is highly recommended to user odd window sizes.'\
              'You provided %s, an even number.' % (win_size, ))


def assert_indices_in_range(width, height, xleft, xright, yup, ydown):
    """
    Asserts index out of image range.
    """
    assert xleft >= 0 and xleft <= width, \
        "index xleft:%s out of range (%s<= xleft < %s)" % (xleft, 0, width)

    assert xright >= 0 and xright <= width, \
        "index xright:%s out of range (%s<= xright < %s)" % (xright, 0, width)

    assert yup >= 0 and yup <= height, \
        "index yup:%s out of range. (%s<= yup < %s)" % (yup, 0, height)

    assert ydown >= 0 and ydown <= height, \
        "index ydown:%s out of range. (%s<= ydown < %s)" % (ydown, 0, height)


def lee_filter(img, win_size=3, cu=0.25):
    """
    Apply lee to a numpy matrix containing the image, with a window of
    win_size x win_size.
    """
    assert_window_size(win_size)

    # we process the entire img as float64 to avoid type overflow error
    img = np.float64(img)
    img_filtered = np.zeros_like(img)
    N, M = img.shape
    win_offset = win_size // 2

    for i in range(0, N):
        xleft = i - win_offset
        xright = i + win_offset

        if xleft < 0:
            xleft = 0
        if xright >= N:
            xright = N

        for j in range(0, M):
            yup = j - win_offset
            ydown = j + win_offset

            if yup < 0:
                yup = 0
            if ydown >= M:
                ydown = M

            assert_indices_in_range(N, M, xleft, xright, yup, ydown)

            pix_value = img[i, j]
            window = img[xleft:xright, yup:ydown]
            # w_t = weighting(window, cu)
            w_t = 1.0 
            window_mean = window.mean()
            new_pix_value = (pix_value * w_t) + (window_mean * (1.0 - w_t))
            if new_pix_value < 0: new_pix_value =0 
            # assert new_pix_value >= 0.0, \
            #         "ERROR: lee_filter(), pixel filtered can't be negative"

            img_filtered[i, j] = round(new_pix_value)

    return img_filtered

    '''
    https://github.com/birgander2/PyRAT/blob/master/pyrat/filter/Despeckle.py
    '''

'''
https://github.com/dityatamas/final-project-speckle-reduction/blob/ce331b2d669242497c8d24d9e1a9cd903894c877/despeckling.py
'''
def SRAD(src, niter=7, gamma=0.8, filter_size=5, option=1):
    img = src.copy()

    img = cv2.normalize(src.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # Convert to normalized floating point
    # M, N = img.shape

    imgout = img.copy()
    # imgout = np.exp(imgout)

    index = filter_size // 2
    data_final = imgout.copy()

    deltaN = np.zeros_like(imgout)
    deltaS = deltaN.copy()
    deltaE = deltaN.copy()
    deltaW = deltaN.copy()
    g2 = deltaN.copy()
    l = deltaN.copy()
    q0_squared = deltaN.copy()
    q_squared = deltaN.copy()
    g = deltaN.copy()

    M = imgout.shape[0]
    N = imgout.shape[1]

    for ii in range(niter):
        # Homogeneous ROI to calculate the ICOV
        for p in range(index, M - index):
            for q in range(index, N - index):
                q0_squared[p, q] = np.std(data_final[p, q]) / ((np.mean(data_final[p, q]))** 2)

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                deltaN[i, j] = imgout[i, j - 1] - imgout[i, j]
                deltaS[i, j] = imgout[i, j + 1] - imgout[i, j]
                deltaE[i, j] = imgout[i + 1, j] - imgout[i, j]
                deltaW[i, j] = imgout[i - 1, j] - imgout[i, j]

                # Normalized discrete gradient magnitude squared
                g2[i, j] = (deltaN[i, j] ** 2 + deltaS[i, j] ** 2 + deltaE[i, j] ** 2 + deltaW[i, j] ** 2) / ((imgout[i, j]) ** 2)

                # Normalized discrete laplacian
                l[i, j] = (deltaN[i, j] + deltaS[i, j] + deltaE[i, j] + deltaW[i, j]) / imgout[i, j]

                # Instantaneous coefficient of variation for edge detection
                q_squared[i, j] = ((0.5 * g2[i, j]) - ((1 / 16) * (l[i, j] ** 2))) / ((1 + ((1 / 4) * l[i, j])) ** 2)

                # conduction gradients
                if option == 1:
                    g[i, j] = 1 / (1 + (q_squared[i, j] - q0_squared[i, j]) / (q0_squared[i, j] * (1 + q0_squared[i, j])))
                elif option == 2:
                    g[i, j] = np.exp(-(q_squared[i, j] - q0_squared[i, j]) / (q0_squared[i, j] * (1 + q0_squared[i, j])))

                g[i, j] = np.nan_to_num(g[i, j])

                # d = (g * deltaE) + (g * deltaW) + (g * deltaS) + (g * deltaN)
                imgout[i, j] = imgout[i, j] + (gamma / 4) * (g[i, j] * deltaN[i, j] + g[i, j + 1] * deltaS[i, j] + g[i + 1, j] * deltaE[i, j] + g[i, j] *deltaW[i, j])

    # imgout = imgout.astype("uint8")

    return imgout*255

def OSRAD(src, niter=7, gamma=0.8, kernel=5, option=1):
    img = src.copy()

    img = cv2.normalize(src.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # Convert to normalized floating point
    # M, N = img.shape

    imgout = img.copy()
    # imgout = np.exp(imgout)

    index = kernel // 2
    data_final = imgout.copy()

    deltaN = np.zeros_like(imgout)
    deltaS = deltaN.copy()
    deltaE = deltaN.copy()
    deltaW = deltaN.copy()
    g2 = deltaN.copy()
    l = deltaN.copy()
    q0_squared = deltaN.copy()
    q_squared = deltaN.copy()
    g = deltaN.copy()
    abc = np.ones_like(imgout)
    ctang = 1

    M = imgout.shape[0]
    N = imgout.shape[1]

    for ii in range(niter):
        # Homogeneous ROI to calculate the ICOV
        for p in range(index, M - index):
            for q in range(index, N - index):
                q0_squared[p, q] = np.std(data_final[p, q]) / ((np.mean(data_final[p, q]))** 2)

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                deltaN[i, j] = imgout[i, j - 1] - imgout[i, j]
                deltaS[i, j] = imgout[i, j + 1] - imgout[i, j]
                deltaE[i, j] = imgout[i + 1, j] - imgout[i, j]
                deltaW[i, j] = imgout[i - 1, j] - imgout[i, j]

                # Normalized discrete gradient magnitude squared
                g2[i, j] = (deltaN[i, j] ** 2 + deltaS[i, j] ** 2 + deltaE[i, j] ** 2 + deltaW[i, j] ** 2) / ((imgout[i, j]) ** 2)

                # Normalized discrete laplacian
                l[i, j] = (deltaN[i, j] + deltaS[i, j] + deltaE[i, j] + deltaW[i, j]) / imgout[i, j]

                # Instantaneous coefficient of variation for edge detection
                q_squared[i, j] = ((0.5 * g2[i, j]) - ((1 / 16) * (l[i, j] ** 2))) / ((1 + ((1 / 4) * l[i, j])) ** 2)

                # conduction gradients
                if option == 1:
                    g[i, j] = 1 / (1 + (q_squared[i, j] - q0_squared[i, j]) / (q0_squared[i, j] * (1 + q0_squared[i, j])))
                elif option == 2:
                    g[i, j] = np.exp(-(q_squared[i, j] - q0_squared[i, j]) / (q0_squared[i, j] * (1 + q0_squared[i, j])))

                g[i, j] = np.nan_to_num(g[i, j])

                im = np.identity(2, dtype=float)
                ik = [(g[i,j]),ctang]
                ikS = [(g[i, j+1]), ctang]
                ikE = [(g[i+1, j]), ctang]
                d = ik*im
                dS = ikS*im
                dE = ikE*im
                tmp = (d * deltaN[i, j] + dS * deltaS[i, j] + dE * deltaE[i, j] + d *deltaW[i, j])
                abc[i,j] = abc[i,j] * (gamma / 4) * tmp.sum()
                imgout[i, j] = imgout[i, j] + abc[i,j]
    # imgout = imgout.astype("uint8")

    return imgout*255 