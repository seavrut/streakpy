"""filters.py
===
Holds a couple useful filters for signal processing of CCD data.

1D: Low pass, Savitzky-Golay
2D: median, Gaussian, Sobel
"""

import numpy as np
from scipy import signal, ndimage
from typing import Union as U, Optional as O


def lowpass1d(x, y, cutoff, width, atten=60) -> U[np.ndarray, np.ndarray]:
    '''Apply a simple low-pass filter to a signal.
    
    cutoff: frequency to center cutoff at
    width: width of transition region at cutoff
    atten: how much to attenuate cutoff frequencies, in dB

    returns -> x_filtered, y_filtered
    '''
    from scipy.signal import kaiserord, lfilter, firwin

    sample_rate =  1
    nyq_rate = sample_rate / 2.0
    transwidth = width/nyq_rate # width of transition region in Hz
    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(atten, transwidth)    
    taps = firwin(N, cutoff/nyq_rate, window=('kaiser', beta))
    y_filtered = lfilter(b=taps, a=1.0, x=y)
    y_filtered = y_filtered[N-1:]
    delay = 0.5 * (N-1) / sample_rate
    x_filtered = x[N-1:]-delay
    return x_filtered, y_filtered

def savgol1d(x:np.ndarray, y:np.ndarray, window_length:int, polyorder:int, **kwargs
                      ) -> U[np.ndarray, np.ndarray]:
        y_filtered = signal.savgol_filter(x=y,
                                          window_length=window_length,
                                          polyorder=polyorder,
                                          **kwargs)
        x_filtered = x
        return x_filtered, y_filtered

def medfilt2d(image:np.ndarray, kernel) -> np.ndarray:
    return signal.medfilt2d(image, kernel)

def gaussian2d(image:np.ndarray, sigma, **kwargs) -> np.ndarray:
    return ndimage.gaussian_filter(image, sigma, **kwargs)

def median_filter(image:np.ndarray, kernel) -> np.ndarray:
    return ndimage.median_filter(image, kernel)

def sobel(image:np.ndarray) -> U[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Applies Sobel operator to an image.
    
    returns: horizontal Sobel, vertical Sobel, Sobel magnitude, Sobel angle in degrees
    """
    hori = ndimage.sobel(np.float32(image), axis=0)
    vert = ndimage.sobel(np.float32(image), axis=1)

    magnitude = np.sqrt(hori**2 + vert**2)
    angle = np.rad2deg(np.arctan2(vert,hori)) # [-180, 180]
    
    return hori, vert, magnitude, angle