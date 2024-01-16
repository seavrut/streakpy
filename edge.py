"""edge.py
===
Python module containing the Edge class.
"""

import numpy as np

class Edge:
    '''Edge
    ===
    On a 2D image, stores x and y arrays of an edge.
    Runs the whole width of the image but masks the values of values not on the edge.
    Fits the edge to a polynomial of a given degree and stores those coefficients.
    Has methods for pruning outliers to the edge, shifting the edge vertically, and
    saving to a file.
    '''
    def __init__(self, xarr, yarr, degree=1, simple_ythresh:int|float=100, simple_xthresh:int|float=200, simple_xmin=None, simple_xmax=None) -> None:
        self._x = xarr
        self._y = yarr
        self.degree = degree
        self.exists = len(self.y) > 10 and self.ymedian > 175
        if self.exists:
            self.simple_pruning(simple_ythresh, simple_xthresh, simple_xmin, simple_xmax)
            #self.coeffs = np.polyfit(self.x, self.y, self.degree)

    @property
    def y(self):
        return self._y[self._y > 0]
    
    @property
    def x(self):
        return self._x[self._y > 0]
    
    @property
    def ymedian(self):
        return np.median(self.y)
    
    @property
    def xmedian(self):
        return np.median(self.x)

    def fit(self, x = None, recalc=False, degree=None):
        if degree is not None:
            self.degree = degree
        if recalc:
            self.coeffs = np.polyfit(self.x, self.y, self.degree)
        if x is None:
            x = self.x
        return np.polyval(self.coeffs, x)
    
    def simple_pruning(self, ythresh, xthresh, xmin=None, xmax=None):
        '''simple pruning. gets rid of all values outside of median +/- threshold
         
        optional xlim parameter: if not None, will prune edge values at x < xlim '''
        #mask = np.any(self.y > self.ymedian + ythresh or self.y < self.ymedian - ythresh)
        self._y[self._y > self.ymedian + ythresh] = 0
        self._y[self._y < self.ymedian - ythresh] = 0

        self._y[self._x > self.xmedian + xthresh] = 0
        self._y[self._x < self.xmedian - xthresh] = 0
        
        if xmin is not None:
            self._y[self._x < xmin] = 0

        if xmax is not None:
            self._y[self._x > xmax] = 0

        self.exists = len(self.y) > 10 and self.ymedian > 175
        if self.exists:
            self.coeffs = np.polyfit(self.x, self.y, self.degree)

    def fine_pruning(self, ythresh, window_radius):
        '''for each array value, prune the value if it is an outlier in a window around it'''
        pruning_indices = []
        for i, value in enumerate(self._y):
            if value > 0:
                if i < window_radius:
                    window = self._y[:i+window_radius]
                elif i > len(self._y) - window_radius:
                    window = self._y[i-window_radius:]
                else:
                    window = self._y[i-window_radius:i+window_radius]
                window_median = np.median(window)
                if np.abs(value - window_median) > ythresh:
                    # prune the value
                    pruning_indices.append(i)
        self._y[pruning_indices] = 0
        self.exists = len(self.y) > 10 and self.ymedian > 175
        if self.exists:
            self.coeffs = np.polyfit(self.x, self.y, self.degree)

    def shift(self, yshift):
        for i, value in enumerate(self._y):
            if value > 0:
                self._y[i] += yshift

    def save(self, filepath, verbose=False):
        with open(filepath, 'w') as f:
            for i, j in zip(self._x, self._y):
                s = f'{i}, {j}\n'
                f.write(s)

        if verbose: print(f'Saved edge to {filepath}')


