'''streak.py
===
Contains the Streak parent class and ThreeCrystalStreak and TrexStreak child classes.
A Streak instance holds a zoomed in portion of a RawCCD image where the streak actually is.
Has methods for plotting and giving vertically integrated line-outs.

The TrexStreak class also has methods for straightening a streak and obtaining finer
time-resolved line-outs. 
'''

import numpy as np
import matplotlib.pyplot as plt
from typing import Union as U, Optional as O

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 6

import rawccd

class Streak:
    def __init__(self, ccd:"rawccd.RawCCD", bounds:list|tuple|np.ndarray, raw=True) -> None:
        self._bounds = bounds
        self.filename = ccd.filename
        self._data = ccd.rect(self.bounds, raw)
        # if calculated, a streak should inherit edges from the ccd
        self._lineout = self._integrate_section()
        #self._y = np.arange(self.bounds[1], self.bounds[3])
        self._y = np.arange(self.height)
        self._x = np.arange(self.bounds[0], self.bounds[2])
        self._energy = self.x
    
    @property
    def bounds(self):
        return self._bounds
    @property
    def data(self):
        return self._data
    @property
    def lineout(self):
        return self._lineout
    @property
    def x(self):
        return self._x
    @property
    def y(self):
        return self._y
    @property
    def energy(self):
        # NEED TO IMPLEMENT
        return self._energy
    @property
    def shape(self):
        '''image shape (num rows, num cols)'''
        return self.data.shape
    @property
    def width(self):
        return self.shape[1]
    @property
    def height(self):
        return self.shape[0]
    
    def plot(self, fig:O[plt.Figure]=None, ax:O[plt.Axes]=None, figsize=(3,1.5), subplot=111,
             vmin:O[float]=None, vmax:O[float]=None, cmap='Greys_r', savepath:O[str]=None) -> U[plt.Figure, plt.Axes]:
        '''Plot image
        '''
        if fig is None:
            fig = plt.figure(dpi=300, figsize=figsize, constrained_layout=True)
        if ax is None:
            ax = fig.add_subplot(subplot)

        ax.imshow(self.data, extent=(self.x[0], self.x[-1], self.y[-1], self.y[0]),
                  vmin=vmin, vmax=vmax, cmap=cmap)
        #ax.set(xlabel='$x$', ylabel='$y$')
        ax.set_title(self.filename, fontsize=5)
        fig.show()
        if savepath is not None:
            fig.savefig(savepath, transparent=False)
        return fig, ax
    
    def plot_lineout(self, fig:O[plt.Figure]=None, ax:O[plt.Axes]=None, figsize=(3,1.5), subplot=111,
                     savepath:O[str]=None, **kwargs) -> U[plt.Figure, plt.Axes]:
        '''Plot lineout
        '''
        if fig is None:
            fig = plt.figure(dpi=300, figsize=figsize, constrained_layout=True)
        if ax is None:
            ax = fig.add_subplot(subplot)

        color = kwargs.pop('c', 'k')
        linewidth = kwargs.pop('lw', 0.75)

        ax.plot(self.x, self.lineout, c=color, lw=linewidth, **kwargs)
        ax.set(xlabel=r'$x$', ylabel=r'$\textrm{counts}$')
        fig.show()
        if savepath is not None:
            fig.savefig(savepath, transparent=False)
        return fig, ax

    def filter_lineout(self, filtfunc, *args, **kwargs) -> None:
        '''Apply a 1D filter to the lineout
        '''
        self._x, self._lineout =  filtfunc(self.x, self.lineout, *args, **kwargs)

    def filter_image(self, filtfunc, *args, **kwargs):
        self._data = filtfunc(self.data, *args, **kwargs)
    
    """ def update(self, bounds:list|tuple|np.ndarray=None) -> None:
        '''Pull fresh image data from ccd if it has been updated.
           Opportunity to define new bounds.
           Un-does any filter previously applied to the lineout.'''
        if bounds is not None: # update to new bounds
            if len(bounds) != 4:
                raise ValueError('bounds must have length 4')
            self._bounds = bounds
        self._data = self._ccd.rect(self.bounds)
        self._lineout = self._integrate_section()
        self._y = np.arange(self.bounds[1], self.bounds[3])
        self._x = np.arange(self.bounds[0], self.bounds[2])
        self._energy = self.x """

    def straighten(self) -> None:
        raise NotImplementedError

    def _integrate_section(self) -> np.ndarray:
        '''Sums vertically across the streak'''
        line_out = np.zeros(self.width)
        for i, col in enumerate(self.data.T): # iterates over each column in section
            line_out[i] = np.sum(col)
        return line_out
    

class ThreeCrystalStreak(Streak):
    def __init__(self, ccd:"rawccd.ThreeCrystal", bounds, raw=True) -> None:
        super().__init__(ccd, bounds, raw)

class TrexStreak(Streak):
    def __init__(self, ccd:"rawccd.Trex", bounds, raw=True, y_edge_shift=0) -> None:
        super().__init__(ccd, bounds, raw)
        try:
            self.bottom_edge = self._ccd.bottom_edge
            self.bottom_edge._y -= y_edge_shift
            self.bottom_edge.fit(recalc=True)
        except AttributeError:
            pass
