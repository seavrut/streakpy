"""rawccd.py
======
Python module containing the RawCCD parent class and ThreeCrystal and Trex child classes.

The RawCCD object stores a .tiff image as a numpy array and has built in methods for convenient
image processing of the data.

It is intented for a RawCCD instances to hold the entire image and Streak instances to isolate
the important part of the image.

Trex objects have methods for automatic edge-finding and vertical aligning of streaks.
"""
# NOTE: There are no public setters for variables. Access private class variables through __name within this file.

import numpy as np
from scipy import ndimage
from warnings import warn
from PIL import Image
import matplotlib.pyplot as plt
from typing import Union as U, Optional as O

import streak, filters, edge

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 6

class RawCCD:

    def __init__(self, filepath:str, index:int=0, verbose=True) -> None:
        try:
            with Image.open(filepath) as im:
                im.seek(index) # if filepath is a multipage .tif this will navigate to page i
                self._data = np.array(im)
                self._unfiltered_data = self.data
                self._verbose = verbose
                if verbose: print('Loaded page %d of %s' % (index, filepath))
                self._filename = filepath
                self._streaks = []
        except FileNotFoundError:
            print('ERROR: %s not found' % filepath)
        except EOFError:
            print('ERROR: %d index out of bounds in %s' % (index, filepath))

        

        '''remember: image data is stored in rows and columns which is opposite to typical coordinates: (y,x).
            a horizontal row across the image is to be accessed by the vertical distance from the top as data[y]
            a vertical column is to be accessed by data[:, x] by horizontal distance from the left edge'''

    @property
    def filename(self):
        return self._filename
    @property
    def streaks(self):
        return self._streaks
    @property
    def verbose(self):
        return self._verbose
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
    @property
    def data(self):
        return self._data
    @property
    def unfiltered_data(self):
        return self._unfiltered_data
    
    def rotate(self, degrees:float, reshape:bool=False, raw=False) -> None:
        '''Rotate image. Also updates the saved shape of the image.
        
        degrees: degrees to rotate image data by CCW (float) 
        reshape: whether to reshape image array to fit the full rotated image (bool=False)
        '''
        self._data = ndimage.rotate(self.data, degrees, reshape=reshape)
        if raw:
            self._unfiltered_data = ndimage.rotate(self.unfiltered_data, degrees, reshape=reshape)
    
    def subtract_background(self, value:int|np.ndarray, raw=False) -> None:
        '''Simple background subtractions of a value
        
        value: value to subtract from image data. can be a single float or an array with same dimensions as the image'''
        if isinstance(value, np.ndarray) and value.shape != self.shape:
            raise ValueError(f'value must have same dimensions as image. {value.shape} != {self.shape}')
        self._data -= value
        self._data = np.clip(self.data, 0, None)
        if raw:
            self._unfiltered_data -= value
            self._unfiltered_data = np.clip(self.unfiltered_data, 0, None)

    def plot(self, fig:O[plt.Figure]=None, ax:O[plt.Axes]=None, figsize=(3,3), subplot=111,
             vmin:O[float]=None, vmax:O[float]=None, cmap:O[str]='Greys_r', raw=False,
             savepath:O[str]=None) -> U[plt.Figure, plt.Axes]:
        '''Plot image
        
        ax: axes to plot image on
        vmin: minimum value of colormap
        vmax: maximum value of colormap
        '''
        if fig is None:
            fig = plt.figure(dpi=300, figsize=figsize, constrained_layout=True)
        if ax is None:
            ax = fig.add_subplot(subplot)
        if raw:
            ax.imshow(self.unfiltered_data, origin='upper',  vmin=vmin, vmax=vmax, cmap=cmap)
        else:
            ax.imshow(self.data, origin='upper',  vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set(xlabel='$x$', ylabel='$y$')
        ax.set_title(self.filename, fontsize=5)
        
        if savepath is not None:
            fig.savefig(savepath, transparent=False)
        return fig, ax


    def update(self):
        '''Updates all streaks stored in image.
            Does not reassign bounds. To do that, access streaks individually.'''
        for s in self.streaks:
            s.update()

    def vert_slice(self, x, raw=False) -> np.ndarray:
        if raw:
            return self.unfiltered_data[:, x]
        return self.data[:, x]

    def hori_slice(self, y, raw=False) -> np.ndarray:
        if raw:
            return self.unfiltered_data[y, :]
        return self.data[y, :]
    
    def rect(self, bounds:list|tuple|np.ndarray, raw=False) -> np.ndarray:
        '''Get rectangular section of image defined by upper left corner and lower right corner.
        
        bounds: (xmin, ymin, xmax, ymax) corner coordinates (list|tuple|np.ndarray)'''
        if len(bounds) != 4:
            raise ValueError('bounds must have length 4')
        xmin = bounds[0]
        ymin = bounds[1]
        xmax = bounds[2]
        ymax = bounds[3]

        if xmin < 0 or ymin < 0 or xmax > self.width or ymax > self.height:
            message = 'rect bounds exceed image bounds.'
            temp = (0, 0, self.width, self.height)
            message += f'\n {bounds} ⊄ {temp}'
            warn(message, RuntimeWarning)
        
        if raw:
            return self.unfiltered_data[ymin:ymax, xmin:xmax]
        return self.data[ymin:ymax, xmin:xmax]
    
    def crop(self, bounds:list|tuple|np.ndarray, raw=False) -> None:
        if len(bounds) != 4:
            raise ValueError('bounds must have length 4')
        xmin = bounds[0]
        ymin = bounds[1]
        xmax = bounds[2]
        ymax = bounds[3]

        self._data = self.data[ymin:ymax, xmin:xmax]
        if raw:
            self._unfiltered_data = self.unfiltered_data[ymin:ymax, xmin:xmax]

    def define_streak(self, bounds, raw=False) -> streak.Streak:
        self.streaks.append(streak.Streak(self, bounds, raw))
        return self.streaks[-1]
    
    def filter_image(self, filtfunc, *args, **kwargs):
        self._data = filtfunc(self.data, *args, **kwargs)

    def shift_image(self, shift, raw=False, *args):
        '''Shift the image using scipy.ndimage.shift

        shift: sequence specifying shift in each axis'''
        self._data = ndimage.shift(self._data, shift, *args)
        if raw:
            self._unfiltered_data = ndimage.shift(self.unfiltered_data, shift, *args)


class ThreeCrystal(RawCCD):
    def __init__(self, filepath:str, index:int = 0, verbose=True) -> None:
        super().__init__(filepath, index, verbose)
        self.rotate(90, True, True)

    def define_streak(self, bounds, raw=True) -> streak.ThreeCrystalStreak:
        self.streaks.append(streak.ThreeCrystalStreak(self, bounds, raw))
        return self.streaks[-1]

class Trex(RawCCD):
    def __init__(self, filepath:str, index:int=0, verbose=True) -> None:
        super().__init__(filepath, index, verbose)
        self.rotate(180, True, True)

    def define_streak(self, bounds, raw=True, y_edge_shift=0) -> streak.TrexStreak:
        self.streaks.append(streak.TrexStreak(self, bounds, raw, y_edge_shift))
        return self.streaks[-1]
    
    def detect_bottom_edge(self, high_percentile_thresh, low_percentile_thresh, size_thresh,
                            degree:int=1, border_width:int=10, simple_xmin=None, simple_xmax=None,
                            fine_pruning_thresh:int|float=50, fine_pruding_radius=50):
        '''detect edges of streak
        
        Tt: top edge threshold value
        Tb: bottom edge threshold value
        deg: polynomial degree to fit edge to (int, default=1)
        top_buffer: number of pixels to ignore from top of image (int, default=10)'''
        
        hori, vert, mag, angle = filters.sobel(self.data)

        workingdata = -hori

        threshold1 = np.percentile(workingdata, high_percentile_thresh)
        threshold2 = np.percentile(workingdata, low_percentile_thresh)

        edges = np.ones_like(workingdata) * 0.5
        edges = np.where(workingdata > threshold1, 1.0, edges)
        edges = np.where(workingdata < threshold2, 0.0, edges)
        for i, row in enumerate(edges[1:-1]):
            for j, col in enumerate(row[1:-1]): # avoiding the edges for indexing reasons
                if edges[i,j]== 0.5:
                    if np.any(edges[i-1:i+2, j-1:j+2] == 1):
                        edges[i,j] = 1.0
                    else:
                        edges[i,j] = 0.0

        edges = ndimage.binary_opening(edges)
        edges = ndimage.binary_closing(edges)

        mask_border = np.ones_like(edges, dtype=int)
        mask_border[border_width:-border_width, border_width:-border_width] = 0
        mask_border = mask_border.astype(bool)
        edges[mask_border] = 0

        labeled_edges, num_labels = ndimage.label(edges)
        sizes = ndimage.sum(edges, labeled_edges, range(num_labels + 1))
        mask_size = sizes < size_thresh
        labeled_edges[mask_size[labeled_edges]] = 0
        labeled_edges = np.clip(labeled_edges, 0, 1)

        bottom_edge_arr = np.zeros(self.width, dtype=int)
        for i, col in enumerate(labeled_edges.T):
            mask = np.argwhere(col >= 1)
            if len(mask) > 0:
                index = mask.T[0][-1]
                bottom_edge_arr[i] = index

        temp = edge.Edge(np.arange(self.width), bottom_edge_arr, degree=degree, simple_xmin=simple_xmin, simple_xmax=simple_xmax)

        if temp.exists:
            temp.fine_pruning(fine_pruning_thresh, fine_pruding_radius)
            self.bottom_edge = temp
            return self.bottom_edge
        else:
            message = 'No edge found.'
            if self.verbose: warn('No edge found.', RuntimeWarning)
            raise RuntimeError(message)


    def detect_streak_bounds(self, x:O[int], top_pad=100, bot_pad=125, left_pad=500, right_pad=250):
        if x is None: # column to evaluate streak y position at
            x = int(self.width/2)

        bot = self.bottom_edge.fit(x)
        y = int(np.rint(bot))

        bounds = (x - left_pad, y - top_pad, x + right_pad, y + bot_pad)
        if bounds[0] < 0 or bounds[1] < 0 or bounds[2] > self.width or bounds[3] > self.height:
            message = 'Streak bounds exceed image bounds.'
            temp = (0, 0, self.width, self.height)
            message += f'\n {bounds} ⊄ {temp}'
            warn(message, RuntimeWarning)
            raise RuntimeError(message)
        return bounds, y - top_pad
        