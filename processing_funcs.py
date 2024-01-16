"""processing_funcs.py
===
Helper script of this package. Contains functions used in processing.py.
"""

__version__ = '2.2.0'
__author__ = 'Sofia Avrutsky'

import numpy as np
import os
import matplotlib.pyplot as plt
from time import perf_counter
from argparse import ArgumentParser
from configparser import ConfigParser, ExtendedInterpolation, ParsingError
from PIL import Image
from scipy.optimize import minimize_scalar
from scipy.interpolate import CubicSpline
from scipy.signal import peak_widths, savgol_filter
from typing import Union as U, Optional as O, overload
from os import PathLike

import rawccd, filters, edge

def remove_backgrounds():
    pass

def generate_background(files, fullpath):
    temp = rawccd.Trex(os.path.join(fullpath, files[0]), verbose=False)
    all_images = np.zeros((len(files), temp.height, temp.width), dtype=float)
    all_images[0] = temp.data
    for i, f in enumerate(files[1:]):
        with Image.open(os.path.join(fullpath, f)) as im:
            all_images[i] = np.array(im, dtype=float)
    avg_image = all_images.mean(axis=0)
    background_row_averages = avg_image.mean(axis=0)
    background = np.ones_like(avg_image) * background_row_averages
    background = np.rot90(np.rot90(background))
    return background

def calculate_edges(high_percentile_thresh:int|float, low_percentile_thresh:int|float,
        size_thresh:int|float, medfilt_kernel_size:int|tuple|list,
        degree:int, fine_pruning_thresh:int|float, fine_pruning_radius:int,
        simple_xmin:int|float, simple_xmax:int|float, crop_bounds:tuple|list,
        fullpath:str, edgepath:str, croppath:str, verbose:bool=False):
    
    files = os.listdir(fullpath)
    try:
        files.remove('.DS_Store')
    except ValueError:
        pass
    files.sort()

    print('%d total images'%len(files))

    # generate background
    background = generate_background(files, fullpath)
    background = (np.clip(background, 0, None)).astype(np.uint16)

    for i, f in enumerate(files):
        if verbose: print('%d/%d: ' % (i+1, len(files)), end='')
        else:
            print('.', end='', flush=True)
            if i % 10 == 0: print('|', end='')
            if i % 50 == 0: print()

        ccd = rawccd.Trex(os.path.join(fullpath, f), verbose=verbose)
        ccd.subtract_background(background, raw=True)
        ccd.crop(crop_bounds, raw=True)
        im = Image.fromarray(ccd.unfiltered_data)
        ccd.filter_image(filters.median_filter, medfilt_kernel_size)
        try:
            edge = ccd.detect_bottom_edge(high_percentile_thresh, low_percentile_thresh, size_thresh, degree=degree,
                                        simple_xmin=simple_xmin, simple_xmax=simple_xmax,
                                        fine_pruning_thresh=fine_pruning_thresh, fine_pruding_radius=fine_pruning_radius)
            name = f[:-5] + '.csv'
            edge.save(os.path.join(edgepath, f[:-5] + '.csv'), verbose)
            """ except AttributeError:
            if verbose: print('Skipping..\n') """
            if verbose: print(f'Saved cropped image to {os.path.join(croppath, f)}')
            im.save(os.path.join(croppath, f))
        except RuntimeError:
            if verbose: print('\tSkipping...')
    print()

def align_streaks_e(align_x:int, align_y:int, degree:int,
                simple_xmin:int, simple_xmax:int,
                simple_xthresh:int|float, simple_ythresh:int|float,
                weights_params:tuple|list,
                px_to_ps:float,
                edgepath:str, croppath:str, miscpath:str,
                show_fig:bool=True, save_fig:bool=True, verbose:bool=False):
    
    def error(y_shift, y, y_mean, weights=None):
        arr = (y_mean - (y+y_shift))**2
        if weights is None:
            weights = np.ones_like(arr)
        return np.dot(arr, weights)

    def gauss(x, gauss_param):
        assert len(gauss_param) == 2 or len(gauss_param) == 5
        mu1 = gauss_param[0]
        sigma1 = gauss_param[1]
        x_ = (np.float16(x) - mu1) / sigma1
        out = np.exp(-0.5*x_**2) / np.sqrt(2.0*np.pi) / sigma1

        if len(gauss_param) == 5:
            mu2 = gauss_param[2]
            sigma2 = gauss_param[3]
            amp2 = gauss_param[4]
            x__ = (np.float16(x) - mu2) / sigma2
            if amp2 is None:
                amp2 = 1
            out += amp2 * np.exp(-0.5*x__**2) / np.sqrt(2.0*np.pi) / sigma2
        return out
    
    def y_to_time(y):
        return y*px_to_ps
    
    files = os.listdir(edgepath)
    try:
        files.remove('.DS_Store')
    except ValueError:
        pass
    files.sort()
    print('%d total images'%len(files))

    try:
        x, y = np.genfromtxt(os.path.join(edgepath, files[0]), dtype=np.uint16, unpack=True, delimiter=',')
    except IndexError:
        raise SystemExit(f'{edgepath} is empy')

    WEIGHTS = gauss(x, weights_params)

    all_edges = np.zeros((len(files), len(y)))
    shifts = np.ones(len(files), dtype=int) * -999
    bad_streak_filenames = []

    fig, ax = plt.subplots(2,2, dpi=300, figsize=(6,3), sharex=True)
    fig2, ax2 = plt.subplots(1,1, dpi=300, figsize=(3,2))
    ax = ax.flatten()

    for i, f in enumerate(files):
        print('.', end='', flush=True)
        if i % 10 == 0:
            print('|', end='')
        if i % 50 == 0:
            print()

        _, y = np.genfromtxt(os.path.join(edgepath, f), unpack=True)
        
        e = edge.Edge(x, y, degree, 
                      simple_ythresh=simple_ythresh,
                      simple_xthresh=simple_xthresh, 
                      simple_xmin=simple_xmin,
                      simple_xmax=simple_xmax)
        try:
            yshift = align_y - int(np.rint(e.fit(align_x)))
        except AttributeError:
            print(f, 'removed (edge does not exist).')
            os.remove(os.path.join(edgepath, f))
            os.remove(os.path.join(croppath, f[:-4]+'.tiff'))
            bad_streak_filenames.append(f[:-4]+'.tiff')
            continue
        e.shift(yshift)
        all_edges[i] = e._y
        shifts[i] = yshift
        ax[0].plot(e.x, e.y, lw=0.1, c='r', alpha=0.5)

    all_edges = np.ma.masked_less(all_edges, 1)
    std = all_edges.std(axis=0)
    avg = all_edges.mean(axis=0)
    ax[0].plot(x, avg, c='k')
    ax[0].fill_between(x, avg - std, avg + std, facecolor='blue', alpha=0.2)
    ax2.plot(x, y_to_time(std), c='b', label='rough alignment')

    for i, e in enumerate(all_edges):
        res = minimize_scalar(error, args=(e, avg, WEIGHTS))
        all_edges[i] += res.x
        ax[1].plot(x, all_edges[i], c='r', lw=0.1, alpha=0.5)

    std2 = all_edges.std(axis=0)
    avg2 = all_edges.mean(axis=0)
    ax2.plot(x, y_to_time(std2), c='r', label='min error alignment')
    ax2.axhline(0.5, c='k', ls='--', lw=0.5)
    ax[1].plot(x, avg2, c='k')
    ax[1].fill_between(x, avg2 - std2, avg2 + std2, facecolor='blue', alpha=0.2)


    ax[2].fill_between(x, avg - std, avg + std, facecolor='blue', alpha=0.2)
    ax[2].fill_between(x, avg2 - std2, avg2 + std2, facecolor='red', alpha=0.2)

    ax[3].plot(x, WEIGHTS)
    ax[3].invert_yaxis()

    for a in ax:
        a.invert_yaxis()
        a.set(aspect='equal')

    ax[3].set(aspect='auto', xlim=(simple_xmin,simple_xmax))
    ax2.set(ylabel='std dev of onset time [ps]', xlim=(simple_xmin, simple_xmax), ylim=(0,1.5))
    ax2.legend()

    print('\n\nDeleting:', np.argwhere(shifts == -999).flatten())
    files = np.delete(files, np.argwhere(shifts == -999).flatten())
    shifts = np.delete(shifts, np.argwhere(shifts == -999).flatten())

    temp = os.path.join(miscpath,'avg_edge.npy')
    if verbose: print(f'\nSaving avg_edge.npy to {temp}')
    np.save(temp, avg2.data)

    temp = os.path.join(miscpath,'shifts.npy')
    if verbose: print(f'\nSaving shifts.npy to {temp}')
    np.save(temp, shifts)

    temp = os.path.join(miscpath,'shifts.csv')
    with open(temp, 'w') as f:
        for filename, shiftvalue in zip(files, shifts):
            f.write(f'{filename[:-4]}, {shiftvalue}\n')
    if verbose: print(f'Saving shifts.csv to {temp}')

    temp = os.path.join(miscpath, 'bad_streak_filenames.csv')
    if verbose: print(f'Saving bad_streak_filenames.csv to {temp}')
    with open(temp, 'w') as f:
        for i, filename in enumerate(bad_streak_filenames):
            f.write(f'{filename},\n')

    if save_fig:
        temp = os.path.join(miscpath,'edgealignment.png')
        if verbose: print(f'Saving figure to {temp}')
        fig.savefig(temp, transparent=False)
        temp = os.path.join(miscpath,'onsettimeerror.png')
        if verbose: print(f'Saving figure to {temp}')
        fig2.savefig(temp, transparent=False)

    if show_fig:
        fig.show()
        fig2.show()
        input()
    else:
        plt.close(fig)
        plt.close(fig2)

    print()
    return shifts

def sum_streaks(crop_bounds:tuple|list, medfilt_kernel_size:int|tuple|list, align_y:int,
                streak_height:int, image_height:int, simple_xmin:int, simple_xmax:int,
                miscpath:str, croppath:str, dewarppath:str,
                balance:float=0.5, save_streak_images:bool=False, show_fig:bool=True, save_fig:bool=True, verbose:bool=False):
    
    def get_vertical_max_and_width(vertslice, rel_height, window_length=7, poly_order=3):
        mean = vertslice.mean(axis=0)
        
        _, mean = filters.savgol1d(np.arange(len(mean)), mean, window_length, poly_order)
        ymax = np.argmax(mean)
        
        width, height, left, right = peak_widths(mean, [np.argmax(mean)], rel_height=rel_height)
        return ymax, width[0]
    
    files = os.listdir(croppath)
    try:
        files.remove('.DS_Store')
    except ValueError:
        pass
    files.sort()

    NUM = len(files)    

    if balance < 0 or balance > 1:
        raise ValueError(f'argument balance must be between 0 and 1. balance={balance}')

    print('Using %d of %d total images\n'%(NUM, len(files)))

    """ try:
        existing_bad_streak_filenames = list(np.genfromtxt(os.path.join(miscpath, 'bad_streak_filenames.csv'),
                                                delimiter=',', dtype=str, usecols=0))
        writeformat = 'a'
    except FileNotFoundError:
        existing_bad_streak_filenames = []
        writeformat = 'w' """

    shifts = np.load(os.path.join(miscpath, 'shifts.npy'))
    assert len(shifts) == len(files)

    shiftscsvpath = os.path.join(miscpath,'shifts.csv')
    with open(shiftscsvpath, 'w') as f:
        f.write('filename, shift, good shot\n')

    temp = rawccd.Trex(os.path.join(croppath, files[0]), verbose=False)
    streaks = np.zeros((NUM, streak_height, temp.width), dtype=np.uint32)
    #valid_streak_filenames = []
    #bad_streak_filenames = []
    indices_to_delete = []
    for i, f in enumerate(files):
        if i >= NUM: break
        if i % 10 == 0: print('|', end='')
        if i % 50 == 0: print()
        if shifts[i] > align_y - streak_height*(1-balance) or shifts[i] < streak_height*(balance) + align_y - image_height: # if the streak is too close to the top or bottom, exclude it.
            if verbose: print(f'OOB: {f}\tshift={shifts[i]}')
            else: print(',', end='', flush=True)
            indices_to_delete.append(i)
            #if f not in existing_bad_streak_filenames:
            #    bad_streak_filenames.append(f)
            with open(shiftscsvpath, 'a') as openfile:
                openfile.write(f'{f[:-13]}.tiff, {shifts[i]}, False\n')
            continue
        else:
            with open(shiftscsvpath, 'a') as openfile:
                openfile.write(f'{f[:-13]}.tiff, {shifts[i]}, True\n')

        if verbose: print('%d/%d:' % (i+1, NUM), end=' ')
        else: print('.', end='', flush=True)

        ccd = rawccd.Trex(os.path.join(croppath, f), verbose=verbose)
        ccd.rotate(180, True, True)
        ccd.shift_image((shifts[i], 0), raw=True)
        ccd.filter_image(filters.median_filter, medfilt_kernel_size)
        streaks[i] = ccd.rect((0, int(align_y - streak_height*(1-balance)), ccd.width, int(align_y + streak_height*(balance))), raw=False)
        #valid_streak_filenames.append(f)
        if save_streak_images:
            im = Image.fromarray(streaks[i])
            im.save(f'{dewarppath}/{f[:-13]}_straightened.tiff')

    streaks = np.delete(streaks, indices_to_delete, axis=0)
    print(f'\n{len(streaks)}/{NUM}')

    summed = streaks.mean(axis=0)
    x = np.arange(0, temp.width)
    max_intensity_by_column = edge.Edge(x, np.argmax(summed, axis=0), 5,
                                        simple_ythresh=150,
                                        simple_xthresh = 500,
                                        simple_xmin=simple_xmin,
                                        simple_xmax=simple_xmax)
    max_intensity_by_column.fine_pruning(20,50)

    temp = os.path.join(miscpath, 'streaks.npy')
    if verbose: print(f'\nSaving streaks.npy to {temp}')
    np.save(temp, streaks)

    temp = os.path.join(miscpath, 'summed.npy')
    if verbose: print(f'Saving summed.npy to {temp}')
    np.save(temp, summed)

    temp = os.path.join(miscpath, 'max_intensity_by_column.csv')
    if verbose: print(f'Saving max_intensity_by_column.csv to {temp}')
    max_intensity_by_column.save(temp)

    """ temp = os.path.join(miscpath, 'valid_streak_filenames.csv')
    if verbose: print(f'Saving valid_streak_filenames.csv to {temp}')
    with open(temp, 'w') as f:
        for i, filename in enumerate(valid_streak_filenames):
            f.write(f'{filename},\n')


    temp = os.path.join(miscpath, 'bad_streak_filenames.csv')
    if verbose: print(f'Updating bad_streak_filenames.csv at {temp}')
    with open(temp, writeformat) as f:
        for i, filename in enumerate(bad_streak_filenames):
            f.write(f'{filename},\n') """

    if save_fig or show_fig:
        fig, ax = plt.subplots(figsize=(6,2), dpi=300, constrained_layout=True)
        ax.imshow(summed, cmap='gist_stern')
        ax.set(aspect='equal', title='Sum of %d streaks' % len(streaks))
        ax.plot(max_intensity_by_column.x, max_intensity_by_column.y, c='g', lw=0.3)
        ax.plot(max_intensity_by_column.x, max_intensity_by_column.fit(recalc=True), c='lime', lw=0.3)
        if save_fig:
            temp = os.path.join(miscpath, 'summed.png')
            fig.savefig(temp, transparent=False)
        if show_fig:
            fig.show()
            input()
        else:
            plt.close(fig)

def isolate_strip(warpline, arr, above, below) -> np.ma.MaskedArray:
    '''Isolates a strip of an image above and below a given line.'''
    i = 0
    if arr.ndim == 3:
        i += 1
    ymesh, xmesh = np.mgrid[0:arr.shape[i], 0:arr.shape[i + 1]]
    mask = (ymesh >= warpline - above) & (ymesh < warpline + below)
    mask = np.ma.make_mask(mask)
    mask = np.invert(mask)
    if arr.ndim == 3:
        mask = np.broadcast_to(mask, arr.shape)
    mask = np.ma.MaskedArray(arr, mask)
    mask = np.ma.masked_equal(mask, 0)
    return mask

def straighten_stack(strip, strip_height) -> np.ma.MaskedArray:
    out = np.zeros((strip.shape[2], strip.shape[0], strip_height))

    index_of_first_not_masked_streak = 0
    for i, streak in enumerate(strip):
        if not np.all(streak.mask):
            index_of_first_not_masked_streak = i
            break
    for i, col in enumerate(np.moveaxis(strip, -1, 0)):
        goodpart = col[:,~col[index_of_first_not_masked_streak].mask]
        tempcol = np.zeros((strip.shape[0], strip_height))
        if goodpart.shape[1] == 0:
            pass
        elif goodpart.shape[1] < strip_height:
            if i < strip.shape[2]/2: # first half
                tempcol[:, :goodpart.shape[1]] = goodpart
                #continue
            if i >= strip.shape[2]/2: # second half
                tempcol[:, -goodpart.shape[1]:] = goodpart
        else:
            tempcol = goodpart
        out[i] = tempcol

    out = np.ma.masked_equal(out, 0)
    out = np.moveaxis(out, 0, -1)
    return out

def straighten_strip(strip, strip_width) -> np.ma.MaskedArray:
    out = np.zeros((strip.shape[1], strip_width))
    for i, col in enumerate(np.moveaxis(strip, -1, 0)):
        goodpart = col[~col.mask]
        tempcol = np.zeros(strip_width)
        if goodpart.shape[0] == 0:
            pass
        elif goodpart.shape[0] < strip_width:
            if i < strip.shape[1]/2: # first half
                tempcol[:goodpart.shape[0]] = goodpart
                #continue
            if i >= strip.shape[1]/2: # second half
                tempcol[-goodpart.shape[0]:] = goodpart
        else:
            tempcol = goodpart
        out[i] = tempcol

    out = np.ma.masked_less(out, 0)
    out = np.moveaxis(out, 0, -1)
    return out

def lineout(arr, axis=0, bounds=None):
    if bounds is None:
        bounds = (0, 0, arr.shape[1], arr.shape[0])
    return arr[bounds[1]:bounds[3], bounds[0]:bounds[2]].sum(axis=axis)

def dewarp_streaks_e(strip_height:int, y_offset:int,
                   warpcurvepath:PathLike, dewarppath:PathLike, miscpath:PathLike,
                   show_fig:bool=False, save_fig:bool=False, verbose:bool=False):

    # EDGE DETECTION VERSION
    streaks = np.load(f'{miscpath}/streaks.npy')
    N, Y, X = streaks.shape
    x = np.arange(X)
    
    if warpcurvepath[-4:] == '.npy':
        warp_px = np.load(warpcurvepath)
    elif warpcurvepath[-4:] == '.csv':
        warpx, warpy = np.genfromtxt(warpcurvepath, delimiter=',', unpack=True)
        warp_px = np.column_stack((warpx, warpy))
    else: raise ValueError(f'argument warpcurvepath should be a path to a .npy or .csv file.\nwarpcurvepath={warpcurvepath}')

    warp_interp = CubicSpline(warp_px[:,0], warp_px[:,1])

    strip = isolate_strip(warp_interp(x), streaks, strip_height-y_offset, y_offset)
    straight = straighten_stack(strip, strip_height)
    if verbose: print(f'Saving streaks_straightened.npy to {miscpath}/streaks_straightened.npy\n')
    np.save(f'{miscpath}/streaks_straightened.npy', np.ma.filled(straight, -999))

    valid_streak_filenames = np.genfromtxt(f'{miscpath}/valid_streak_filenames.csv',
                                           delimiter=',', dtype=str, usecols=0)
    for i, (streak, filename, stripe) in enumerate(zip(straight, valid_streak_filenames, strip)):
        temp = np.ma.filled(streak, 0)
        im = Image.fromarray(temp.astype(np.uint16))
        im.save(f'{dewarppath}/{filename}')

        """ temp = np.ma.filled(stripe, 480)
        im = Image.fromarray(temp.astype(np.uint16))
        im.save(f'data/streak/6-01/Layer_250nmTi_1umAl_experiment/streaks/{filename}') """

        if verbose: print(f'{i+1}/{N}: Saved {dewarppath}/{filename}')
        else:
            if i % 10 == 0: print('|', end='')
            if i % 50 == 0: print()
            print('.', end='', flush=True)

    if save_fig or show_fig:
        fig1, (ax1,ax2) = plt.subplots(2,1, figsize=(4,3), dpi=300, constrained_layout=True)
        ax1.plot(warp_px[:,0], warp_px[:,1], c='k', lw=1)
        ax1.plot(x, warp_interp(x), c='k', lw=0.5, ls='--', label='warp curve')
        ax1.plot(x, warp_interp(x) + y_offset, c='k', lw=0.5, ls='--', label='+ y_offset')
        ax1.plot(x, warp_interp(x) + y_offset - strip_height, c='k', lw=0.5, ls='--', label='+ y_offset - strip_height')
        ax1.imshow(strip.mean(axis=0), cmap='gist_stern')
        ax1.set(xlabel='x [px]', ylabel='y [px]')

        ax2.imshow(straight.mean(axis=0), cmap='gist_stern')
        ax2.set(xlabel='x [px]', ylabel='y [px]')
        if save_fig:
            fig1.savefig(f'{miscpath}/strip.png', transparent=False)
        if show_fig:
            fig1.show()
            input()
        else:
            plt.close(fig1)

def dewarp_streaks_s(strip_height:int, y_offset:int,
                    crop_bounds:tuple|list, medfilt_kernel_size:int|tuple,
                    fullpath:PathLike, warpcurvepath:PathLike, croppath:PathLike, miscpath:PathLike,
                    verbose:bool=False):
    # first read in full images, calculate dataset background, crop and subtract background
    # then straighten due to warp curve. then save to cropped directory
    # also while the file is open, calculate horizontal lineout and save to numpy array,
    # at the end save this numpy array to the misc direcgory 

    files = os.listdir(fullpath)
    try:
        files.remove('.DS_Store')
    except ValueError:
        pass
    files.sort()
    N = len(files)

    print('%d total images'%N)

    # generate background
    background = generate_background(files, fullpath)
    background = (np.clip(background, 0, None)).astype(np.uint16)

    warp_px = np.load(warpcurvepath)
    warp_interp = CubicSpline(warp_px[:,0], warp_px[:,1])

    ylineouts = np.ones((N, strip_height)) * -999

    for i, f in enumerate(files):
        if verbose: print('%d/%d: ' % (i+1, len(files)), end='')
        else:
            print('.', end='', flush=True)
            if i % 10 == 0: print('|', end='')
            if i % 50 == 0: print()

        ccd = rawccd.Trex(os.path.join(fullpath, f), verbose=verbose)
        ccd.subtract_background(background, raw=True)
        ccd.crop(crop_bounds, raw=True)
        ccd.filter_image(filters.median_filter, medfilt_kernel_size)

        x = np.arange(ccd.width)
        mask = isolate_strip(warp_interp(x), ccd.data, y_offset, strip_height - y_offset)
        straight = straighten_strip(mask, strip_height)
        im = Image.fromarray(straight)
        if verbose: print(f'Saved cropped image to {os.path.join(croppath, f)}')
        im.save(os.path.join(croppath, f'{f[:-5]}_cropped.tiff'))

        ylineout = lineout(straight, axis=1, bounds=(400, 0, 700, strip_height))
        ylineout = ylineout - np.median(ylineout) # crude way of removing the floor. a better method would fit the floor to a line and subtract it here
        ylineouts[i] = ylineout

    np.save(f'{miscpath}/ylineouts.npy', ylineouts) # might throw a not implemented error here if ylineouts is masked?

def align_streaks_s(align_y:int, intensity_threshold:float,
                  miscpath:PathLike):
    # load ylineouts, calculate ymax on each one and yonset, calculate shifts array
    # build valid_filenames list and bad_filesnames list
    # save shifts array
    
    ylineouts = np.load(f'{miscpath}/ylineouts.npy')
    ylineouts_filtered = savgol_filter(ylineouts, 11, 3, axis=1)
    ymaxes = np.argmax(ylineouts_filtered,  axis=1)
    ymaxes_flattened = np.array([i*ylineouts.shape[1] + index for i,index in zip(range(len(ylineouts)), ymaxes)])

    print(ylineouts_filtered.flatten()[ymaxes_flattened])

    shifts= align_y - ymaxes
    shifts[ylineouts_filtered.flatten()[ymaxes_flattened] < intensity_threshold] = -999 # somehow eliminate shots where there is no streak (ymax is not prominent above the noise)
    np.save(f'{miscpath}/shifts.npy', shifts)
