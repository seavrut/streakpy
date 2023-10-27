"""processing.py
===
Main script of this package. Process a dataset of spectroscopic streak camera CCD outputs.
Contains both processing.calculate_edges() and processing.align_edges(), which can be imported
and ran in another Python program, or this program works as a command line script with usage
detailed below.

usage: python processing.py [-h] [-V] [-v] [-c] [-a] [-s] input

positional arguments:
  input                 filepath to .ini file for configuration of inputs

options:
  -h, --help            show this help message and exit
  -V, --version         show program's version number and exit
  -v, --verbosee        print all output
  -c, --calculate_edges run calculate_edges
  -a, --align_edges     run align_edges
  -s, --sum_streaks     run sum_streaks
"""

__version__ = '2.1.0'
__author__ = 'Sofia Avrutsky'

import numpy as np
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from configparser import ConfigParser, ExtendedInterpolation
from PIL import Image
from scipy.optimize import minimize_scalar
from typing import Union as U, Optional as O

import rawccd, filters, edge

def calculate_edges(high_percentile_thresh:int, low_percentile_thresh:int,
        size_thresh:int, medfilt_kernel_size:int|tuple|list,
        degree:int, bkgrnd_window_radius:int,
        fine_pruning_thresh:int|float, fine_pruning_radius:int,
        simple_xmin:int|float, simple_xmax:int|float, crop_bounds:tuple|list,
        fullpath:str, edgepath:str, croppath:str, verbose:bool=False):

    files = os.listdir(fullpath)
    try:
        files.remove('.DS_Store')
    except ValueError:
        pass
    files.sort()

    print('%d total images'%len(files))

    for i, f in enumerate(files):
        if verbose: print('%d/%d: ' % (i+1, len(files)), end='')
        else:
            print('.', end='', flush=True)
            if i % 10 == 0: print('|', end='')
            if i % 50 == 0: print()

        ccd = rawccd.Trex(os.path.join(fullpath, f), verbose=verbose)
        top_strip = ccd.rect((0, 0, ccd.width, 100))
        row_bkgrnd_averages = np.zeros(ccd.width)
        for i, col in enumerate(top_strip.T):
            if i < bkgrnd_window_radius:
                thick_column = top_strip.T[:i+bkgrnd_window_radius]
            elif i > ccd.width - bkgrnd_window_radius:
                thick_column = top_strip.T[i-bkgrnd_window_radius:]
            else:
                thick_column = top_strip.T[i-bkgrnd_window_radius:i + bkgrnd_window_radius]
            row_bkgrnd_averages[i] = thick_column.mean()
        background = np.ones(ccd.shape) * row_bkgrnd_averages - 500
        background = (np.clip(background, 0, None)).astype(np.uint16)
        ccd.subtract_background(background, raw=True)
        ccd.crop(crop_bounds, raw=True)
        im = Image.fromarray(ccd.unfiltered_data)
        im.save(os.path.join(croppath, f))
        ccd.filter_image(filters.median_filter, medfilt_kernel_size)
        try:
            edge = ccd.detect_bottom_edge(high_percentile_thresh, low_percentile_thresh, size_thresh, degree=degree,
                                        simple_xmin=simple_xmin, simple_xmax=simple_xmax,
                                        fine_pruning_thresh=fine_pruning_thresh, fine_pruding_radius=fine_pruning_radius)
            name = f[:-5] + '.csv'
            edge.save(os.path.join(edgepath, f[:-5] + '.csv'), verbose)
            """ except AttributeError:
            if verbose: print('Skipping..\n') """
        except RuntimeError:
            if verbose: print('Skipping...\n')
    print()

def align_edges(align_x:int, degree:int,
                simple_xmin:int, simple_xmax:int,
                simple_xthresh:int|float, simple_ythresh:int|float,
                weights_params:tuple|list,
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
        return y*20*5.3/1000.0
    
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

    all_edges = np.empty((len(files), len(y)))
    shifts = np.zeros(len(files), dtype=int)

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
            yshift = 500 - int(np.rint(e.fit(align_x)))
        except AttributeError:
            print(f, 'removed')
            os.remove(os.path.join(edgepath, f))
            os.remove(os.path.join(croppath, f[:-4]+'.tiff'))
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

    if verbose: print('\n\nDeleting:', np.argwhere(shifts == 0).flatten())
    files = np.delete(files, np.argwhere(shifts == 0).flatten())
    shifts = np.delete(shifts, np.argwhere(shifts == 0).flatten())

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

    if save_fig:
        temp = os.path.join(miscpath,'edgealignment.png')
        if verbose: print(f'Saving figure to {temp}')
        fig.savefig(temp, transparent=False)
        temp = os.path.join(miscpath,'onsettimeerror.png')
        if verbose: print(f'Saving figure to {temp}')
        fig2.savefig(temp, transparent=False)

    if show_fig:
        plt.show()

    print()
    return shifts

def sum_streaks(crop_bounds:tuple|list, medfilt_kernel_size:int|tuple|list, shift_thresh:int,
                streak_height:int, 
                edgepath:str, miscpath:str, croppath:str,
                num:O[int]=None, show_fig:bool=True, save_fig:bool=True, verbose:bool=False):
    files = os.listdir(edgepath)
    try:
        files.remove('.DS_Store')
    except ValueError:
        pass
    files.sort()

    if num is None: NUM = len(files)
    else: NUM = num

    print('Using %d of %d total images\n'%(NUM, len(files)))

    shifts = np.load(os.path.join(miscpath, 'shifts.npy'))
    assert len(shifts) == len(files)
    temp = rawccd.Trex(os.path.join(croppath, files[0][:-4]+'.tiff'), verbose=False)
    streaks = np.zeros((NUM, streak_height, temp.width), dtype=np.uint32)
    #lineouts = np.zeros((len(files), crop_bounds[2]-crop_bounds[0]))
    num_used = 0
    for i, f in enumerate(files):
        if i >= NUM: break
        if shifts[i] > streak_height/2 or shifts[i] < -streak_height/2: # if the streak is too close to the top or bottom, exclude it.
            print(',', end='', flush=True)
            continue
        num_used += 1
        if verbose: print('%d/%d:' % (i+1, NUM), end=' ')
        else:
            print('.', end='', flush=True)
            if num_used % 10 == 0: print('|', end='')
            if num_used % 50 == 0: print()

        ccd = rawccd.Trex(os.path.join(croppath, f[:-4]+'.tiff'), verbose=verbose)
        ccd.rotate(180, True, True)
        ccd.shift_image((shifts[i], 0), raw=True)
        ccd.filter_image(filters.median_filter, medfilt_kernel_size)
        #st = ccd.define_streak((0, shift_thresh, ccd.width, shift_thresh + streak_height), raw=False)
        #st.straighten()
        #lineouts[i] = st.lineout
        streaks[i] = ccd.rect((0, int(500 - streak_height/2), ccd.width, int(500 + streak_height/2)), raw=False)

    summed = streaks.mean(axis=0)
    x = np.arange(0, temp.width)
    max_intensity_by_column = edge.Edge(x, np.argmax(summed, axis=0), 5, 100, 400, 300, 900)
    max_intensity_by_column.fine_pruning(20,50)
    """ lineouts = np.ma.masked_less(lineouts, 1)
    avg = lineouts.mean(axis=0)
    std = lineouts.std(axis=0) """

    temp = os.path.join(miscpath, 'streaks.npy')
    if verbose: print(f'\nSaving streaks.npy to {temp}')
    np.save(temp, streaks)

    temp = os.path.join(miscpath, 'summed.npy')
    if verbose: print(f'Saving summed.npy to {temp}')
    np.save(temp, summed)

    temp = os.path.join(miscpath, 'max_intensity_by_column.csv')
    if verbose: print(f'Saving max_intensity_by_column.csv to {temp}')
    max_intensity_by_column.save(temp)

    if save_fig or show_fig:
        fig, ax = plt.subplots(figsize=(6,2), dpi=300, constrained_layout=True)
        ax.imshow(summed, cmap='gist_stern')
        ax.set(aspect='equal', title='Sum of %d streaks' % num_used)
        ax.plot(max_intensity_by_column.x, max_intensity_by_column.y, c='g', lw=0.3)
        ax.plot(max_intensity_by_column.x, max_intensity_by_column.fit(recalc=True), c='lime', lw=0.3)
        if save_fig:
            temp = os.path.join(miscpath, 'summed.png')
            fig.savefig(temp, transparent=False)
        if show_fig:
            plt.show()

if __name__ == '__main__':
    print()
    parser = ArgumentParser(description='Process a dataset of spectroscopic streak camera CCD outputs.')
    parser.add_argument('input',
                        help='filepath to .ini file for configuration of inputs')
    parser.add_argument('-V', '--version', action="version",
                        version = f"{parser.prog} version {__version__}")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help = 'print all output')
    parser.add_argument('-c', '--calculate_edges', action='store_true',
                        help = 'run calculate_edges')
    parser.add_argument('-a', '--align_edges', action='store_true',
                        help = 'run align_edges')
    parser.add_argument('-s', '--sum_streaks', action='store_true',
                        help = 'run sum_streaks')
    args = parser.parse_args()
    
    config = ConfigParser(interpolation=ExtendedInterpolation(),
                      converters={'tuple':eval}, allow_no_value=True)
    if len(config.read(args.input)) == 0:
        raise FileNotFoundError(f'File not founds {args.input}')

    path = config['paths']['full']
    if not os.path.isdir(path):
        raise FileNotFoundError(f'Directory of full CCD images not found at {path}')
    
    try:
        path = config['paths']['edge']
        os.mkdir(path)
        print(f'Creating edge directory at {path}')
    except FileExistsError:
        pass

    try:
        path = config['paths']['crop']
        os.mkdir(path)
        print(f'Creating cropped directory at {path}')
    except FileExistsError:
        pass

    try:
        path = config['paths']['misc']
        os.mkdir(path)
        print(f'Creating misc directory at {path}')
    except FileExistsError:
        pass
        

    if args.calculate_edges:
        print('\nRunning calculate_edges\n')
        try:
            calculate_edges(
                high_percentile_thresh=config['edge.finding'].getint('high_percentile_thresh'),
                low_percentile_thresh=config['edge.finding'].getint('low_percentile_thresh'),
                size_thresh=config['edge.finding'].getint('size_thresh'),
                medfilt_kernel_size=config['edge.finding'].getint('medfilt_kernel_size'),
                degree=config['edge.finding'].getint('degree'),
                bkgrnd_window_radius=config['background.processing'].getint('average_window_radius'),
                fine_pruning_radius=config['edge.finding'].getint('fine_pruning_radius'),
                fine_pruning_thresh=config['edge.finding'].getint('fine_pruning_thresh'),
                simple_xmin=config['edge.finding'].getint('simple_xmin'),
                simple_xmax=config['edge.finding'].getint('simple_xmax'),
                crop_bounds=config['background.processing'].gettuple('crop_bounds'),
                fullpath=config['paths']['full'],
                edgepath=config['paths']['edge'],
                croppath=config['paths']['crop'],
                verbose=args.verbose
            )

        except KeyError as e:
            print(f'KeyError encountered in config file {args.input} while attempting to access key {e}')

    if args.align_edges:
        print('\nRunning align_edges\n')
        try:
            weight_params = [config['edge.aligning.weightparameters'].getfloat('mu1'),
                             config['edge.aligning.weightparameters'].getfloat('sigma1'),
                             config['edge.aligning.weightparameters']['mu2'],
                             config['edge.aligning.weightparameters']['sigma2'],
                             config['edge.aligning.weightparameters']['amp2']]
            if weight_params[2] is None and weight_params[3] is None and weight_params[4] is None:
                weight_params = weight_params[:2]
            else:
                weight_params[2] = float(weight_params[2])
                weight_params[3] = float(weight_params[3])
                weight_params[4] = float(weight_params[4])
            
            align_edges(align_x=config['edge.aligning'].getint('align_x'),
                        degree=config['edge.finding'].getint('degree'),
                        simple_xmin=config['edge.finding'].getint('simple_xmin'),
                        simple_xmax=config['edge.finding'].getint('simple_xmax'),
                        simple_xthresh=config['edge.aligning'].getfloat('simple_xthresh'),
                        simple_ythresh=config['edge.aligning'].getfloat('simple_ythresh'),
                        weights_params=weight_params,
                        edgepath=config['paths']['edge'],
                        croppath=config['paths']['crop'],
                        miscpath=config['paths']['misc'],
                        show_fig=config['edge.aligning'].getboolean('show_fig'),
                        save_fig=config['edge.aligning'].getboolean('save_fig'),
                        verbose=args.verbose
                        )
        except KeyError as e:
            print(f'KeyError encountered in config file {args.input} while attempting to access key {e}')

    if args.sum_streaks:
        print('\nRunning sum_streaks\n')
        try:
            sum_streaks(crop_bounds=config['background.processing'].gettuple('crop_bounds'),
                        medfilt_kernel_size=config['background.processing'].gettuple('medfilt_kernel_size'),
                        shift_thresh=config['streak.summing'].getint('shift_thresh'),
                        streak_height=config['streak.summing'].getint('streak_height'),
                        edgepath=config['paths']['edge'],
                        miscpath=config['paths']['misc'],
                        croppath=config['paths']['crop'],
                        num=config['streak.summing'].getint('num'),
                        show_fig=config['streak.summing'].getboolean('show_fig'),
                        save_fig=config['streak.summing'].getboolean('save_fig'),
                        verbose=args.verbose)
        except KeyError as e:
            print(f'KeyError encountered in config file {args.input} while attempting to access key {e}')