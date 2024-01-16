"""processing.py
===
Main script of this package. Process a dataset of spectroscopic streak camera CCD outputs.

usage: python processing.py [-h] [-V] [-v] [-r] [-c] [-a] [-s] [-d] input

positional arguments:
  input                 filepath to .ini file for configuration of inputs

options:
  -h, --help                show this help message and exit
  -V, --version             show program's version number and exit
  -v, --verbose             print all output
  -r, --remove_backgrounds  run remove_backgrounds
  -c, --calculate_edges     run calculate_edges
  -a, --align_streaks       run align_streaks
  -s, --sum_streaks         run sum_streaks
  -d, --dewarp_streaks      run dewarp_streaks
"""

__version__ = '2.3.0'
__author__ = 'Sofia Avrutsky'

import numpy as np
import os
from time import perf_counter
from argparse import ArgumentParser
from configparser import ConfigParser, ExtendedInterpolation, ParsingError

from processing_funcs import (
    remove_backgrounds,
    calculate_edges,
    align_streaks_e,
    align_streaks_s,
    sum_streaks,
    dewarp_streaks_e,
    dewarp_streaks_s
)

if __name__ == '__main__':
    print()
    parser = ArgumentParser(description='Process a dataset of spectroscopic streak camera CCD outputs.')
    parser.add_argument('input',
                        help='filepath to .ini file for configuration of inputs')
    parser.add_argument('-V', '--version', action="version",
                        version = f"{parser.prog} version {__version__}")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help = 'print all output')
    #parser.add_argument('-r', '--remove_backgrounds', action='store_true',
    #                    help = 'run remove_backgrounds')
    parser.add_argument('-c', '--calculate_edges', action='store_true',
                        help = 'run calculate_edges')
    parser.add_argument('-a', '--align_streaks', action='store_true',
                        help = 'run align_streaks')
    parser.add_argument('-s', '--sum_streaks', action='store_true',
                        help = 'run sum_streaks')
    parser.add_argument('-d', '--dewarp_streaks', action='store_true',
                        help = 'run dewarp_streaks')
    args = parser.parse_args()
    
    config = ConfigParser(interpolation=ExtendedInterpolation(),
                      converters={'tuple':eval}, allow_no_value=True)
    if len(config.read(args.input)) == 0:
        raise FileNotFoundError(f'File not founds {args.input}')

    USE_EDGE_DETECTION = config['algorithms'].getboolean('use_edge_detection')
    if USE_EDGE_DETECTION is None:
        USE_EDGE_DETECTION = False
    USE_STRAIGHTENED_PEAK_DETECTION = config['algorithms'].getboolean('use_straightened_peak_detection')
    if USE_STRAIGHTENED_PEAK_DETECTION is None:
        USE_STRAIGHTENED_PEAK_DETECTION = False 
    if np.all((USE_EDGE_DETECTION, USE_STRAIGHTENED_PEAK_DETECTION)) or np.all((not USE_EDGE_DETECTION, not USE_STRAIGHTENED_PEAK_DETECTION)):
        msg = 'Only one algorithm must be specified as True. ' + \
               'All other algorithms must be either absent from the config file '+\
               'or set to False.'
        raise ParsingError(msg)

    path = config['paths']['full']
    if not os.path.isdir(path):
        raise FileNotFoundError(f'Directory of full CCD images not found at {path}')
    
    if USE_EDGE_DETECTION:
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

    try:
        path = config['paths']['dewarp']
        os.mkdir(path)
        print(f'Creating dewarped directory at {path}')
    except FileExistsError:
        pass


    if USE_EDGE_DETECTION:
        print('\nUsing edge detection algorithm. Order of functions run will be',
              '1. calculate_edges 2. align_streaks 3. sum_streaks 4. dewarp_streaks')
        
        if args.calculate_edges:
            print('\nRunning calculate_edges\n')
            starttime = perf_counter()
            try:
                calculate_edges(
                    high_percentile_thresh=config['edge.finding'].getfloat('high_percentile_thresh'),
                    low_percentile_thresh=config['edge.finding'].getfloat('low_percentile_thresh'),
                    size_thresh=config['edge.finding'].getint('size_thresh'),
                    medfilt_kernel_size=config['edge.finding'].getint('medfilt_kernel_size'),
                    degree=config['edge.finding'].getint('degree'),
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
            endtime = perf_counter()
            print('%.3f s' % (endtime - starttime))

        if args.align_streaks:
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
                
                px_to_ps = float(config['axes.conversions'].get('px_to_ps', '0.14'))

                align_streaks_e(align_x=config['edge.aligning'].getint('align_x'),
                            align_y=config['edge.aligning'].getint('align_y'),
                            degree=config['edge.finding'].getint('degree'),
                            simple_xmin=config['edge.finding'].getint('simple_xmin'),
                            simple_xmax=config['edge.finding'].getint('simple_xmax'),
                            simple_xthresh=config['edge.aligning'].getfloat('simple_xthresh'),
                            simple_ythresh=config['edge.aligning'].getfloat('simple_ythresh'),
                            weights_params=weight_params,
                            px_to_ps=px_to_ps,
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
                num = config['streak.summing']['num']
                if num is not None: num = int(num)
                try: balance = float(config.get('streak.summing', 'balance', fallback='0.5'))
                except TypeError: balance = 0.5

                sum_streaks(crop_bounds=config['background.processing'].gettuple('crop_bounds'),
                            medfilt_kernel_size=config['background.processing'].gettuple('medfilt_kernel_size'),
                            align_y=config['edge.aligning'].getint('align_y'),
                            streak_height=config['streak.summing'].getint('streak_height'),
                            simple_xmin=config['streak.summing'].getint('simple_xmin'),
                            simple_xmax=config['streak.summing'].getint('simple_xmax'),
                            miscpath=config['paths']['misc'],
                            croppath=config['paths']['crop'],
                            num=num,
                            balance=balance,
                            show_fig=config['streak.summing'].getboolean('show_fig'),
                            save_fig=config['streak.summing'].getboolean('save_fig'),
                            verbose=args.verbose)
            except KeyError as e:
                print(f'KeyError encountered in config file {args.input} while attempting to access key {e}')

        if args.dewarp_streaks:
            print('\nRunning dewarp_streaks\n')
            try:
                try: y_offset = int(config.get('streak.dewarping', 'y_offset', fallback='0'))
                except TypeError: y_offset = 0
                dewarp_streaks_e(
                            strip_height = config['streak.dewarping'].getint('strip_height'),
                            y_offset=y_offset,
                            warpcurvepath=config['paths']['warp'],
                            dewarppath=config['paths']['dewarp'],
                            miscpath=config['paths']['misc'],
                            show_fig=config['streak.dewarping'].getboolean('show_fig'),
                            save_fig=config['streak.dewarping'].getboolean('save_fig'),
                            verbose=args.verbose
                            )
            except KeyError as e:
                print(f'KeyError encountered in config file {args.input} while attempting to access key {e}')

    elif USE_STRAIGHTENED_PEAK_DETECTION:
        print('\nUsing straightened peak detection algorithm. Order of functions run will be',
              '1. dewarp_streaks 2. align_streaks 3. sum_streaks')
        
        if args.dewarp_streaks:
            print('\nRunning dewarp_streaks\n')
            try:
                try: y_offset = int(config.get('streak.dewarping', 'y_offset', fallback='0'))
                except TypeError: y_offset = 0
                dewarp_streaks_s(
                            strip_height = config['streak.dewarping'].getint('strip_height'),
                            y_offset=y_offset,
                            crop_bounds=config['background.processing'].gettuple('crop_bounds'),
                            medfilt_kernel_size=config['background.processing'].gettuple('medfilt_kernel_size'),
                            fullpath=config['paths']['full'],
                            warpcurvepath=config['paths']['warp'],
                            croppath=config['paths']['crop'],
                            miscpath=config['paths']['misc'],
                            verbose=args.verbose
                        )
            
            except KeyError as e:
                print(f'KeyError encountered in config file {args.input} while attempting to access key {e}')

        if args.align_streaks:
            print('\nRunning align_streaks\n')
            try:
                align_streaks_s(
                    align_y = config['edge.aligning'].getint('align_y'),
                    intensity_threshold = config['edge.aligning'].getfloat('intensity_threshold'),
                    miscpath= config['paths']['misc']
                )
            except KeyError as e:
                print(f'KeyError encountered in config file {args.input} while attempting to access key {e}')

        if args.sum_streaks:
            print('\nRunning sum_streaks\n')
            try:
                try: balance = float(config.get('streak.summing', 'balance', fallback='0.5'))
                except TypeError: balance = 0.5

                sum_streaks(crop_bounds=config['background.processing'].gettuple('crop_bounds'),
                            medfilt_kernel_size=config['background.processing'].gettuple('medfilt_kernel_size'),
                            align_y=config['edge.aligning'].getint('align_y'),
                            streak_height=config['streak.summing'].getint('streak_height'),
                            image_height=config['streak.dewarping'].getint('strip_height'),
                            simple_xmin=config['streak.summing'].getint('simple_xmin'),
                            simple_xmax=config['streak.summing'].getint('simple_xmax'),
                            miscpath=config['paths']['misc'],
                            croppath=config['paths']['crop'],
                            dewarppath=config['paths']['dewarp'],
                            balance=balance,
                            save_streak_images=True,
                            show_fig=config['streak.summing'].getboolean('show_fig'),
                            save_fig=config['streak.summing'].getboolean('save_fig'),
                            verbose=args.verbose)
            except KeyError as e:
                print(f'KeyError encountered in config file {args.input} while attempting to access key {e}')

        if args.calculate_edges:
            raise RuntimeWarning('use_edge_detection was not specified. calculate_edges will be skipped.')