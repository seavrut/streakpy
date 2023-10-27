# streakpy
A Python package for processing and analyzing TREX Streak camera data. 

Main functionality lies in processing.py. It works with a command line interface.

```
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
```

Other files are helper classes for storing and handling the data.
`example.ini` is a template for an input configuration file to run processing.py with.
Documentation for input config file parameters doesn't exist yet, but will at some point. At this point, safe to say that whatever is in `example.ini` is necessary to run `processing.py`.
