# Contributing

Contributions to `unumpy` are welcome and appreciated. Contributions can take the form of bug reports, documentation, code, and more.

## Getting the code

Make a fork of the main [unumpy repository](https://github.com/Quansight-Labs/unumpy) and clone the fork:

```
git clone https://github.com/<your-github-username>/unumpy
```

## Install

Note that unumpy supports Python versions >= 3.5. If you're running `conda` and would prefer to have dependencies
pulled from there, use

```
conda env create -f .conda/environment.yml
conda activate uarray
```

`unumpy` and all development dependencies can be installed via:

```
pip install -e ".[all]"
```


This will create an environment named `uarray` which you can use for development.

## Testing

Tests can be run from the main uarray directory as follows:

```
pytest
```

To run a subset of tests:

```
pytest unumpy/tests/test_numpy.py
```
