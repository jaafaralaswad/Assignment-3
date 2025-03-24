![Python Version](https://img.shields.io/badge/python-3.12-blue)
![OS](https://img.shields.io/badge/os-ubuntu%20%7C%20macos%20%7C%20windows-blue)
![License](https://img.shields.io/badge/license-MIT-green)


# ME700 Assignment 3

## Table of Contents

- [Introduction](#introduction)
- [Conda Environment, Installation, and Testing](#conda-environment-installation-and-testing)
- [The Finite Element Method](#the-finite-element-method)
- [Tutorials](#tutorials)
- [More Information](#more-information)

## Introduction
This repository presents the work developed to fulfill the requirements of Assignment 3 for the course ME700.


## Conda environment, install, and testing

This procedure is very similar to what we did in class. First, you need to download the repository and unzip it. Then, to install the package, use:

```bash
conda create --name assignment-3-env python=3.12
```

After creating the environment (it might have already been created by you earlier), make sure to activate it, use:

```bash
conda activate assignment-3-env
```

Check that you have Python 3.12 in the environment. To do so, use:

```bash
python --version
```

Create an editable install of the assignemnt codes. Use the following line making sure you are in the correct directory:

```bash
pip install -e .
```

You must do this in the correct directory; in order to make sure, replace the dot at the end by the directory of the folder "Assignment-3-main" that you unzipped earlier: For example, on my computer, the line would appear as follows:

```bash
pip install -e /Users/jaafaralaswad/Downloads/Assignment-3-main
```

Now, you can test the code, make sure you are in the tests directory. You can know in which directory you are using:

```bash
pwd
```

Navigate to the tests folder using the command:

```bash
cd
```

On my computer, to be in the tests folder, I would use:

```bash
cd /Users/jaafaralaswad/Downloads/Assignment-3-main/tests
```


Once you are in the tests directory, use the following to run the tests:

```bash
pytest -s test_main.py
```

Code coverage should be 100%.

To run the tutorial, make sure you are in the tutorials directory. You can navigate their as you navigated to the tests folder. On my computer, I would use:

```bash
cd /Users/jaafaralaswad/Downloads/Assignment-3-main/tutorials
```

Once you are there, you can use:

```bash
pip install jupyter
```

Depending on which tutorial you want to use, you should run one of the following lines:


```bash
jupyter notebook brick_element.ipynb
```


A Jupyter notebook will pop up.



## The Finite Element Method
To be written


## Tutorials

To be written.

## More information

To be written
