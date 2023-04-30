# Overview of methods

+ Network Generation
+ Networked Spread
+ Differential Equation Models
+ Nonparametric R0 Estimation
+ Data connector to CovidCast

# Installation

Install Python 3, and create a virtual environment:

```bash
python -m pip install virtualenv
python -m virtualenv <VENV_PATH>
```

Choose a `VENV_PATH` outside of the git repo, so we don't end up pushing all that... Now activate the virtual environment and install the requirements. This varies by operating system, but on Linux / Mac:

```bash
source <VENV_PATH>/bin/activate
pip install <GIT_REPO>/requirements.txt
```

In your working directory (outside the repo), create a `.config.yaml` file with the appropriate values (see `.config.example.yaml`). You should be ready to develop!

# Contact

This was initiated by Alec McGail, am2873@cornell.edu
Good luck, have fun!