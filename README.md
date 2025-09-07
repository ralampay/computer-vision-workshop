# Computer Vision Workshop

## Setup

It would be advisable to create your own environment to isolate dependencies.

### Option 1: Using Python's built in `venv`

```bash
# Create a virtual environment named .venv
python3 -m venv .venv

# Activate the environment (Linux / macOS)
source .venv/bin/activate

# Activate the environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Check Python version
python --version
```

### Option 2: Using `pyenv`

Install `pyenv` first. See instructions [here](https://github.com/pyenv/pyenv).

```bash
# Install Python version (if not yet installed)
pyenv install 3.12

# Create a new virtual environment
pyenv virtualenv 3.12 cv-workshop

# Activate the environment
pyenv activate cv-workshop

# Verify Python version
python --version
pip install opencv-python torch ultralytics
```

### Install Dependencies from `requirements.txt`

```bash
pip install -r requirements.txt
```
