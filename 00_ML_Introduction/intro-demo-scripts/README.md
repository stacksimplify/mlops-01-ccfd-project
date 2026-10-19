# Section 00 Intro Demo Scripts

One script makes the Section 00 intro demo ready: it trains the **baseline model** and the **tuned model**, starts **both FastAPI apps**, and opens them **side by side** in your browser.

| File | What it is |
|---|---|
| `run_intro_demo.sh` | The one script to run (nine stages, below) |
| `side-by-side.html` | The page that shows both APIs next to each other, with the Net Benefit of each model (before and after tuning) at the bottom |
| `ccfd-project/` | An unchanged copy of the Section 08 project (the script only runs it) |
| `p3_02_inference_api_baseline_v2_demo.py` | Demo version of the baseline API: runs the unchanged course API and adds one button to its web page |
| `p5_04_inference_api_v2_demo.py` | Demo version of the tuned API: the same idea |
| `demo_ui.py` | Adds the purple **Real Fraud from Test Data (row 7382)** button to the course web page |

## Before you run it (quick setup, one time)

These are quick reference steps. The full, step-by-step setup is in [Section 02: Setup Python Environment](../../02_Setup_Python/README.md).

**On Windows:** skip steps 1 and 2: install Miniconda as in [Section 02, Step-01-02](../../02_Setup_Python/README.md#step-01-02-windows), open **Anaconda PowerShell Prompt**, and continue with step 3.

**1. Is Miniconda already installed? Check its folder first.** `conda --version` can fail even when Miniconda is installed (step 2 fixes that), so look for the folder:

```bash
# Look for an existing Miniconda folder (no output means none is installed)
ls -d ~/miniconda3 /opt/homebrew/Caskroom/miniconda/base 2>/dev/null
```

If a folder is listed, Miniconda is installed: **do not install it again**, go to step 2. Only if nothing is listed, install it: open https://www.anaconda.com/docs/getting-started/installation and choose Miniconda for your system, or on macOS with Homebrew:

```bash
# Install Miniconda with Homebrew (macOS)
brew install --cask miniconda
```

Full steps: [Section 02, Step-01: Install Miniconda](../../02_Setup_Python/README.md#step-01-install-miniconda). Two Minicondas on one machine (each with its own `mlops-env1`) cause confusing errors, so keep only one.

**2. Seeing `zsh: command not found: conda` or `Run 'conda init' before 'conda activate'`?** Miniconda is installed, but your zsh terminal is not set up for it yet. Run `conda init zsh` once with the full path of the Miniconda from step 1 (the one whose `envs/` folder has your `mlops-env1`):

```bash
# Set up conda for zsh: Miniconda installed with the installer from the website
~/miniconda3/bin/conda init zsh

# Set up conda for zsh: Miniconda installed with Homebrew (Apple Silicon Mac; on an Intel Mac use /usr/local/Caskroom/...)
/opt/homebrew/Caskroom/miniconda/base/bin/conda init zsh
```

Then load the new setting in the terminal you have open (or close it and open a new one):

```bash
# Load the new conda setting in this terminal
source ~/.zshrc
```

Check it worked:

```bash
# Activate the course environment
conda activate mlops-env1

# Check that the course libraries are installed
python -c "import fastapi, sklearn; print('course env OK')"
```

Two things that do not work: plain `conda init` (without `zsh`) only sets up the bash shell, and a terminal that was already open keeps the old setting until you run `source ~/.zshrc`. Use `bash` instead of `zsh` (and `~/.bash_profile`) if `echo $SHELL` shows `/bin/bash`. On Windows, use the **Anaconda PowerShell Prompt** that Miniconda installs. More help: [Section 02, Troubleshooting](../../02_Setup_Python/README.md#troubleshooting).

**3. Create the course environment (skip if `conda env list` already shows `mlops-env1`).** On a fresh Miniconda install, accept the Terms of Service first, once:

```bash
# Accept the Terms of Service (main channel)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main

# Accept the Terms of Service (r channel)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

Then create the environment, go to this folder, and install the course libraries:

```bash
# Create the course environment with Python 3.14
conda create -n mlops-env1 python=3.14 -c conda-forge -y

# Activate the course environment
conda activate mlops-env1

# Go to the intro demo folder (from the course repo folder)
cd 00_ML_Introduction/intro-demo-scripts

# Install the course libraries
pip install -r ccfd-project/requirements.txt
```

Full steps: [Section 02, Step-02: Create Conda Environment](../../02_Setup_Python/README.md#step-02-create-conda-environment) and [Step-03: Install Dependencies](../../02_Setup_Python/README.md#step-03-install-dependencies).

## Run it

```bash
# Activate the course environment
conda activate mlops-env1

# Go to the intro demo folder (from the course repo folder)
cd 00_ML_Introduction/intro-demo-scripts

# Run the intro demo (Ctrl+C stops it)
bash run_intro_demo.sh
```

If `conda activate` does not work yet, the script still runs: it looks for the course environment `mlops-env1` in the usual Miniconda places and uses it.

It takes about 2 minutes on a laptop. On Windows, run it in Git Bash. Each stage's header also shows where that step is in the course (for example "In the course: Section 03, Synthetic Dataset Generation").

| Stage | Section | What happens | Where to look |
|---|---|---|---|
| 1 | 03 | Generate the dataset: 10,000 card transactions (`p1_01`) | `ccfd-project/data/` |
| 2 | 04 A | Train the baseline model (`p3_01`) | `ccfd-project/models/p3_01_baseline/` |
| 3 | 04 B | Start the FastAPI app for the baseline model (demo version of `p3_02`) | http://localhost:8000/ui and http://localhost:8000/docs |
| 4 | 06 | Add the new feature `amount_ratio` (`p4_01`) | `ccfd-project/data/` |
| 5 | 07 | Find the best settings with a grid search, about a minute (`p5_02`) | `ccfd-project/artifacts/p5_02_best_config.yaml` |
| 6 | 07 | Train the final tuned model (`p5_03`) | `ccfd-project/models/p5_03_production/` |
| 7 | 08 | Start the FastAPI app for the tuned model (demo version of `p5_04`) | http://localhost:8001/ui and http://localhost:8001/docs |
| 8 | (this intro demo only) | Open both APIs side by side | http://localhost:8300/side-by-side.html |
| 9 | 04 B and 08 | Send the final test data to both models | The difference is printed in the terminal |

**See the difference on the page:** on both sides, click the purple **Real Fraud from Test Data (row 7382)** button, then **CHECK FRAUD** on each side. Row 7382 is a real fraud from our test data: $363.26 at a gas station, card not present. The baseline model lets it through; the tuned model catches it.

Press **Ctrl+C** (or close the terminal) to stop all three: both APIs and the page server. It prints "Stopped. Ports 8000, 8001 and 8300 are free again." and you can start the demo again right away. Each step's output is kept in `logs/`.

Ports 8000, 8001 and 8300 must be free. If an API from Section 04 B or 08 is still running, stop it first.
