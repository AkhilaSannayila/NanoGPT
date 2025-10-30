nanoGPT Hyperparameter Experiment

This repository contains my submission for the Deep Learning assignment (CSCE @ UNT, Amir Mirzaeinia), based on Andrej Karpathy's nanoGPT.

The project involved two parts:

Code Analysis: A detailed study of the model.py, train.py, and sample.py files.

Hyperparameter Experiments: Running a grid of 32 experiments to analyze the impact of key hyperparameters on model performance.

Final Report

My complete 12-page analysis, including all study question answers, results, and conclusions, is available in this repository:

View Final Report (PDF)

Summary of Experimental Findings

The 32 experiments were based on the "Group 1" parameters (block_size=64, n_layer=4). The key findings were:

Model Size (n_embd): The larger n_embd=256 (16.03M params) models consistently outperformed the n_embd=128 (7.23M params) models, achieving a lower validation loss.

Dropout: dropout=0.2 was the most critical parameter for preventing overfitting. All Top 5 models used this setting, and it dramatically stabilized validation loss on longer training runs (2000 iterations).

Top 5 Best Models (by Validation Loss)

Rank

Min Validation Loss

Experiment Name

1

4.7089

exp30-nh8-ne256-b16-mi1000-d0.2

2

4.7089

exp32-nh8-ne256-b16-mi2000-d0.2

3

4.7179

exp14-nh4-ne256-b16-mi1000-d0.2

4

4.7179

exp16-nh4-ne256-b16-mi2000-d0.2

5

4.7356

exp28-nh8-ne256-b8-mi2000-d0.2

Validation Loss Plot

This plot shows the validation loss for all 32 runs. The classic "U-shaped" curve of overfitting is clearly visible in the models that used dropout=0.1 and were trained for 2000 iterations.

How to Reproduce

This project was run on Windows 11 with PowerShell and an NVIDIA GPU.

1. Setup

# 1. Clone the repository
git clone [Your-Repo-Link-Here]
cd [Your-Repo-Name]

# 2. Create and activate a Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt


2. Prepare Data

# Download and prepare the Shakespeare dataset
python data/shakespeare/prepare.py


3. Run Experiments (Optional - Logs Provided)

All 32 raw .log files are included in the /logs directory. To re-run the experiments:

# This will run all 32 experiments for Group 1.
# This will take a long time and requires a CUDA-enabled GPU.
.\run_experiments.ps1


4. Analyze Results

You can use the provided Python scripts to analyze the .log files.

# Parse all 32 logs and print the Top 5 best models
python parse_logs.py

# Generate the training and validation loss plots
python plot_logs.py


5. Sample from a Model

# Sample from the best-performing model (exp32)
python sample.py --out_dir="out-g1-exp32-bs64-nl4-nh8-ne256-b16-mi2000-d0.2"


Repository Structure

/
├── Final_Report.pdf         # The complete assignment write-up
├── report.md                  # Markdown source for the report
|
├── model.py                 # Core GPT model architecture
├── train.py                 # Training script
├── sample.py                # Text generation script
|
├── run_experiments.ps1        # My script to automate all 32 runs
├── parse_logs.py              # My script to parse logs and find top models
├── plot_logs.py               # My script to generate loss plots
|
├── training_loss_plot.png   # Final plot of training loss
├── validation_loss_plot.png # Final plot of validation loss
|
├── logs/                      # Contains all 32 raw training.log files
│   ├── out-g1-exp1-... .log
│   └── ... (31 more logs)
|
├── data/shakespeare/          # (Contains train.bin/val.bin after running prepare.py)
├── requirements.txt           # All Python dependencies
└── README.md                  # This file


Acknowledgements

This project is an assignment for the CSCE Deep Learning course at UNT, taught by Amir Mirzaeinia.

The core nanoGPT code is from Andrej Karpathy's repository: https://github.com/karpathy/nanoGPT
