# LightGBM benchmark

This is a simple program to benchmark LightGBM to compare changes in hardware, compiler settings,
and software versions.

We use the KDD 1999 data set, which we set up as a binary classification problem with 4,898,431 columns
x 40 variables. It is large enough to give a reliable benchmark, while being small enough not to take a
long time to run.

This code is *not* for intended for:

* Comparing the speed of LightGBM to XGBoost, other gradient boosting frameworks, or other classification algorithms
* Compare the accuracy of gradient boosting frameworks
* Tuning hyperparameters
* Producing model with real-world application

## Usage

Install the prerequisites

`pip3 install -r requirements.txt`

Run the benchmark

`python3 benchmark_lightgbm.py`

