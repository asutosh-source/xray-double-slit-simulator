# X-Ray Double-Slit Interference --- Competition Bundle

**Author:** Asutosh Sahoo\
**Institution:** Veer Surendra Sai University of Technology (VSSUT)

> **Project status:** This repository contains an interactive
> computational model and simulated statistical benchmarks. It does
> **not** generate or measure real X-rays, and no physical experiment
> was performed.

## Overview

This project explores the statistical detection of a double-slit
interference pattern using an interactive simulation and reproducible
Monte Carlo benchmarks. It is designed as a competition demonstration
and educational research project.

The simulator models a one-dimensional sinusoidal fringe signal with an
additive uniform background and a fixed total photon count. It compares
scores against calibrated thresholds and estimates false-positive rates
and detection power using independent simulated datasets.

**Scientific context:** Hard-X-ray double-slit interference has been
demonstrated previously, including by Gureyev et al. (2024), DOI:
[10.1364/OE.521892](https://doi.org/10.1364/OE.521892). This project
does not claim the physical effect as a new discovery. Its contribution
is the computational model, statistical evaluation, and
competition-ready presentation of the simulated results.

## Contents

-   `app.py` --- Interactive simulator and calibrated Monte Carlo
    benchmark.
-   `Presentation.pptx` --- Competition presentation with
    speaker-friendly explanations.
-   `Research_Paper.docx` / `Research_Paper.pdf` --- Competition report.
    Outputs are labelled as simulated.
-   `Competition_Kit.txt` --- Project pitch, poster abstract, judge Q&A,
    demo run sheet, and integrity checklist.
-   `benchmark_default.csv` / `benchmark_default.json` --- Reproducible
    statistical results for the default configuration.
-   `power_curve.csv` --- Simulated detection-power results across
    photon counts.
-   `figures/` --- Simulation figures used in the report and
    presentation.
-   `requirements.txt` --- Python dependencies.

## Requirements

-   Windows, macOS, or Linux
-   Python 3.10 or newer
-   Internet access for the initial installation of Python dependencies

## Quick start on Windows

1.  Install [Python 3.10 or newer](https://www.python.org/downloads/).

2.  Download or clone this repository, then extract it if downloaded as
    a ZIP.

3.  Open **Command Prompt** in the project folder---the folder
    containing `app.py` and `requirements.txt`.

4.  Install the dependencies:

    ``` bash
    py -m pip install -r requirements.txt
    ```

5.  Launch the Streamlit app:

    ``` bash
    py -m streamlit run app.py
    ```

6.  Open the local URL printed in the terminal. It is usually
    `http://localhost:8501`.

If the `py` command is unavailable, use `python` instead:

``` bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

To stop the app, return to the terminal and press **Ctrl+C**.

## Statistical method

For each score, the workflow is:

1.  Calibrate a decision threshold using null simulations at the target
    significance level (`alpha`).
2.  Estimate the false-positive rate on an **independent null simulation
    set**.
3.  Estimate detection power on a **separate alternative simulation
    set**.
4.  Report Wilson 95% confidence intervals to quantify binomial Monte
    Carlo uncertainty.

The default benchmark uses a fringe frequency specified in advance. If
the fringe frequency is searched for rather than fixed beforehand,
additional multiple-testing calibration is required.

### Default benchmark configuration

  Parameter                                Value
  -------------------------------------- -------
  Total photon count (`N`)                  3000
  Visibility (`V`)                           0.7
  Background parameter (`B`)                 0.1
  Target false-positive rate (`alpha`)      0.05
  Trials per stage                          5000

The saved files `benchmark_default.csv` and `benchmark_default.json`
record the default-condition results. Fixed random seeds are saved in
the output files to support reproducibility.

### Power-curve configuration

`power_curve.csv` contains simulated power results versus photon count,
using:

-   `V = 0.25`
-   `B = 0.25`
-   500 trials per stage per photon-count setting

Small run-to-run differences may occur if parameters, code, or software
environments change.

## Model assumptions and limitations

The model is intentionally simplified. It uses:

-   A one-dimensional sinusoidal fringe pattern
-   An additive uniform background
-   A fixed total photon count

It does **not** currently model:

-   The finite-slit diffraction envelope
-   X-ray energy bandwidth
-   Detailed spatial or temporal coherence effects
-   Detector point-spread function
-   Detector dead time or dark counts
-   Instrument drift
-   Correlated noise

Consequently, simulation results should not be interpreted as
measurements of a real X-ray apparatus or as claims about real detector
performance. The confidence intervals quantify Monte Carlo sampling
uncertainty under the model; they do not account for model
misspecification or experimental uncertainty.

## Reproducibility

-   Use the saved CSV and JSON files to inspect the default benchmark
    results.
-   Use `power_curve.csv` to inspect simulated power across photon
    counts.
-   Check the random seeds and parameter settings recorded in the
    outputs when reproducing results.
-   If you modify the simulation, rerun the benchmarks and clearly
    document the changed assumptions and configuration.

## Responsible interpretation

When presenting or evaluating this project:

-   Clearly label simulated figures and results as **simulated**.
-   Do not state or imply that real X-rays were generated, measured, or
    experimentally tested in this project.
-   Do not claim the existence of X-ray double-slit interference as a
    new discovery.
-   Distinguish Monte Carlo uncertainty from real-world experimental
    uncertainty.
-   Treat the results as evidence about the stated computational model,
    not as proof of physical detector performance.

## Acknowledgement of prior work

Hard-X-ray double-slit interference has been demonstrated previously.
One relevant reference is:

Gureyev et al. (2024). DOI:
[10.1364/OE.521892](https://doi.org/10.1364/OE.521892).

Please consult the original publication for its exact methods,
experimental setup, and conclusions.
