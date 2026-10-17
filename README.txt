X-RAY DOUBLE-SLIT INTERFERENCE — COMPETITION BUNDLE
Author: Asutosh Sahoo | Veer Surendra Sai University of Technology (VSSUT)

IMPORTANT STATUS LABEL
This is an interactive computational model. It does not generate or measure real X-rays, and no physical experiment was performed. Hard-X-ray double-slit interference has been demonstrated previously (Gureyev et al., 2024; DOI 10.1364/OE.521892). Do not claim the physical effect as new.

WINDOWS QUICK START
1. Install Python 3.10 or newer: https://www.python.org/downloads/
2. Extract this ZIP.
3. Open Command Prompt in the extracted project folder.
4. Run: py -m pip install -r requirements.txt
5. Run: py -m streamlit run app.py
6. Open the local URL printed in the terminal (usually http://localhost:8501).
If `py` is unavailable, use `python` instead.

FILES
- app.py: interactive simulator and calibrated Monte Carlo benchmark.
- Presentation.pptx: competition slides with speaker-friendly explanations.
- Research_Paper.docx / .pdf: competition report; all outputs are labelled simulated.
- Competition_Kit.txt: pitch, poster abstract, judge Q&A, demo run sheet and integrity checklist.
- benchmark_default.csv / .json: reproducible default-condition statistical results.
- power_curve.csv: simulated power results versus photon count.
- figures/: simulation figures used in the report and presentation.

STATISTICAL METHOD
For each score, a threshold is calibrated on null simulations at target alpha. An independent null set estimates false-positive rate, and a separate alternative set estimates detection power. Wilson 95% intervals quantify binomial Monte Carlo uncertainty only. The benchmark assumes the fringe frequency is specified in advance; searching unknown frequencies needs additional multiple-testing calibration.

REPRODUCIBILITY
Fixed seeds are saved in the output files. `benchmark_default.csv` and `benchmark_default.json` describe N=3000, V=0.7, B=0.1, target alpha=0.05 and 5000 trials per stage. `power_curve.csv` uses V=0.25, B=0.25 and 500 trials per stage per photon count. Small run-to-run differences can occur if parameters or code change.

LIMITATIONS
The model is a one-dimensional sinusoid with additive uniform background and fixed total photon count. It omits finite-slit diffraction envelope, energy bandwidth, detailed coherence, detector point-spread function, dead time, dark counts, drift and correlated noise. Simulated performance is not a physical detector performance claim.
