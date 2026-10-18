"""X-ray double-slit interference simulator with detector effects and calibrated tests.

SIMULATION ONLY: this app does not generate real X-rays or contain experimental data.
Install: python -m pip install -r requirements.txt
Run:     python -m streamlit run app.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.ndimage import gaussian_filter1d

HC_KEV_NM = 1.239841984


def wavelength_nm(energy_kev):
    return HC_KEV_NM / float(energy_kev)


def integrate_profile(edges_um, period_um, visibility, background_fraction=0.0,
                      slit_width_nm=100.0, wavelength_value_nm=0.05,
                      distance_m=10.0, background_gradient=0.0,
                      phase_drift_rad=0.0, phase=0.0, n_quad=12):
    """Numerically integrate a finite-slit envelope × two-slit fringe over bins.

    The envelope is sinc^2(pi*a*x/(lambda*L)); the interference term is
    1 + V*cos(2*pi*x/p + phase). A uniform/additive background can have a
    linear spatial gradient. Phase drift is represented by averaging uniformly
    over a total phase excursion, not by a time-resolved instrument model.
    """
    edges_m = np.asarray(edges_um, dtype=float) * 1e-6
    a_m = float(slit_width_nm) * 1e-9
    lam_m = float(wavelength_value_nm) * 1e-9
    L_m = float(distance_m)
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    left, right = edges_m[:-1], edges_m[1:]
    xs = (left[:, None] + right[:, None]) / 2 + (right-left)[:, None] * nodes[None, :] / 2
    dx = (right-left)[:, None] / 2
    # np.sinc(z) = sin(pi*z)/(pi*z)
    envelope = np.sinc(a_m * xs / (lam_m * L_m)) ** 2
    k = 2*np.pi / (float(period_um)*1e-6)
    if phase_drift_rad > 0:
        phases = np.linspace(-phase_drift_rad/2, phase_drift_rad/2, 21) + phase
        fringe = np.mean([1 + visibility*np.cos(k*xs + ph) for ph in phases], axis=0)
    else:
        fringe = 1 + visibility*np.cos(k*xs + phase)
    # Background is additive relative to the central envelope scale.
    xscale = max(np.max(np.abs(xs)), 1e-30)
    bg = background_fraction * np.clip(1 + background_gradient*(xs/xscale), 0.05, 2.0)
    intensity = np.maximum(envelope*fringe + bg, 0.0)
    integrals = np.sum(intensity * weights[None, :] * dx, axis=1)
    integrals = np.maximum(integrals, 0.0)
    return integrals / max(integrals.sum(), 1e-300)


def detector_blur_probabilities(probabilities, bins_sigma):
    if bins_sigma <= 0:
        return probabilities / probabilities.sum()
    p = gaussian_filter1d(np.asarray(probabilities, dtype=float), sigma=bins_sigma, mode="nearest")
    p = np.maximum(p, 0)
    return p / p.sum()


def simulate_counts(n_expected, probs, rng, counting_mode):
    if counting_mode == "Poisson photon counting (variable total)":
        return rng.poisson(float(n_expected) * probs)
    return rng.multinomial(int(n_expected), probs)


def fourier_score(counts, centers, period_um):
    y = np.asarray(counts, dtype=float) - np.mean(counts)
    phase = 2*np.pi*centers/period_um
    c, s = np.sum(y*np.cos(phase)), np.sum(y*np.sin(phase))
    denom = np.sqrt(np.sum(y*y)*np.sum(np.cos(phase)**2 + np.sin(phase)**2)) + 1e-12
    return float(np.hypot(c, s)/denom)


def fit_modulation_score(counts, centers, period_um):
    phase = 2*np.pi*centers/period_um
    X = np.column_stack([np.ones_like(centers), np.cos(phase), np.sin(phase)])
    beta, *_ = np.linalg.lstsq(X, np.asarray(counts, dtype=float), rcond=None)
    return float(np.hypot(beta[1], beta[2])/(abs(beta[0])+1e-12))


def score_counts(counts, centers, period_um, period_candidates, method, search_frequency):
    periods = period_candidates if search_frequency else np.array([period_um])
    fn = fourier_score if method == "Fourier score" else fit_modulation_score
    scores = np.array([fn(counts, centers, p) for p in periods])
    i = int(np.argmax(scores))
    return float(scores[i]), float(periods[i])


def wilson_interval(successes, trials, z=1.959963984540054):
    if trials <= 0:
        return (float("nan"), float("nan"))
    phat = successes/trials
    denom = 1 + z*z/trials
    center = (phat + z*z/(2*trials))/denom
    half = z*np.sqrt(phat*(1-phat)/trials + z*z/(4*trials*trials))/denom
    return max(0.0, center-half), min(1.0, center+half)


def calibrated_benchmark(n_trials, n_expected, p_null, p_alt, centers, period_um,
                         alpha, seed, counting_mode, search_frequency, period_candidates):
    """Calibrate on null simulations; validate FPR and power on independent streams.

    Empirical p-values use the calibration-score distribution and +1 correction.
    In scan mode, each trial's score is the maximum over the searched periods, so
    the null calibration includes the frequency-search penalty.
    """
    rng_cal = np.random.default_rng(seed)
    rng_null = np.random.default_rng(seed + 1_000_003)
    rng_alt = np.random.default_rng(seed + 2_000_033)
    results = {}
    for method in ("Fourier score", "Sinusoidal-fit score"):
        cal = np.empty(n_trials)
        null_scores = np.empty(n_trials)
        alt_scores = np.empty(n_trials)
        alt_periods = np.empty(n_trials)
        for i in range(n_trials):
            c = simulate_counts(n_expected, p_null, rng_cal, counting_mode)
            cal[i], _ = score_counts(c, centers, period_um, period_candidates, method, search_frequency)
            c0 = simulate_counts(n_expected, p_null, rng_null, counting_mode)
            null_scores[i], _ = score_counts(c0, centers, period_um, period_candidates, method, search_frequency)
            c1 = simulate_counts(n_expected, p_alt, rng_alt, counting_mode)
            alt_scores[i], alt_periods[i] = score_counts(c1, centers, period_um, period_candidates, method, search_frequency)
        # Calibrated empirical upper-tail p-values with tie handling.
        null_p = (1 + np.sum(cal[None, :] >= null_scores[:, None], axis=1))/(n_trials+1)
        alt_p = (1 + np.sum(cal[None, :] >= alt_scores[:, None], axis=1))/(n_trials+1)
        fp, tp = int(np.sum(null_p <= alpha)), int(np.sum(alt_p <= alpha))
        results[method] = {
            "threshold": float(np.quantile(cal, 1-alpha, method="higher")),
            "false_positives": fp, "null_trials": n_trials,
            "false_positive_rate": fp/n_trials,
            "false_positive_ci95": wilson_interval(fp, n_trials),
            "detections": tp, "alternative_trials": n_trials,
            "power": tp/n_trials, "power_ci95": wilson_interval(tp, n_trials),
            "median_best_period_um_in_alternative": float(np.median(alt_periods)) if search_frequency else float(period_um),
        }
    results["metadata"] = {
        "target_alpha": alpha, "trials_per_stage_per_method": n_trials,
        "expected_photon_count": int(n_expected), "counting_mode": counting_mode,
        "frequency_mode": "searched frequency; threshold calibrated on max score" if search_frequency else "known frequency",
        "searched_period_range_um": [float(period_candidates.min()), float(period_candidates.max())] if search_frequency else [float(period_um), float(period_um)],
        "seed": int(seed),
        "interpretation": "Monte Carlo results under this specified simulation model; not experimental measurements.",
    }
    return results


st.set_page_config(page_title="X-ray Double-Slit Simulator — Extended", layout="wide")
st.title("When Does Randomness Become a Pattern?")
st.subheader("Extended computational X-ray double-slit interference simulator")
st.warning("SIMULATION ONLY. No real X-rays, physical slits, detector measurements, or independent experimental dataset are included.")

with st.sidebar:
    st.header("Optical / geometric model")
    energy = st.slider("Photon energy (keV)", 1.0, 40.0, 25.0, 0.5)
    slit_sep_nm = st.slider("Slit separation d (nm)", 100.0, 2000.0, 500.0, 25.0)
    slit_width_nm = st.slider("Individual slit width a (nm)", 10.0, 500.0, 100.0, 10.0)
    distance_m = st.slider("Detector distance L (m)", 1.0, 120.0, 10.0, 1.0)
    visibility = st.slider("Fringe visibility V", 0.0, 1.0, 0.7, 0.05)
    background = st.slider("Background fraction", 0.0, 1.0, 0.1, 0.05)
    background_gradient = st.slider("Background gradient (− to + across detector)", -0.8, 0.8, 0.0, 0.1)
    photons = st.select_slider("Expected / fixed photon count", options=[100, 300, 1000, 3000, 10000, 30000, 100000], value=3000)
    pixels = st.slider("Detector bins", 32, 512, 128, 16)
    blur_sigma = st.slider("Detector blur (standard deviation, bins)", 0.0, 5.0, 0.5, 0.25)
    phase_drift = st.slider("Total phase drift during acquisition (radians)", 0.0, 3.14, 0.0, 0.1)
    counting_mode = st.selectbox("Counting statistics", ["Fixed total (multinomial)", "Poisson photon counting (variable total)"])
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)

lam_nm = wavelength_nm(energy)
period_um = lam_nm * 1e-9 * distance_m / (slit_sep_nm * 1e-9) * 1e6
x_min, x_max = -3*period_um, 3*period_um
edges = np.linspace(x_min, x_max, pixels+1)
centers = (edges[:-1]+edges[1:])/2
base_args = dict(slit_width_nm=slit_width_nm, wavelength_value_nm=lam_nm, distance_m=distance_m,
                 background_gradient=background_gradient, phase_drift_rad=phase_drift)
p_alt = integrate_profile(edges, period_um, visibility, background, **base_args)
p_alt = detector_blur_probabilities(p_alt, blur_sigma)
p_null = integrate_profile(edges, period_um, 0.0, background, **base_args)
p_null = detector_blur_probabilities(p_null, blur_sigma)
rng = np.random.default_rng(int(seed))
counts = simulate_counts(photons, p_alt, rng, counting_mode)
# Event locations are sampled from the same binned model; sub-bin positions are illustrative.
idx = np.repeat(np.arange(pixels), counts)
if len(idx):
    positions = rng.uniform(edges[idx], edges[idx+1])
else:
    positions = np.array([])
expected = (photons if counting_mode == "Fixed total (multinomial)" else photons) * p_alt

m1, m2, m3, m4 = st.columns(4)
m1.metric("Wavelength", f"{lam_nm:.5f} nm")
m2.metric("Predicted fringe period λL/d", f"{period_um:.4f} µm")
m3.metric("Simulated detections", f"{int(counts.sum()):,}")
m4.metric("Slit width / separation", f"{slit_width_nm/slit_sep_nm:.3f}")

left, right = st.columns(2)
with left:
    st.markdown("### Simulated individual detection events")
    fig, ax = plt.subplots(figsize=(8, 3.7))
    if len(positions) > 0:
        ax.scatter(positions, rng.uniform(0, 1, len(positions)), s=4, alpha=0.22)
    ax.set_xlabel("Detector position (µm)")
    ax.set_yticks([])
    ax.set_title("Dots are simulated events, not detector records")
    st.pyplot(fig)
with right:
    st.markdown("### Accumulated detector-bin counts")
    fig2, ax2 = plt.subplots(figsize=(8, 3.7))
    ax2.bar(centers, counts, width=np.diff(edges), alpha=0.65, label="Simulated counts")
    ax2.plot(centers, expected, linewidth=2, label="Expected mean counts")
    # Plot the finite-slit envelope shape as a visual guide.
    x_dense = np.linspace(x_min, x_max, 1000)
    envelope = np.sinc((slit_width_nm*1e-9)*(x_dense*1e-6)/((lam_nm*1e-9)*(distance_m)))**2
    ax2.plot(x_dense, expected.max()*envelope/max(envelope.max(), 1e-12), linestyle="--", label="Single-slit envelope (scaled)")
    ax2.set_xlabel("Detector position (µm)")
    ax2.set_ylabel("Counts per bin")
    ax2.legend(fontsize=8)
    st.pyplot(fig2)

with st.expander("Model details and equations", expanded=False):
    st.markdown(r"""
    **Finite-slit far-field model**

    \\[I(x) \propto \operatorname{sinc}^2\!\left(\frac{a x}{\lambda L}\right)\left[1+V\cos\left(\frac{2\pi x}{p}+\phi\right)\right]+B(x)\\]

    where \(a\) is slit width, \(p=\lambda L/d\) is the fringe period, \(V\) is effective visibility and \(B(x)\) is additive background. The app numerically integrates intensity across each bin.

    Detector blur is modelled as a Gaussian convolution in bin space. Phase drift is approximated by averaging over a uniform phase excursion. These are simplified detector-effect models, not a calibrated beamline instrument model.
    """)

st.markdown("### Calibrated Monte Carlo detection benchmark")
st.write("Thresholds are calibrated using simulated no-interference data. False-positive rate is estimated on an independent null stream; power is estimated on an independent interference stream. In frequency-search mode, every trial uses the maximum score over the selected period grid, and the calibration null uses the same search to account for the look-elsewhere effect.")
alpha = st.select_slider("Target false-positive rate α", options=[0.01, 0.05, 0.10], value=0.05)
trials = st.select_slider("Trials per stage (calibration, null validation, interference)", options=[200, 500, 1000, 2000], value=500)
frequency_mode = st.radio("Fringe-frequency assumption", ["Known frequency", "Search a range of frequencies"], horizontal=True)
search_frequency = frequency_mode == "Search a range of frequencies"
search_fraction = st.slider("Search range around predicted period (± fraction)", 0.1, 0.8, 0.4, 0.1, disabled=not search_frequency)
search_points = st.slider("Candidate periods in search", 5, 61, 21, 4, disabled=not search_frequency)
period_candidates = np.linspace(period_um*(1-search_fraction), period_um*(1+search_fraction), search_points) if search_frequency else np.array([period_um])
if st.button("Run calibrated benchmark"):
    with st.spinner("Running independent Monte Carlo stages; frequency search can take longer..."):
        result = calibrated_benchmark(trials, photons, p_null, p_alt, centers, period_um,
                                      alpha, int(seed)+17, counting_mode, search_frequency, period_candidates)
    rows = []
    for method in ("Fourier score", "Sinusoidal-fit score"):
        r = result[method]
        lo, hi = r["false_positive_ci95"]
        plo, phi = r["power_ci95"]
        st.markdown(f"**{method}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Calibrated score cutoff", f"{r['threshold']:.4f}")
        c2.metric("Independent null false-positive rate", f"{r['false_positive_rate']:.1%}", f"95% Wilson CI {lo:.1%}–{hi:.1%}")
        c3.metric("Detection power", f"{r['power']:.1%}", f"95% Wilson CI {plo:.1%}–{phi:.1%}")
        rows.append({"method": method, "threshold": r["threshold"], "false_positive_rate": r["false_positive_rate"],
                     "fpr_ci95_low": lo, "fpr_ci95_high": hi, "power": r["power"],
                     "power_ci95_low": plo, "power_ci95_high": phi,
                     "trials_per_stage": trials, "target_alpha": alpha, "expected_photons": photons,
                     "counting_mode": counting_mode, "frequency_mode": result["metadata"]["frequency_mode"]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.json(result)
    st.download_button("Download benchmark results (CSV)", pd.DataFrame(rows).to_csv(index=False),
                       file_name="benchmark_results_extended.csv", mime="text/csv")
    st.caption("Wilson intervals quantify binomial Monte Carlo sampling uncertainty conditional on the calibration sample. They do not include calibration-threshold uncertainty, model misspecification, or real-instrument uncertainty.")

st.markdown("### Existing project evidence and limitations")
st.markdown("""
- **Present in this upgraded app:** known-frequency Fourier and sinusoidal-fit scores; independent Monte Carlo calibration/null/power streams; Wilson intervals; photon-count controls; a saved-results download; finite slit-width diffraction envelope; nonuniform linear background; Gaussian detector blur; phase-drift approximation; optional Poisson counting; and calibrated frequency scanning.
- **Still missing:** independent measured experimental data. No experimental data are bundled, so this app cannot claim validation against a real synchrotron/detector dataset. Add a sourced dataset and document its instrument metadata, preprocessing and provenance before making that comparison.
- **Model caveats:** scalar paraxial far-field approximation; one-dimensional detector; idealized rectangular slit; simplified blur/drift/background; no detector dead time, energy spectrum, geometry uncertainty, or correlated noise. This is a more realistic simulation, not a full beamline simulator.
""")
st.caption("Project by Asutosh Sahoo | Veer Surendra Sai University of Technology (VSSUT)")
