"""
Photon-by-photon X-ray double-slit interference: educational simulation.
No real X-ray source or detector is used.

Install: python -m pip install -r requirements.txt
Run:     python -m streamlit run app.py
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

HC_KEV_NM = 1.239841984  # h*c in keV·nm


def wavelength_nm(energy_kev):
    return HC_KEV_NM / float(energy_kev)


def expected_profile(x_um, fringe_period_um, visibility, background_fraction=0.0, phase=0.0):
    """Relative intensity I(x)=1+V*cos(2πx/p+phase)+B."""
    return 1.0 + visibility * np.cos(2 * np.pi * x_um / fringe_period_um + phase) + background_fraction


def bin_probabilities(edges, period_um, visibility, background_fraction=0.0, phase=0.0):
    """Exact bin-integrated probabilities for a sinusoidal intensity over detector edges."""
    a, b = edges[:-1], edges[1:]
    k = 2 * np.pi / period_um
    integrals = (1.0 + background_fraction) * (b - a)
    integrals += visibility * (np.sin(k * b + phase) - np.sin(k * a + phase)) / k
    integrals = np.maximum(integrals, 0.0)
    return integrals / integrals.sum()


def sample_positions(n_photons, edges, period_um, visibility, background_fraction, rng):
    """Sample continuous positions via exact integrated bin probabilities, then uniform within bin."""
    probs = bin_probabilities(edges, period_um, visibility, background_fraction)
    idx = rng.choice(len(probs), size=int(n_photons), replace=True, p=probs)
    return rng.uniform(edges[idx], edges[idx + 1])


def binned_counts(positions, edges):
    return np.histogram(positions, bins=edges)[0]


def fourier_score(counts, x_centers, expected_period_um):
    """Known-frequency two-quadrature score; phase is not assumed known."""
    y = counts.astype(float) - np.mean(counts)
    phase = 2 * np.pi * x_centers / expected_period_um
    c = np.sum(y * np.cos(phase))
    s = np.sum(y * np.sin(phase))
    denom = np.sqrt(np.sum(y * y) * np.sum(np.cos(phase)**2 + np.sin(phase)**2)) + 1e-12
    return float(np.sqrt(c*c + s*s) / denom)


def fit_modulation_score(counts, x_centers, expected_period_um):
    """Least-squares sinusoidal amplitude / intercept; frequency fixed, phase free."""
    phase = 2 * np.pi * x_centers / expected_period_um
    X = np.column_stack([np.ones_like(x_centers), np.cos(phase), np.sin(phase)])
    beta, *_ = np.linalg.lstsq(X, counts.astype(float), rcond=None)
    return float(np.hypot(beta[1], beta[2]) / (abs(beta[0]) + 1e-12))


def draw_counts(n, probabilities, rng):
    return rng.multinomial(int(n), probabilities)


def wilson_interval(successes, trials, z=1.959963984540054):
    """95% Wilson score interval for a binomial proportion."""
    if trials <= 0:
        return (float("nan"), float("nan"))
    phat = successes / trials
    denom = 1 + z*z/trials
    center = (phat + z*z/(2*trials)) / denom
    half = z * np.sqrt(phat*(1-phat)/trials + z*z/(4*trials*trials)) / denom
    return max(0.0, center-half), min(1.0, center+half)


def benchmark(n_trials, photon_count, visibility, background_fraction,
              edges, centers, period_um, alpha, seed):
    """
    Pre-specified known spatial frequency; phase is free.
    Separate random streams are used for threshold calibration, null validation,
    and alternative/power evaluation. Each score has its own null-calibrated threshold.
    """
    rng_cal = np.random.default_rng(seed)
    rng_null = np.random.default_rng(seed + 1_000_003)
    rng_alt = np.random.default_rng(seed + 2_000_033)
    n_bins = len(centers)
    null_probs = np.full(n_bins, 1.0/n_bins)
    alt_probs = bin_probabilities(edges, period_um, visibility, background_fraction)

    score_fns = {
        "Fourier score": lambda c: fourier_score(c, centers, period_um),
        "Sinusoidal-fit score": lambda c: fit_modulation_score(c, centers, period_um),
    }
    results = {}
    for name, score_fn in score_fns.items():
        cal = np.array([score_fn(draw_counts(photon_count, null_probs, rng_cal))
                        for _ in range(n_trials)])
        # Empirical upper-tail p-values use only the calibration null sample.
        # The +1 correction and >= tie handling are conservative for discrete scores.
        threshold = float(np.quantile(cal, 1.0-alpha, method="higher"))
        null_scores = np.array([score_fn(draw_counts(photon_count, null_probs, rng_null))
                                for _ in range(n_trials)])
        alt_scores = np.array([score_fn(draw_counts(photon_count, alt_probs, rng_alt))
                               for _ in range(n_trials)])
        null_p = (1 + np.sum(cal[None, :] >= null_scores[:, None], axis=1)) / (n_trials + 1)
        alt_p = (1 + np.sum(cal[None, :] >= alt_scores[:, None], axis=1)) / (n_trials + 1)
        fp = int(np.sum(null_p <= alpha))
        tp = int(np.sum(alt_p <= alpha))
        results[name] = {
            "threshold": threshold,
            "false_positives": fp,
            "null_trials": n_trials,
            "false_positive_rate": fp/n_trials,
            "false_positive_ci95": wilson_interval(fp, n_trials),
            "detections": tp,
            "alternative_trials": n_trials,
            "power": tp/n_trials,
            "power_ci95": wilson_interval(tp, n_trials),
        }
    results["metadata"] = {
        "target_alpha": alpha,
        "trials_per_stage_per_method": n_trials,
        "photon_count": int(photon_count),
        "visibility": float(visibility),
        "uniform_background_fraction": float(background_fraction),
        "seed": int(seed),
        "interpretation": "Power and false-positive rates are Monte Carlo estimates under this simplified model, not experimental results.",
    }
    return results


st.set_page_config(page_title="X-ray Double-Slit Simulator", layout="wide")
st.title("When Does Randomness Become a Pattern?")
st.subheader("Photon-by-photon X-ray double-slit interference — computational exhibition model")
st.warning("SIMULATION ONLY. No real X-rays, physical slits, or detector measurements are used. Results depend on a simplified far-field sinusoidal model.")

with st.sidebar:
    st.header("Model controls")
    energy = st.slider("Photon energy (keV)", 1.0, 40.0, 25.0, 0.5)
    slit_sep_nm = st.slider("Slit separation (nm)", 100.0, 2000.0, 500.0, 25.0)
    distance_m = st.slider("Detector distance (m)", 1.0, 120.0, 10.0, 1.0)
    visibility = st.slider("Fringe visibility V", 0.0, 1.0, 0.7, 0.05)
    background = st.slider("Uniform background B (relative intensity)", 0.0, 1.0, 0.1, 0.05)
    photons = st.select_slider("Total detected photons", options=[100, 300, 1000, 3000, 10000, 30000, 100000], value=3000)
    pixels = st.slider("Detector bins", 32, 512, 128, 16)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)

lam_nm = wavelength_nm(energy)
period_um = lam_nm * 1e-9 * distance_m / (slit_sep_nm * 1e-9) * 1e6
x_min, x_max = -3*period_um, 3*period_um
edges = np.linspace(x_min, x_max, pixels + 1)
centers = (edges[:-1] + edges[1:]) / 2
rng = np.random.default_rng(int(seed))
positions = sample_positions(photons, edges, period_um, visibility, background, rng)
counts = binned_counts(positions, edges)
prob = bin_probabilities(edges, period_um, visibility, background)
expected = photons * prob

m1, m2, m3 = st.columns(3)
m1.metric("Wavelength", f"{lam_nm:.5f} nm")
m2.metric("Predicted fringe period λL/d", f"{period_um:.4f} µm")
m3.metric("Simulated detections", f"{photons:,}")

left, right = st.columns(2)
with left:
    st.markdown("### Individual detection events")
    fig, ax = plt.subplots(figsize=(8, 3.7))
    ax.scatter(positions, rng.uniform(0, 1, len(positions)), s=4, alpha=0.25)
    ax.set_xlabel("Detector position (µm)")
    ax.set_yticks([])
    ax.set_title("Each dot is one simulated photon detection")
    st.pyplot(fig)
with right:
    st.markdown("### Accumulated count pattern")
    fig2, ax2 = plt.subplots(figsize=(8, 3.7))
    ax2.bar(centers, counts, width=np.diff(edges), alpha=0.65, label="Simulated counts")
    ax2.plot(centers, expected, linewidth=2, label="Expected mean counts")
    ax2.set_xlabel("Detector position (µm)")
    ax2.set_ylabel("Counts per bin")
    ax2.legend()
    st.pyplot(fig2)

st.markdown("### Calibrated Monte Carlo benchmark")
st.write("The test assumes the fringe spatial frequency is specified in advance but allows unknown phase. Each score receives empirical upper-tail p-values from calibration-only null simulations (+1 correction; conservative ties). Independent null simulations estimate false-positive rate; independent interference simulations estimate power.")
alpha = st.select_slider("Target false-positive rate α", options=[0.01, 0.05, 0.10], value=0.05)
trials = st.select_slider("Trials per stage (calibration, null validation, interference)", options=[200, 500, 1000, 2000], value=500)
if st.button("Run calibrated benchmark"):
    with st.spinner("Running independent Monte Carlo stages..."):
        result = benchmark(trials, photons, visibility, background, edges, centers,
                           period_um, alpha, int(seed)+17)
    for method in ("Fourier score", "Sinusoidal-fit score"):
        r = result[method]
        lo, hi = r["false_positive_ci95"]
        plo, phi = r["power_ci95"]
        st.markdown(f"**{method}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Calibration score cutoff (approx.)", f"{r['threshold']:.4f}")
        c2.metric("Independent null FPR", f"{r['false_positive_rate']:.1%}",
                  f"95% Wilson CI {lo:.1%}–{hi:.1%}")
        c3.metric("Detection power", f"{r['power']:.1%}",
                  f"95% Wilson CI {plo:.1%}–{phi:.1%}")
    st.json(result)
    st.caption("Wilson intervals condition on the selected calibration sample and quantify validation/power sampling uncertainty only; they do not include calibration-threshold uncertainty, model misspecification or real-instrument uncertainty.")

st.markdown("### Assumptions and limitations")
st.write("- Monochromatic photons and a one-dimensional far-field sinusoidal intensity.")
st.write("- Fringe period p = λL/d; λ = hc/E. Finite-slit envelope and beamline geometry are omitted.")
st.write("- Visibility is an effective parameter; background is uniform and additive.")
st.write("- Photon count is conditioned on a fixed total; counts are multinomial, not a full flux/time model.")
st.write("- Detector response, energy spread, dead time, dark counts, drift and correlated noise are not modelled.")
st.write("- The benchmark assumes the fringe frequency is known in advance. Searching over unknown periods requires a multiple-testing-aware calibration.")
st.caption("Project by Asutosh Sahoo | Veer Surendra Sai University of Technology (VSSUT)")
