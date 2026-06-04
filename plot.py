#!/usr/bin/env python3
"""
plot.py — Post-process Excalibur parametric sweep results.

Reads statepoint files from the OpenMC/ case folders and produces:

  Set 1 (raw):
    fig_neutron_raw.png   — exit-face neutron spectra (ClLiF, FLiBe, air)
    fig_gamma_raw.png     — exit-face gamma  spectra (ClLiF, FLiBe, air)

  Set 2 (air-subtracted):
    fig_neutron_sub.png   — (salt − air) neutron spectra
    fig_gamma_sub.png     — (salt − air) gamma  spectra

  summary.csv             — neutron flux at mouth and 1 m (total, <0.8 MeV, >10 MeV)

All results scaled to a DT source rate of 8.2 × 10⁸ n/s × cone fraction (±45°).
Each figure has two panels: collimated (left) and moderated (right).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import openmc


# ====================================================================
#  Case definitions  (must match excalibur_openmc.py sweep)
# ====================================================================

BASE_DIR = Path("OpenMC")

MODES        = ["collimated", "moderated"]
SALT_TYPES   = ["fluoride", "chloride"]
FERTILE_PCTS = [0.5, 5.0, 10.0]

SALT_BOX_LENGTH = 20.0   # cm — must match excalibur.py SALT_BOX_LENGTH
LENGTH_TAG      = f"_{SALT_BOX_LENGTH:g}cm"

SALT_LABELS = {"fluoride": "FLiBe", "chloride": "ClLiF"}
FERT_LABELS = {"fluoride": "UF4",   "chloride": "UCl3"}

# DT generator source rate — scaled for the 90° forward cone (±45°)
#   Physical rate: 8.2e8 n/s into 4π
#   Cone fraction: Ω/4π = (1 − cos 45°)/2 ≈ 14.64 %
_TOTAL_RATE    = 8.2e8   # n/s (full 4π)
_CONE_FRACTION = (1.0 - math.cos(math.radians(45.0))) / 2.0
SOURCE_RATE    = _TOTAL_RATE * _CONE_FRACTION   # ≈ 1.20e8 n/s into cone

# Line styles for each salt case
STYLE = {
    ("FLiBe", 0.5):  dict(color="#1D9E75", ls="-",  lw=1.4),
    ("FLiBe", 5.0):  dict(color="#0F6E56", ls="--", lw=1.4),
    ("FLiBe", 10.0): dict(color="#0A4F3D", ls="-.", lw=1.4),
    ("ClLiF", 0.5):  dict(color="#D85A30", ls="-",  lw=1.4),
    ("ClLiF", 5.0):  dict(color="#993C1D", ls="--", lw=1.4),
    ("ClLiF", 10.0): dict(color="#6B2A14", ls="-.", lw=1.4),
    ("air",   0.0):  dict(color="#888888", ls=":",  lw=1.6),
}


def case_dir_name(mode, salt_label, fert_label, fpct):
    """Return the folder name for a salt case."""
    return f"{mode}_{salt_label}_{fpct:.1f}mol{fert_label}{LENGTH_TAG}"


def air_dir_name(mode):
    """Return the folder name for an air-background case."""
    return f"{mode}_air_background{LENGTH_TAG}"


# ====================================================================
#  Spectrum extraction
# ====================================================================

def get_spectrum(case_dir, tally_name):
    """
    Read a spectrum tally from the latest statepoint in *case_dir*.

    Returns
    -------
    energy_mid : ndarray   geometric-mean bin centres (eV)
    mean       : ndarray   tally mean per bin
    std_dev    : ndarray   tally standard deviation per bin
    """
    sp_files = sorted(case_dir.glob("statepoint.*.h5"))
    if not sp_files:
        raise FileNotFoundError(f"No statepoint file in {case_dir}")

    sp = openmc.StatePoint(str(sp_files[-1]))
    t  = sp.get_tally(name=tally_name)
    df = t.get_pandas_dataframe()

    energy_low  = df["energy low [eV]"].values
    energy_high = df["energy high [eV]"].values
    energy_mid  = np.sqrt(energy_low * energy_high)
    mean        = df["mean"].values    * SOURCE_RATE
    std_dev     = df["std. dev."].values * SOURCE_RATE

    return energy_mid, mean, std_dev


# ====================================================================
#  Plotting helpers
# ====================================================================

def setup_ax(ax, title, ylabel="Current (s⁻¹ per bin)"):
    """Apply common axis formatting."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.4)


def plot_raw(particle, tally_name, out_filename, xlim=None):
    """
    Plot raw spectra for both modes.
    Lines: each salt case + air background.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, mode in zip(axes, MODES):
        # Salt cases
        for stype in SALT_TYPES:
            sl = SALT_LABELS[stype]
            fl = FERT_LABELS[stype]
            for fpct in FERTILE_PCTS:
                dname = case_dir_name(mode, sl, fl, fpct)
                cdir  = BASE_DIR / dname
                try:
                    e, mu, sd = get_spectrum(cdir, tally_name)
                    label = f"{sl} + {fpct:.1f}% {fl}"
                    ax.plot(e, mu, label=label, **STYLE[(sl, fpct)])
                except FileNotFoundError:
                    print(f"  ⚠ Skipping {dname} — no results")

        # Air background
        adir = BASE_DIR / air_dir_name(mode)
        try:
            e, mu, sd = get_spectrum(adir, tally_name)
            ax.plot(e, mu, label="Air (background)", **STYLE[("air", 0.0)])
        except FileNotFoundError:
            print(f"  ⚠ Skipping {air_dir_name(mode)} — no results")

        setup_ax(ax, f"{mode.capitalize()} — {particle}")
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(out_filename, dpi=200)
    print(f"  Saved {out_filename}")
    plt.close(fig)


def plot_subtracted(particle, tally_name, out_filename, xlim=None):
    """
    Plot air-subtracted spectra for both modes.
    Lines: (salt − air) for each salt case.  Air itself is NOT plotted.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, mode in zip(axes, MODES):
        # Load air background for this mode
        adir = BASE_DIR / air_dir_name(mode)
        try:
            e_air, mu_air, _ = get_spectrum(adir, tally_name)
        except FileNotFoundError:
            print(f"  ⚠ No air background for {mode} — skipping subtracted plot")
            continue

        # Salt cases, subtract air
        for stype in SALT_TYPES:
            sl = SALT_LABELS[stype]
            fl = FERT_LABELS[stype]
            for fpct in FERTILE_PCTS:
                dname = case_dir_name(mode, sl, fl, fpct)
                cdir  = BASE_DIR / dname
                try:
                    e, mu, sd = get_spectrum(cdir, tally_name)
                    mu_sub = mu - mu_air
                    label = f"{sl} + {fpct:.1f}% {fl} − air"
                    ax.plot(e, mu_sub, label=label, **STYLE[(sl, fpct)])
                except FileNotFoundError:
                    print(f"  ⚠ Skipping {dname} — no results")

        setup_ax(ax, f"{mode.capitalize()} — {particle} (air subtracted)",
                 ylabel="Current − background (s⁻¹ per bin)")
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(out_filename, dpi=200)
    print(f"  Saved {out_filename}")
    plt.close(fig)


def integrate_bands(case_dir, tally_name):
    """
    Load a spectrum tally and integrate over three energy bands.

    Returns dict with keys:
        total        — sum over all bins
        sub_0p8MeV   — sum where E_high ≤ 0.8 MeV
        gt_10MeV     — sum where E_low  ≥ 10 MeV
    Returns None if the statepoint is missing.
    """
    sp_files = sorted(case_dir.glob("statepoint.*.h5"))
    if not sp_files:
        return None

    sp = openmc.StatePoint(str(sp_files[-1]))
    t  = sp.get_tally(name=tally_name)
    df = t.get_pandas_dataframe()

    e_lo = df["energy low [eV]"].values
    e_hi = df["energy high [eV]"].values
    mu   = df["mean"].values * SOURCE_RATE

    total      = mu.sum()
    sub_0p8MeV = mu[e_hi <= 0.8e6].sum()
    gt_10MeV   = mu[e_lo >= 10.0e6].sum()

    return dict(total=total, sub_0p8MeV=sub_0p8MeV, gt_10MeV=gt_10MeV)


MOUTH_TALLY = "Neutron flux — Excalibur mouth (on-axis)"
FRONT_TALLY = "Incident neutron flux — salt front face"

# Tallies needed for the U-238 macroscopic-cross-section calculation.
# Names must match excalibur.py exactly.
FLUX_TALLY             = "Neutron flux spectrum — salt box"
URANIUM_SPECTRUM_TALLY = "U capture & fission spectrum — salt box"


# ====================================================================
#  Macroscopic cross sections  (Σ_x = reaction rate / flux)
# ====================================================================

def _macro_xs_from_dataframes(df_flux, df_rr, nuclide="U238"):
    """
    Flux-averaged macroscopic cross sections from tally dataframes.

    Procedure (see Excalibur tallies 4 and 6):
      1. Σ_x(E) = R_x(E) / φ(E)               energy-resolved macroscopic XS,
                                              reaction-rate spectrum / flux
                                              spectrum, bin by bin.
      2. Flux-average over energy to a single value:
             Σ̄_x = Σ_g Σ_x(E_g) φ_g / Σ_g φ_g  =  Σ_g R_x,g / Σ_g φ_g
                 = (energy-integrated rate) / (energy-integrated flux)

    The second equality is why this is robust: empty (zero-flux) bins drop
    out of both sums, so there are no 0/0 artefacts. Both tallies share the
    salt-box cell and the energy grid, so the box volume and the source
    normalization cancel — Σ̄ comes out in cm⁻¹, independent of SOURCE_RATE.

    Because the reaction rate is tallied per nuclide, this is the *partial*
    macroscopic XS from that nuclide alone (Σ̄_x = N_nuclide · σ̄_x). Divide by
    the nuclide number density (atom/b-cm) for the microscopic σ̄ in barns.

    Returns dict(capture=Σ̄_(n,γ), fission=Σ̄_fission) in cm⁻¹, or None if the
    nuclide is absent or the total flux is zero.
    """
    df_n = df_rr[df_rr["nuclide"] == nuclide]
    if df_n.empty:
        return None

    cap = df_n[df_n["score"] == "(n,gamma)"]["mean"].values
    fis = df_n[df_n["score"] == "fission"]["mean"].values

    tot_flux = df_flux["mean"].values.sum()
    if tot_flux <= 0.0:
        return None

    return dict(capture=cap.sum() / tot_flux,
                fission=fis.sum() / tot_flux)


def get_u238_macro_xs(case_dir, nuclide="U238"):
    """
    Read the flux spectrum (tally 4) and the U capture/fission spectrum
    (tally 6) from the latest statepoint in *case_dir* and return the
    flux-averaged macroscopic cross sections for *nuclide*.

    Returns dict(capture=…, fission=…) in cm⁻¹, or None when the statepoint
    or the required tallies are missing — e.g. air-background cases, which
    carry no uranium (tally 6 is only written when fertile material is
    present), or results produced before the flux-spectrum tally was added.
    """
    sp_files = sorted(case_dir.glob("statepoint.*.h5"))
    if not sp_files:
        return None

    sp = openmc.StatePoint(str(sp_files[-1]))
    try:
        t_flux = sp.get_tally(name=FLUX_TALLY)
        t_rr   = sp.get_tally(name=URANIUM_SPECTRUM_TALLY)
    except (LookupError, KeyError):
        return None

    return _macro_xs_from_dataframes(
        t_flux.get_pandas_dataframe(),
        t_rr.get_pandas_dataframe(),
        nuclide=nuclide,
    )


def write_summary_csv(csv_path="summary.csv"):
    """
    Write a CSV with neutron-flux metrics at the mouth and at 1 m
    (salt-box front face) for every case, plus the flux-averaged U-238
    macroscopic capture and fission cross sections in the salt box.

    The U-238 cross sections are also printed to the console as the table
    is built.  They are blank for the air-background cases (no uranium).

    Columns:
        case, mode, salt_type, fertile_mol_pct,
        mouth_total, mouth_sub0.8MeV, mouth_gt10MeV,
        1m_total,    1m_sub0.8MeV,    1m_gt10MeV,
        U238_capture_xs_per_cm, U238_fission_xs_per_cm
    """
    import csv

    header = [
        "case", "mode", "salt_type", "fertile_mol_pct",
        "mouth_total", "mouth_sub0.8MeV", "mouth_gt10MeV",
        "1m_total",    "1m_sub0.8MeV",    "1m_gt10MeV",
        "U238_capture_xs_per_cm", "U238_fission_xs_per_cm",
    ]

    # Build the same case list as the sweep
    cases = []
    for mode in MODES:
        for stype in SALT_TYPES:
            sl = SALT_LABELS[stype]
            fl = FERT_LABELS[stype]
            for fpct in FERTILE_PCTS:
                dname = case_dir_name(mode, sl, fl, fpct)
                cases.append((dname, mode, sl, fpct))
        cases.append((air_dir_name(mode), mode, "air", 0.0))

    # Console header for the flux-averaged U-238 macroscopic cross sections
    print("  U-238 flux-averaged macroscopic cross sections in salt box "
          "(Σ̄ = rate / flux):")
    print(f"    {'case':<48}  {'Σ_capture [1/cm]':>17}  {'Σ_fission [1/cm]':>17}")
    print(f"    {'-'*48}  {'-'*17}  {'-'*17}")

    rows = []
    for dname, mode, slabel, fpct in cases:
        cdir = BASE_DIR / dname

        mouth = integrate_bands(cdir, MOUTH_TALLY)
        front = integrate_bands(cdir, FRONT_TALLY)

        if mouth is None or front is None:
            print(f"    {dname:<48}  {'(no statepoint)':>17}")
            continue

        # Flux-averaged U-238 macroscopic cross sections (None for air cases)
        xs = get_u238_macro_xs(cdir, nuclide="U238")
        if xs is None:
            cap_csv = fis_csv = ""
            cap_txt = fis_txt = "n/a"
        else:
            cap_csv = f"{xs['capture']:.6e}"
            fis_csv = f"{xs['fission']:.6e}"
            cap_txt = cap_csv
            fis_txt = fis_csv

        print(f"    {dname:<48}  {cap_txt:>17}  {fis_txt:>17}")

        rows.append([
            dname, mode, slabel, f"{fpct:.1f}",
            f"{mouth['total']:.6e}",
            f"{mouth['sub_0p8MeV']:.6e}",
            f"{mouth['gt_10MeV']:.6e}",
            f"{front['total']:.6e}",
            f"{front['sub_0p8MeV']:.6e}",
            f"{front['gt_10MeV']:.6e}",
            cap_csv, fis_csv,
        ])

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"\n  Saved {csv_path}  ({len(rows)} rows)")


# ====================================================================
#  Main
# ====================================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Post-process Excalibur parametric sweep results."
    )
    parser.add_argument(
        "-s", nargs="+", type=float, default=[20.0], metavar="CM",
        help="Salt box length(s) in cm to process (default: 20)",
    )
    args = parser.parse_args()

    print("Plotting exit-face spectra from OpenMC results\n")

    for sbl in args.s:
        # Update module-level tag so case_dir_name / air_dir_name pick it up
        LENGTH_TAG = f"_{sbl:g}cm"

        print(f"{'#'*60}")
        print(f"  Salt box length = {sbl:g} cm  (tag: {LENGTH_TAG})")
        print(f"{'#'*60}\n")

        # ---- Set 1: raw spectra ----
        plot_raw("Neutron",
                 "Neutron spectrum — salt exit face",
                 f"fig_neutron_raw{LENGTH_TAG}.png")

        plot_raw("Gamma",
                 "Gamma spectrum — salt exit face",
                 f"fig_gamma_raw{LENGTH_TAG}.png",
                 xlim=(1e4, 20e6))

        # ---- Set 2: air-subtracted spectra ----
        plot_subtracted("Neutron",
                        "Neutron spectrum — salt exit face",
                        f"fig_neutron_sub{LENGTH_TAG}.png")

        plot_subtracted("Gamma",
                        "Gamma spectrum — salt exit face",
                        f"fig_gamma_sub{LENGTH_TAG}.png",
                        xlim=(1e4, 20e6))

        # ---- Summary CSV ----
        print()
        write_summary_csv(f"summary{LENGTH_TAG}.csv")

    print("\nDone.")