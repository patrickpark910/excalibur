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


def _apply_ylim(ax, ylim):
    """
    Apply a y-axis limit (lo, hi) where either bound may be None to leave
    that side auto-scaled to the data.

    e.g. ylim=(1e0, None) pins the floor at 1 and lets the top follow the
    data; ylim=(None, 1e6) caps the top and auto-scales the floor.  When a
    bound is None the axis is first autoscaled to the data so the kept side
    reflects the data rather than a stale/default limit.
    """
    lo, hi = ylim
    if lo is not None and hi is not None:
        ax.set_ylim(lo, hi)
        return
    ax.relim()
    ax.autoscale(enable=True, axis="y")
    auto_lo, auto_hi = ax.get_ylim()
    ax.set_ylim(bottom=lo if lo is not None else auto_lo,
                top=hi if hi is not None else auto_hi)


def plot_raw(particle, tally_name, out_filename, xlim=None, ylim=None):
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
        if ylim is not None:
            _apply_ylim(ax, ylim)
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(out_filename, dpi=200)
    print(f"  Saved {out_filename}")
    plt.close(fig)


def plot_subtracted(particle, tally_name, out_filename, xlim=None, ylim=None):
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
        if ylim is not None:
            _apply_ylim(ax, ylim)
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

    Each reaction additionally carries its 1-sigma absolute uncertainty
    (key ``*_sd``, in cm^-1) and its relative uncertainty (key ``*_re``),
    propagated from the per-bin OpenMC tally standard deviations.
    """
    df_n = df_rr[df_rr["nuclide"] == nuclide]
    if df_n.empty:
        return None

    cap_df = df_n[df_n["score"] == "(n,gamma)"]
    fis_df = df_n[df_n["score"] == "fission"]

    cap    = cap_df["mean"].values
    cap_sd = cap_df["std. dev."].values
    fis    = fis_df["mean"].values
    fis_sd = fis_df["std. dev."].values

    flux    = df_flux["mean"].values
    flux_sd = df_flux["std. dev."].values

    tot_flux = flux.sum()
    if tot_flux <= 0.0:
        return None

    # Energy-integrated reaction rates and their 1-sigma.  The per-bin
    # standard deviations are summed in quadrature: the capture/fission bins
    # are dominated by rare, effectively uncorrelated events, so quadrature
    # tracks the integral's sampling error faithfully.
    R_cap    = cap.sum()
    R_cap_sd = float(np.sqrt(np.sum(cap_sd ** 2)))
    R_fis    = fis.sum()
    R_fis_sd = float(np.sqrt(np.sum(fis_sd ** 2)))

    # Total in-box flux and its 1-sigma (very well converged; small term).
    F_sd     = float(np.sqrt(np.sum(flux_sd ** 2)))
    rel_flux = F_sd / tot_flux

    def _ratio(R, R_sd):
        # Sigma_bar = R / phi.  Relative error of the ratio with R and phi
        # treated as independent.  The neglected (positive) R-phi correlation
        # makes this a mild upper bound, and rel_flux is small, so in practice
        # the reaction-rate sampling term dominates.
        sig = R / tot_flux
        if R > 0.0:
            rel = math.sqrt((R_sd / R) ** 2 + rel_flux ** 2)
        else:
            rel = float("inf")
        return sig, sig * rel, rel

    cap_xs, cap_xs_sd, cap_re = _ratio(R_cap, R_cap_sd)
    fis_xs, fis_xs_sd, fis_re = _ratio(R_fis, R_fis_sd)

    return dict(capture=cap_xs, capture_sd=cap_xs_sd, capture_re=cap_re,
                fission=fis_xs, fission_sd=fis_xs_sd, fission_re=fis_re)


def get_uranium_macro_xs(case_dir):
    """
    Flux-averaged macroscopic capture and fission cross sections in the salt
    box for U-235, U-238, and elemental uranium, from the latest statepoint
    in *case_dir*.

    Uses the flux spectrum (tally 4) and the per-nuclide U capture/fission
    spectrum (tally 6).  Each isotope value is Σ̄_x = Σ_g R_x,g / Σ_g φ_g.

    Elemental uranium is the sum of the two isotope contributions:

        Σ_x(U) = (R_x,U235 + R_x,U238) / φ = Σ_x(U235) + Σ_x(U238)

    which is exact, because macroscopic cross sections (and the reaction
    rates behind them) add over constituents.  The uranium here is at
    natural enrichment, so this is the natural elemental-U macroscopic
    cross section.

    Returns
    -------
    dict keyed by "U235", "U238", "U", each mapping to dict(capture=…,
    fission=…) in cm⁻¹, or None when the statepoint or the required tallies
    are missing (e.g. air-background cases, which carry no uranium).
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

    # Read each tally once; the per-nuclide split is just dataframe filtering.
    df_flux = t_flux.get_pandas_dataframe()
    df_rr   = t_rr.get_pandas_dataframe()

    u235 = _macro_xs_from_dataframes(df_flux, df_rr, "U235")
    u238 = _macro_xs_from_dataframes(df_flux, df_rr, "U238")
    if u235 is None or u238 is None:
        return None

    # Elemental U = U-235 + U-238.  Macroscopic cross sections and the
    # reaction rates behind them add over constituents, so the means sum
    # directly; the 1-sigma uncertainties combine in quadrature (the two
    # isotopes' rare-event reaction rates are treated as independent, which
    # is dominated by U-238 anyway at natural enrichment).
    def _combine(rxn):
        m  = u235[rxn] + u238[rxn]
        sd = math.sqrt(u235[rxn + "_sd"] ** 2 + u238[rxn + "_sd"] ** 2)
        re = (sd / m) if m > 0.0 else float("inf")
        return m, sd, re

    cap_m, cap_sd, cap_re = _combine("capture")
    fis_m, fis_sd, fis_re = _combine("fission")
    u_elem = dict(capture=cap_m, capture_sd=cap_sd, capture_re=cap_re,
                  fission=fis_m, fission_sd=fis_sd, fission_re=fis_re)

    return {"U235": u235, "U238": u238, "U": u_elem}


def _fmt_xs(val, rel):
    """Compact console cell: ``mean ± relative-1sigma%`` (5-char error field)."""
    if not math.isfinite(rel):
        pe = "  n/a"
    else:
        pct = 100.0 * rel
        if pct < 1.0:
            pe = "  <1%"
        elif pct >= 1000.0:
            pe = ">999%"
        else:
            pe = f"{pct:4.0f}%"
    return f"{val:.3e} ±{pe}"


def write_summary_csv(csv_path="summary.csv"):
    """
    Write a CSV with neutron-flux metrics at the mouth and at 1 m
    (salt-box front face) for every case, plus the flux-averaged
    macroscopic capture and fission cross sections in the salt box for
    U-235, U-238, and elemental uranium (U = U-235 + U-238).

    The cross sections are also printed to the console as the table is
    built.  They are blank for the air-background cases (no uranium).

    Columns:
        case, mode, salt_type, fertile_mol_pct,
        mouth_total, mouth_sub0.8MeV, mouth_gt10MeV,
        1m_total,    1m_sub0.8MeV,    1m_gt10MeV,
        U235_capture_xs_per_cm, U235_fission_xs_per_cm,
        U238_capture_xs_per_cm, U238_fission_xs_per_cm,
        U_capture_xs_per_cm,    U_fission_xs_per_cm,
        ...and the same six names with "_unc_" in place of "_" before
        "per_cm": the absolute 1-sigma statistical uncertainty (cm^-1) of
        each cross section.

    The console table additionally prints, for every cross section, its
    relative 1-sigma statistical error as "Σ̄ ± rel%", so poorly converged
    channels (e.g. dilute U-238 capture) are obvious at a glance.
    """
    import csv

    # (nuclide-key, reaction-key) in display / column order
    XS_ORDER = [
        ("U235", "capture"), ("U235", "fission"),
        ("U238", "capture"), ("U238", "fission"),
        ("U",    "capture"), ("U",    "fission"),
    ]
    XS_COLS = ["U235_cap", "U235_fis", "U238_cap", "U238_fis", "U_cap", "U_fis"]
    XS_CSV_HEADER = [
        "U235_capture_xs_per_cm", "U235_fission_xs_per_cm",
        "U238_capture_xs_per_cm", "U238_fission_xs_per_cm",
        "U_capture_xs_per_cm",    "U_fission_xs_per_cm",
    ]
    # Absolute 1-sigma uncertainty (cm^-1) paired with each XS column above.
    XS_UNC_HEADER = [
        "U235_capture_xs_unc_per_cm", "U235_fission_xs_unc_per_cm",
        "U238_capture_xs_unc_per_cm", "U238_fission_xs_unc_per_cm",
        "U_capture_xs_unc_per_cm",    "U_fission_xs_unc_per_cm",
    ]

    header = [
        "case", "mode", "salt_type", "fertile_mol_pct",
        "mouth_total", "mouth_sub0.8MeV", "mouth_gt10MeV",
        "1m_total",    "1m_sub0.8MeV",    "1m_gt10MeV",
        *XS_CSV_HEADER, *XS_UNC_HEADER,
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

    # Console header for the flux-averaged macroscopic cross sections
    print("  Flux-averaged macroscopic cross sections in salt box [1/cm]  "
          "(Σ̄ = rate / flux;  U = U235 + U238):")
    print("    (each cell:  Σ̄ ± relative 1σ statistical error)")
    print(f"    {'case':<44}" + "".join(f"  {c:>16}" for c in XS_COLS))
    print(f"    {'-'*44}" + "".join(f"  {'-'*16}" for _ in XS_COLS))

    rows = []
    for dname, mode, slabel, fpct in cases:
        cdir = BASE_DIR / dname

        mouth = integrate_bands(cdir, MOUTH_TALLY)
        front = integrate_bands(cdir, FRONT_TALLY)

        if mouth is None or front is None:
            print(f"    {dname:<44}  (no statepoint)")
            continue

        # Flux-averaged macroscopic XS for U-235, U-238, elemental U
        # (None for air cases, which carry no uranium)
        xs = get_uranium_macro_xs(cdir)
        if xs is None:
            xs_csv  = [""] * len(XS_ORDER)
            unc_csv = [""] * len(XS_ORDER)
            xs_txt  = ["n/a"] * len(XS_ORDER)
        else:
            vals = [xs[nuc][rxn]         for nuc, rxn in XS_ORDER]
            sds  = [xs[nuc][rxn + "_sd"] for nuc, rxn in XS_ORDER]
            res  = [xs[nuc][rxn + "_re"] for nuc, rxn in XS_ORDER]
            xs_csv  = [f"{v:.6e}" for v in vals]
            unc_csv = [f"{s:.6e}" for s in sds]
            xs_txt  = [_fmt_xs(v, r) for v, r in zip(vals, res)]

        print(f"    {dname:<44}" + "".join(f"  {t:>16}" for t in xs_txt))

        rows.append([
            dname, mode, slabel, f"{fpct:.1f}",
            f"{mouth['total']:.6e}",
            f"{mouth['sub_0p8MeV']:.6e}",
            f"{mouth['gt_10MeV']:.6e}",
            f"{front['total']:.6e}",
            f"{front['sub_0p8MeV']:.6e}",
            f"{front['gt_10MeV']:.6e}",
            *xs_csv, *unc_csv,
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
                 f"fig_neutron_raw{LENGTH_TAG}.png",
                 ylim=(1e0, None))

        plot_raw("Gamma",
                 "Gamma spectrum — salt exit face",
                 f"fig_gamma_raw{LENGTH_TAG}.png",
                 xlim=(1e4, 20e6),
                 ylim=(1e0, None))

        # ---- Set 2: air-subtracted spectra ----
        plot_subtracted("Neutron",
                        "Neutron spectrum — salt exit face",
                        f"fig_neutron_sub{LENGTH_TAG}.png",
                        ylim=(1e0, None))

        plot_subtracted("Gamma",
                        "Gamma spectrum — salt exit face",
                        f"fig_gamma_sub{LENGTH_TAG}.png",
                        xlim=(1e4, 20e6),
                        ylim=(1e0, None))

        # ---- Summary CSV ----
        print()
        write_summary_csv(f"summary{LENGTH_TAG}.csv")

    print("\nDone.")