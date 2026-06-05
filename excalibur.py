#!/usr/bin/env python3
"""
Excalibur OpenMC model — collimated & moderated modes.
=======================================================

Translated from the ZKP MCNP input deck and Jeon (2025) PhD dissertation.

Two operating modes (Jeon, Fig. 1.3):
  "collimated" — tapered slit through steel + bpoly; 14 MeV beam to target
  "moderated"  — steel cylinder whole; MCNP offset wedge in bpoly only

Target: salt box (configurable depth), front face 1 m from Excalibur mouth.

Two salt systems (+ air background):
  "fluoride"  → breeder: FLiBe (Li₂BeF₄)                    + fertile: UF₄
  "chloride"  → breeder: ClLiF (69.5 mol% LiCl–30.5% LiF) + fertile: UCl₃
  "air"       → salt box filled with air   (background spectra)

Usage:
    from excalibur_openmc import Excalibur

    ex = Excalibur("collimated", salt_type="fluoride", fertile_mol_pct=2.0)
    ex.export()                          # writes model_collimated.xml

    ex = Excalibur("moderated", salt_type="chloride", fertile_mol_pct=3.0,
                    u235_enrich=0.05)
    ex.export("my_model.xml")

CLI:
    python excalibur_openmc.py              # builds both modes
    python excalibur_openmc.py collimated   # single mode
"""

import math
import subprocess
import sys
from pathlib import Path
import openmc


class Excalibur:
    """
    Parameterised OpenMC model of the Excalibur DT-neutron source
    with a molten-salt target box.

    Parameters
    ----------
    mode : str
        'collimated' or 'moderated'
    salt_type : str
        'fluoride' (FLiBe + UF₄), 'chloride' (ClLiF + UCl₃),
        or 'air' (empty box for background spectra).
    fertile_mol_pct : float
        Mole-percent fertile compound in the salt (0 = pure breeder).
    u235_enrich : float
        Atom fraction U-235 in uranium (default 0.0072 = natural).
    run_mode : str
        'fixed source' (default) — DT generator at origin, 13.7 MeV,
            90° forward cone toward +X.  Results normalised per source
            neutron (scale by total_rate × 0.1464 for absolute units).
        'eigenvalue' — fission source in salt box; solves for k-eff.
            Only meaningful when the salt contains fissile material.
    salt_box_length : float
        Depth of the salt box along the beam axis (X) in cm.
        Default 10.0 cm.  Front face stays at x = 160.96 cm.
    """

    def __init__(self, mode="collimated", salt_type="fluoride",
                 fertile_mol_pct=0.0, u235_enrich=0.0072,
                 run_mode="fixed source", salt_box_length=10.0):

        if mode not in ("collimated", "moderated"):
            raise ValueError(f"mode must be 'collimated' or 'moderated', got {mode!r}")
        if salt_type not in ("fluoride", "chloride", "air"):
            raise ValueError(f"salt_type must be 'fluoride', 'chloride', or 'air', got {salt_type!r}")
        if run_mode not in ("fixed source", "eigenvalue"):
            raise ValueError(f"run_mode must be 'fixed source' or 'eigenvalue', got {run_mode!r}")
        if salt_box_length <= 0:
            raise ValueError(f"salt_box_length must be positive, got {salt_box_length}")

        self.mode            = mode
        self.salt_type       = salt_type
        self.fertile_mol_pct = fertile_mol_pct
        self.u235_enrich     = u235_enrich
        self.run_mode        = run_mode
        self.salt_box_length = salt_box_length

        # Populated by the _build methods
        self.cells = {}
        self.surfs = {}

        self._build_materials()
        self._build_geometry()
        self._build_settings()
        self._build_tallies()
        self._build_plots()

        self.model = openmc.Model(
            materials = self.materials,
            geometry  = self.geometry,
            settings  = self.settings,
            tallies   = self.tallies,
            plots     = self.plots,
        )

    # ================================================================
    #  MATERIALS
    # ================================================================

    def _build_materials(self):

        self.air = openmc.Material(name="Air")
        self.air.set_density("atom/b-cm", 5.0269e-05)
        self.air.add_nuclide("H1",   2.5956e-07)
        self.air.add_element("C",    7.4820e-09)
        self.air.add_nuclide("N14",  3.9127e-05)
        self.air.add_nuclide("O16",  1.0642e-05)
        self.air.add_element("Ar",   2.3230e-07)

        self.steel = openmc.Material(name="Carbon Steel")
        self.steel.set_density("g/cm3", 7.85)
        self.steel.add_element("C",  0.026,  "wo")
        self.steel.add_element("Si", 0.400,  "wo")
        self.steel.add_element("P",  0.040,  "wo")
        self.steel.add_element("S",  0.050,  "wo")
        self.steel.add_element("Fe", 99.284, "wo")
        self.steel.add_element("Cu", 0.200,  "wo")

        self.bpoly = openmc.Material(name="Borated Polyethylene")
        self.bpoly.set_density("g/cm3", 1.08)
        self.bpoly.add_element("C",   0.815459, "wo")
        self.bpoly.add_nuclide("B10", 0.009950, "wo")
        self.bpoly.add_nuclide("B11", 0.040050, "wo")
        self.bpoly.add_nuclide("H1",  0.134541, "wo")
        self.bpoly.add_s_alpha_beta("c_H_in_CH2")

        # ---- Salt box fill ---------------------------------------------------
        #
        # Strategy: define breeder and fertile as independent Materials with
        # room-temperature solid densities, then combine them at the desired
        # molar ratio via openmc.Material.mix_materials().
        #
        # Molar → atom-fraction conversion for mix_materials('ao'):
        #   ao_i = f_i × (atoms per formula unit)_i / Σ_j f_j × (atoms)_j
        #
        if self.salt_type == "air":
            # Background case: salt box is just air
            self.salt = self.air

        else:
            f   = self.fertile_mol_pct / 100.0   # mole fraction of fertile
            enr = self.u235_enrich

            if self.salt_type == "fluoride":
                # -- Breeder: FLiBe = 2(LiF)·BeF₂ = Li₂BeF₄  (7 atoms/f.u.)
                breeder = openmc.Material(name="FLiBe")
                breeder.set_density("g/cm3", 2.214)       # RT solid
                breeder.add_element("Li", 2.0)
                breeder.add_nuclide("Be9", 1.0)
                breeder.add_nuclide("F19", 4.0)
                n_atoms_breeder = 7

                # -- Fertile: UF₄  (5 atoms/f.u.)
                fertile = openmc.Material(name="UF4")
                fertile.set_density("g/cm3", 6.70)        # RT solid
                fertile.add_nuclide("U235", enr)
                fertile.add_nuclide("U238", 1.0 - enr)
                fertile.add_nuclide("F19", 4.0)
                n_atoms_fertile = 5

            else:  # chloride
                # -- Breeder: ClLiF = 69.5 mol% LiCl + 30.5 mol% LiF
                #    (BABY eutectic, natural Cl enrichment)
                #    Both salts have 2 atoms/molecule → ao fracs = mol fracs.
                licl = openmc.Material(name="LiCl")
                licl.set_density("g/cm3", 2.068)          # RT solid
                licl.add_element("Li", 1.0)
                licl.add_element("Cl", 1.0)               # natural enrichment

                lif = openmc.Material(name="LiF")
                lif.set_density("g/cm3", 2.635)           # RT solid
                lif.add_element("Li", 1.0)
                lif.add_element("F", 1.0)

                breeder = openmc.Material.mix_materials(
                    [licl, lif], [0.695, 0.305], "ao",
                )
                breeder.name = "ClLiF (69.5% LiCl – 30.5% LiF)"
                n_atoms_breeder = 2   # both components are diatomic

                # -- Fertile: UCl₃  (4 atoms/f.u., natural Cl)
                fertile = openmc.Material(name="UCl3")
                fertile.set_density("g/cm3", 5.51)        # RT solid
                fertile.add_nuclide("U235", enr)
                fertile.add_nuclide("U238", 1.0 - enr)
                fertile.add_element("Cl", 3.0)            # natural enrichment
                n_atoms_fertile = 4

            # -- Mix breeder + fertile at the target molar ratio ----
            if f > 0:
                # Convert mol fractions → atom fractions for 'ao' mixing
                ao_breeder = (1.0 - f) * n_atoms_breeder
                ao_fertile = f * n_atoms_fertile
                ao_total   = ao_breeder + ao_fertile

                self.salt = openmc.Material.mix_materials(
                    [breeder, fertile],
                    [ao_breeder / ao_total, ao_fertile / ao_total],
                    "ao",
                )
                self.salt.name = (
                    f"{breeder.name} + "
                    f"{self.fertile_mol_pct:.1f} mol% {fertile.name}"
                )
            else:
                # Pure breeder, no fertile
                self.salt = breeder

        # Build materials list (avoid duplicates when salt is air)
        mat_list = [self.air, self.steel, self.bpoly]
        if self.salt is not self.air:
            mat_list.append(self.salt)
        self.materials = openmc.Materials(mat_list)

    # ================================================================
    #  GEOMETRY
    # ================================================================

    def _build_geometry(self):

        # ---- Shared surfaces ----
        cs_cyl = openmc.ZCylinder(r=40.0,  name="CS outer R")
        cs_top = openmc.ZPlane(z0=30.0,    name="CS top")
        cs_bot = openmc.ZPlane(z0=-30.0,   name="CS bottom")

        bp_xlo = openmc.XPlane(x0=-60.96,  name="BPoly -X")
        bp_xhi = openmc.XPlane(x0= 60.96,  name="BPoly +X")
        bp_ylo = openmc.YPlane(y0=-60.96,  name="BPoly -Y")
        bp_yhi = openmc.YPlane(y0= 60.96,  name="BPoly +Y")
        bp_zlo = openmc.ZPlane(z0=-40.64,  name="BPoly -Z")
        bp_zhi = openmc.ZPlane(z0= 40.64,  name="BPoly +Z")

        tgt_xlo = openmc.XPlane(x0=160.96, name="Salt box -X (face)")
        tgt_xhi = openmc.XPlane(x0=160.96 + self.salt_box_length,
                                name="Salt box +X")
        tgt_ylo = openmc.YPlane(y0=-5.0,   name="Salt box -Y")
        tgt_yhi = openmc.YPlane(y0= 5.0,   name="Salt box +Y")
        tgt_zlo = openmc.ZPlane(z0=-5.0,   name="Salt box -Z")
        tgt_zhi = openmc.ZPlane(z0= 5.0,   name="Salt box +Z")

        room_xlo = openmc.XPlane(x0=-400.0, boundary_type="vacuum")
        room_xhi = openmc.XPlane(x0= 400.0, boundary_type="vacuum")
        room_ylo = openmc.YPlane(y0=-340.0, boundary_type="vacuum")
        room_yhi = openmc.YPlane(y0= 340.0, boundary_type="vacuum")
        room_zlo = openmc.ZPlane(z0=-84.0,  boundary_type="vacuum")
        room_zhi = openmc.ZPlane(z0= 400.0, boundary_type="vacuum")

        src_sph = openmc.Sphere(r=0.5, name="DT source void")

        # ---- Expose key surfaces for tallies ----
        self.surfs["salt_front"] = tgt_xlo
        self.surfs["salt_back"]  = tgt_xhi
        self.surfs["mouth"]      = bp_xhi

        # ---- Compound regions ----
        cs_cylinder = -cs_cyl & +cs_bot & -cs_top
        bp_box      = +bp_xlo & -bp_xhi & +bp_ylo & -bp_yhi & +bp_zlo & -bp_zhi
        tgt_box     = +tgt_xlo & -tgt_xhi & +tgt_ylo & -tgt_yhi & +tgt_zlo & -tgt_zhi
        room_box    = +room_xlo & -room_xhi & +room_ylo & -room_yhi & +room_zlo & -room_zhi

        # ---- Source void ----
        src_cell = openmc.Cell(name="DT source void", fill=self.air)
        src_cell.region = -src_sph
        self.cells["source"] = src_cell

        # ---- Mode-dependent aperture ----
        if self.mode == "collimated":
            # 2″ tall, 17.5° total slit through BOTH steel and bpoly
            _slope_y = math.tan(math.radians(17.5 / 2.0))

            slit_right  = openmc.Plane(a=-_slope_y, b=1.0, c=0.0, d=0.0,
                                       name="Slit right")
            slit_left   = openmc.Plane(a= _slope_y, b=1.0, c=0.0, d=0.0,
                                       name="Slit left")
            slit_ztop   = openmc.ZPlane(z0= 2.54, name="Slit +Z")
            slit_zbot   = openmc.ZPlane(z0=-2.54, name="Slit -Z")
            slit_xstart = openmc.XPlane(x0=0.0,   name="Slit x=0")

            slit_region = (
                -slit_right & +slit_left
                & +slit_zbot & -slit_ztop
                & +slit_xstart & -bp_xhi
            )

            slit_cell = openmc.Cell(name="Collimator slit (air)",
                                    fill=self.air)
            slit_cell.region = slit_region & +src_sph
            self.cells["wedge"] = slit_cell

            cs_cell = openmc.Cell(name="Carbon steel", fill=self.steel)
            cs_cell.region = cs_cylinder & ~slit_region & +src_sph
            self.cells["steel"] = cs_cell

            bp_cell = openmc.Cell(name="Borated polyethylene", fill=self.bpoly)
            bp_cell.region = bp_box & ~cs_cylinder & ~slit_region & +src_sph
            self.cells["bpoly"] = bp_cell

        else:  # moderated
            # MCNP offset-plane wedge in bpoly only; steel whole
            wedge_upper  = openmc.Plane(a=0.27722, b= 1.0, c=0.0, d=41.5086,
                                        name="Mod wedge upper (MCNP 107)")
            wedge_lower  = openmc.Plane(a=0.27722, b=-1.0, c=0.0, d=41.5086,
                                        name="Mod wedge lower (MCNP 108)")
            wedge_ztop   = openmc.ZPlane(z0= 27.474, name="Mod wedge +Z")
            wedge_zbot   = openmc.ZPlane(z0=-27.474, name="Mod wedge -Z")
            wedge_xstart = openmc.XPlane(x0=10.686,  name="Mod wedge x-start")

            wedge_region = (
                -wedge_upper & -wedge_lower
                & +cs_cyl
                & +wedge_xstart & -bp_xhi
                & +wedge_zbot & -wedge_ztop
            )

            wedge_cell = openmc.Cell(name="Moderator wedge (air)",
                                     fill=self.air)
            wedge_cell.region = wedge_region & +src_sph
            self.cells["wedge"] = wedge_cell

            cs_cell = openmc.Cell(name="Carbon steel", fill=self.steel)
            cs_cell.region = cs_cylinder & +src_sph
            self.cells["steel"] = cs_cell

            bp_cell = openmc.Cell(name="Borated polyethylene", fill=self.bpoly)
            bp_cell.region = bp_box & ~cs_cylinder & ~wedge_region & +src_sph
            self.cells["bpoly"] = bp_cell

        # ---- Salt box ----
        tgt_cell = openmc.Cell(name="Salt box", fill=self.salt)
        tgt_cell.region = tgt_box
        self.cells["target"] = tgt_cell

        # ---- Room air ----
        air_region = room_box & +src_sph
        for c in [cs_cell, bp_cell, tgt_cell]:
            air_region = air_region & ~c.region
        air_region = air_region & ~self.cells["wedge"].region

        air_cell = openmc.Cell(name="Room air", fill=self.air)
        air_cell.region = air_region
        self.cells["air"] = air_cell

        # ---- Graveyard ----
        graveyard = openmc.Cell(name="Graveyard")
        graveyard.region = ~room_box
        self.cells["graveyard"] = graveyard

        # ---- Assemble ----
        root = openmc.Universe(cells=list(self.cells.values()))
        self.geometry = openmc.Geometry(root)

    # ================================================================
    #  SETTINGS
    # ================================================================

    def _build_settings(self):
        s = openmc.Settings()
        s.photon_transport = True
        s.temperature      = {"default": 293.6}

        if self.run_mode == "fixed source":
            # ---- Fixed source: DT generator at the origin ----
            #   13.7 MeV neutrons (lab-frame), emitted into a 90° total
            #   cone (±45° half-angle) aimed in +X toward the target.
            #   This covers the full collimator acceptance and saves
            #   compute vs. isotropic 4π emission.
            #
            #   Solid-angle fraction sampled:
            #     Ω/4π = (1 − cos 45°)/2 ≈ 14.6 %
            #   When scaling to absolute rates, use:
            #     effective_source_rate = total_rate × 0.1464
            #     (i.e. 8.2e8 × 0.1464 ≈ 1.20e8  n/s into the cone)
            s.run_mode = "fixed source"
            s.batches    = 100
            s.particles  = int(1e5)      # per batch

            s.source = openmc.IndependentSource(
                space  = openmc.stats.Point((0.0, 0.0, 0.0)),
                angle  = openmc.stats.PolarAzimuthal(
                    mu  = openmc.stats.Uniform(math.cos(math.radians(45.0)), 1.0),
                    phi = openmc.stats.Uniform(0.0, 2.0 * math.pi),
                    reference_uvw = (1.0, 0.0, 0.0),
                    reference_vwu = (0.0, 1.0, 0.0),
                ),
                energy = openmc.stats.Discrete([13.7e6], [1.0]),
            )

        else:
            # ---- Eigenvalue: fission source in the salt box ----
            #   Only meaningful when there is fissile material.
            s.run_mode = "eigenvalue"
            s.batches    = 250
            s.inactive   = 50
            s.particles  = 50_000

            s.source = openmc.IndependentSource(
                space=openmc.stats.Box(
                    lower_left  = (160.96, -5.0, -5.0),
                    upper_right = (160.96 + self.salt_box_length, 5.0, 5.0),
                ),
            )

        self.settings = s

    # ================================================================
    #  TALLIES
    # ================================================================

    def _build_tallies(self):
        """
        Tallies 1–5 are always present.
        Tallies 6–7 are added when the salt contains uranium.

          1. Neutron spectrum — salt exit face  (current)
          2. Gamma  spectrum — salt exit face   (current)
          3. Incident neutron flux — salt front face (current)
          4. Neutron flux spectrum — salt box   (track-length flux)
          5. On-axis neutron flux — Excalibur mouth (mesh flux)
          6. Energy-resolved (n,γ) and fission rates by U-235/U-238 in salt
          7. Total (energy-integrated) (n,γ) and fission rates by U-235/U-238
        """
        tallies = openmc.Tallies()

        # Shared 200-bin log energy grid for the neutron / flux tallies:
        # 1e-5 eV → 20 MeV
        n_bins = 200
        e_min, e_max = 1e-5, 20.0e6
        energy_bins = [e_min * (e_max / e_min) ** (i / n_bins)
                       for i in range(n_bins + 1)]
        e_filt = openmc.EnergyFilter(energy_bins)

        # Dedicated gamma grid: the same number of log bins packed into
        # 1e4 eV → 20 MeV.  There are essentially no photons below ~10 keV
        # here, so dropping that range concentrates the resolution where the
        # gammas actually are (~60 bins/decade vs. ~16 on the shared grid).
        # Note: photons below e_min_gamma fall outside the filter and are
        # not scored — intended, since that range is empty.
        n_bins_gamma = 200
        e_min_gamma  = 1.0e4
        gamma_bins = [e_min_gamma * (e_max / e_min_gamma) ** (i / n_bins_gamma)
                      for i in range(n_bins_gamma + 1)]
        e_filt_gamma = openmc.EnergyFilter(gamma_bins)

        n_filt = openmc.ParticleFilter(["neutron"])
        g_filt = openmc.ParticleFilter(["photon"])

        # Cell filter on the salt box — reused by the flux-spectrum tally
        # below and (if present) the uranium reaction-rate tallies.
        salt_cell_filt = openmc.CellFilter(self.cells["target"])

        # (1) Neutron spectrum — salt exit face
        t1 = openmc.Tally(name="Neutron spectrum — salt exit face")
        t1.filters = [openmc.SurfaceFilter(self.surfs["salt_back"]),
                      e_filt, n_filt]
        t1.scores  = ["current"]
        tallies.append(t1)

        # (2) Gamma spectrum — salt exit face  (finer grid, ≥ 10 keV)
        t2 = openmc.Tally(name="Gamma spectrum — salt exit face")
        t2.filters = [openmc.SurfaceFilter(self.surfs["salt_back"]),
                      e_filt_gamma, g_filt]
        t2.scores  = ["current"]
        tallies.append(t2)

        # (3) Incident neutron flux — salt front face
        t3 = openmc.Tally(name="Incident neutron flux — salt front face")
        t3.filters = [openmc.SurfaceFilter(self.surfs["salt_front"]),
                      e_filt, n_filt]
        t3.scores  = ["current"]
        tallies.append(t3)

        # (4) Neutron flux spectrum through the salt box.
        #   Track-length scalar flux averaged over the salt volume,
        #   energy-resolved on the shared grid.  Distinct from the surface
        #   currents above (1 and 3): those count neutrons crossing the box
        #   faces, whereas this captures the flux *inside* the salt — the
        #   quantity that drives reaction rates, dose, and spectral indices.
        #   Output is volume-integrated (particle-cm per source neutron);
        #   divide by the salt-box volume for flux per unit volume.
        t4 = openmc.Tally(name="Neutron flux spectrum — salt box")
        t4.filters = [salt_cell_filt, e_filt, n_filt]
        t4.scores  = ["flux"]
        tallies.append(t4)

        # (5) On-axis neutron flux at Excalibur mouth
        mouth_mesh = openmc.RegularMesh()
        mouth_mesh.lower_left  = (60.96, -1.0, -1.0)
        mouth_mesh.upper_right = (62.96,  1.0,  1.0)
        mouth_mesh.dimension   = (1, 1, 1)

        t5 = openmc.Tally(name="Neutron flux — Excalibur mouth (on-axis)")
        t5.filters = [openmc.MeshFilter(mouth_mesh), e_filt, n_filt]
        t5.scores  = ["flux"]
        tallies.append(t5)

        # (6)–(7) Uranium reaction rates in the salt box.
        #   Only added when uranium is present (salt ≠ air, fertile > 0),
        #   because tally.nuclides requires the nuclides to exist in the
        #   problem.
        if self.salt_type != "air" and self.fertile_mol_pct > 0:

            # (6) Energy-resolved capture & fission by nuclide
            t6 = openmc.Tally(name="U capture & fission spectrum — salt box")
            t6.filters  = [salt_cell_filt, e_filt, n_filt]
            t6.nuclides = ["U235", "U238"]
            t6.scores   = ["(n,gamma)", "fission"]
            tallies.append(t6)

            # (7) Total (energy-integrated) capture & fission by nuclide
            t7 = openmc.Tally(name="U capture & fission total — salt box")
            t7.filters  = [salt_cell_filt, n_filt]
            t7.nuclides = ["U235", "U238"]
            t7.scores   = ["(n,gamma)", "fission"]
            tallies.append(t7)

        self.tallies = tallies

    # ================================================================
    #  PLOTS
    # ================================================================

    def _build_plots(self):
        color_map = {
            self.air:          "lightyellow",
            self.steel: "slategray",
            self.bpoly:        "mediumseagreen",
        }
        if self.salt is not self.air:
            color_map[self.salt] = "darkorange"

        # Elevation (side) view — vertical XZ slice through the beam axis
        elevation = openmc.Plot()
        elevation.filename = f"plot_elevation_{self.mode}"
        elevation.basis    = "xz"
        elevation.origin   = (55.0, 0.0, 0.0)
        elevation.width    = (350.0, 150.0)
        elevation.pixels   = (1750, 750)
        elevation.color_by = "material"
        elevation.colors   = color_map

        # Plan (top-down) view — horizontal XY slice through the beam axis
        plan = openmc.Plot()
        plan.filename = f"plot_plan_{self.mode}"
        plan.basis    = "xy"
        plan.origin   = (55.0, 0.0, 0.0)
        plan.width    = (350.0, 200.0)
        plan.pixels   = (1750, 1000)
        plan.color_by = "material"
        plan.colors   = color_map

        self.plots = openmc.Plots([elevation, plan])

    # ================================================================
    #  EXPORT & RUN
    # ================================================================

    def export(self, case_dir=None):
        """Export model.xml into *case_dir* (created if needed)."""
        if case_dir is None:
            case_dir = "."
        self._case_dir = Path(case_dir)
        self._case_dir.mkdir(parents=True, exist_ok=True)
        model_path = self._case_dir / "model.xml"
        self.model.export_to_model_xml(str(model_path))
        print(f"  → {model_path}")
        return model_path

    def plot_geometry(self):
        """
        Generate the geometry plots (plan + elevation) for this case by
        running OpenMC in plotting mode.  Reads the <plots> defined in
        model.xml and writes plot_plan_<mode>.png and
        plot_elevation_<mode>.png into the case directory.
        """
        case_dir = getattr(self, "_case_dir", Path("."))
        print(f"  Plotting geometry (plan + elevation) in {case_dir} …")
        result = subprocess.run(
            ["openmc", "--plot", "model.xml"],
            cwd=str(case_dir),
        )
        if result.returncode != 0:
            print(f"  ⚠ Geometry plotting exited with code {result.returncode}")
        return result.returncode

    def run(self):
        """
        Plot the geometry, then run the OpenMC source calculation in the
        directory used by the last export() call.

        The plan + elevation geometry plots are regenerated every time.
        The source calculation is skipped when a statepoint already exists
        for this case, so re-running the sweep only fills in missing
        results (and refreshes the plots) without repeating transport.
        """
        case_dir = getattr(self, "_case_dir", Path("."))

        # Always (re)generate the plan + elevation geometry plots
        self.plot_geometry()

        # Skip the (expensive) source calculation if results already exist
        existing = sorted(case_dir.glob("statepoint.*.h5"))
        if existing:
            print(f"  ✓ Statepoint exists ({existing[-1].name}) — "
                  f"skipping source calculation")
            return 0

        print(f"  Running OpenMC in {case_dir} …")
        result = subprocess.run(
            ["openmc", "model.xml"],
            cwd=str(case_dir),
        )
        if result.returncode != 0:
            print(f"  ⚠ OpenMC exited with code {result.returncode}")
        else:
            print(f"  ✓ Done — results in {case_dir}")
        return result.returncode


# ====================================================================
#  CLI — parametric sweep
# ====================================================================

if __name__ == "__main__":

    import argparse
    from itertools import product

    parser = argparse.ArgumentParser(
        description="Excalibur parametric sweep over salt compositions and box lengths."
    )
    parser.add_argument(
        "-s", nargs="+", type=float, default=[10.0], metavar="CM",
        help="Salt box length(s) along the beam axis in cm (default: 10)",
    )
    args = parser.parse_args()

    base_dir = Path("OpenMC")
    base_dir.mkdir(exist_ok=True)

    modes        = ["collimated", "moderated"]
    salt_types   = ["fluoride",   "chloride"]
    fertile_pcts = [0.5, 5.0, 10.0]

    salt_labels = {"fluoride": "FLiBe", "chloride": "ClLiF"}
    fert_labels = {"fluoride": "UF4",   "chloride": "UCl3"}

    for sbl in args.s:
        length_tag = f"_{sbl:g}cm"

        # ---- Build case list per length ----
        cases = []
        for mode, stype, fpct in product(modes, salt_types, fertile_pcts):
            name = (f"{mode}_{salt_labels[stype]}"
                    f"_{fpct:.1f}mol{fert_labels[stype]}{length_tag}")
            cases.append((name, mode, stype, fpct))

        for mode in modes:
            cases.append((f"{mode}_air_background{length_tag}", mode, "air", 0.0))

        print(f"\n{'#'*60}")
        print(f"  Salt box length = {sbl:g} cm")
        print(f"  Running {len(cases)} cases → {base_dir}/")
        print(f"{'#'*60}\n")

        for i, (case_name, mode, stype, fpct) in enumerate(cases, 1):
            case_dir = base_dir / case_name

            print(f"{'='*60}")
            print(f"  [{i}/{len(cases)}]  {case_name}")
            print(f"{'='*60}")

            ex = Excalibur(mode=mode, salt_type=stype, fertile_mol_pct=fpct,
                           salt_box_length=sbl)
            ex.export(case_dir)
            ex.run()