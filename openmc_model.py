"""
openmc_model.py - Dinh nghia mo hinh hinh hoc, vat lieu, nguon va tallies cho OpenMC
Chuan theo Paper 1 (Paper 10.1 - PNE 2022, Fig. 1 va Section 2.2):
- Nguon photon diem tai z = 0 cm (chieu doc theo truc +z)
- Ong chuan truc chi (Pb collimator) tu z = 10 cm den z = 50 cm (lo r = 0.5 cm, vo r = 3.0 cm)
- Mau thuy tinh (Glass S1-S8) dat tai z = 50.0 cm den 50.0 + x cm (r = 5.0 cm)
- Vung do F4 Tally equivalent tai z = 70.0 den 72.0 cm (r = 5.0 cm, V = 157.0796 cm3)
- Vo chi boc ngoai (Outer Pb Shield) tu z = 45 den 75 cm (r_in = 8 cm, r_out = 10 cm)
- Tallies: Tong thong luong (Total flux) va Thong luong dinh (Uncollided peak flux +/- 1%)
"""
import os
import openmc
try:
    openmc.config['cross_sections'] = None
except Exception:
    pass
import numpy as np
from pathlib import Path

# Duong dan thu vien hat nhan mac dinh trong WSL
DEFAULT_CROSS_SECTIONS = "/home/ht/nuclear_data/endfb-viii.0-hdf5/endfb-viii.0-hdf5/cross_sections.xml"

# ---------------------------------------------------------------------------
# Thanh phan vat lieu chuan 8 mau thuy tinh (S1 -> S8) - Tinh theo wt% (Paper 1 Table 1)
# ---------------------------------------------------------------------------
COMPOSITION = {
    # S1: 45 B2O3 - 10 SiO2 - 40 Li2O - 5 Al2O3 - 0 Gd2O3 (rho = 2.390 g/cm3)
    "s1": {
        "density": 2.390,
        "comp": {"Li": 0.18583, "B": 0.13976, "O": 0.60120, "Al": 0.02646, "Si": 0.04674}
    },
    # S2: 40 B2O3 - 10 SiO2 - 40 Li2O - 5 Al2O3 - 5 Gd2O3 (rho = 2.725 g/cm3)
    "s2": {
        "density": 2.725,
        "comp": {"Li": 0.18583, "B": 0.12423, "O": 0.57335, "Al": 0.02646, "Si": 0.04674, "Gd": 0.04338}
    },
    # S3: 35 B2O3 - 10 SiO2 - 40 Li2O - 5 Al2O3 - 10 Gd2O3 (rho = 2.908 g/cm3)
    "s3": {
        "density": 2.908,
        "comp": {"Li": 0.18583, "B": 0.10870, "O": 0.54550, "Al": 0.02646, "Si": 0.04674, "Gd": 0.08676}
    },
    # S4: 30 B2O3 - 10 SiO2 - 40 Li2O - 5 Al2O3 - 15 Gd2O3 (rho = 3.103 g/cm3)
    "s4": {
        "density": 3.103,
        "comp": {"Li": 0.18583, "B": 0.09317, "O": 0.51765, "Al": 0.02646, "Si": 0.04674, "Gd": 0.13014}
    },
    # S5: 60 B2O3 - 10 SiO2 - 10 CaO - 20 Gd2O3 (rho = 3.836 g/cm3)
    "s5": {
        "density": 3.836,
        "comp": {"B": 0.18635, "O": 0.52192, "Si": 0.04674, "Ca": 0.07147, "Gd": 0.17352}
    },
    # S6: 55 B2O3 - 10 SiO2 - 10 CaO - 25 Gd2O3 (rho = 3.909 g/cm3)
    "s6": {
        "density": 3.909,
        "comp": {"B": 0.17082, "O": 0.49407, "Si": 0.04674, "Ca": 0.07147, "Gd": 0.21690}
    },
    # S7: 50 B2O3 - 10 SiO2 - 10 CaO - 30 Gd2O3 (rho = 4.181 g/cm3)
    "s7": {
        "density": 4.181,
        "comp": {"B": 0.15529, "O": 0.46622, "Si": 0.04674, "Ca": 0.07147, "Gd": 0.26028}
    },
    # S8: 45 B2O3 - 10 SiO2 - 10 CaO - 35 Gd2O3 (rho = 4.411 g/cm3)
    "s8": {
        "density": 4.411,
        "comp": {"B": 0.13976, "O": 0.43837, "Si": 0.04674, "Ca": 0.07147, "Gd": 0.30366}
    },
}

DENSITY = {k: v["density"] for k, v in COMPOSITION.items()}


def build_materials(mat_key: str) -> tuple[openmc.Materials, openmc.Material, openmc.Material]:
    """Tao danh sach vat lieu gom Lead (Chi) va Mau thuy tinh (neu khong phai Blank)."""
    try:
        openmc.config['cross_sections'] = None
    except Exception:
        pass

    # Chi (Lead) cho ong chuan truc va vo chan ngoai
    pb = openmc.Material(name="Lead")
    pb.add_element("Pb", 1.0)
    pb.set_density("g/cm3", 11.34)

    sample_mat = None
    mats_list = [pb]

    mat_k = mat_key.lower()
    if mat_k != "blank" and mat_k in COMPOSITION:
        prop = COMPOSITION[mat_k]
        sample_mat = openmc.Material(name=mat_k.upper())
        sample_mat.set_density("g/cm3", prop["density"])
        for el, frac in prop["comp"].items():
            sample_mat.add_element(el, frac, percent_type="wo")
        mats_list.append(sample_mat)

    materials = openmc.Materials(mats_list)
    return materials, pb, sample_mat


def build_geometry(thickness_cm: float, pb_mat: openmc.Material, sample_mat: openmc.Material = None):
    """
    Xay dung hinh hoc mo phong chuan Paper 1:
    - Collimator: z = [10, 50], r = [0.5, 3.0]
    - Sample: z = [50, 50 + th], r = [0, 5.0]
    - Detector: z = [70, 72], r = [0, 5.0]
    - Outer Shield: z = [45, 75], r = [8.0, 10.0]
    """
    # Gioi han the gioi
    s_world = openmc.Sphere(x0=0.0, y0=0.0, z0=40.0, r=100.0, boundary_type="vacuum")

    # Cac mat phang Z
    z11 = openmc.ZPlane(z0=10.0)
    z12 = openmc.ZPlane(z0=50.0)
    z13 = openmc.ZPlane(z0=50.0 + float(thickness_cm))
    z14 = openmc.ZPlane(z0=45.0)
    z15 = openmc.ZPlane(z0=70.0)
    z16 = openmc.ZPlane(z0=72.0)
    z17 = openmc.ZPlane(z0=75.0)

    # Cac hinh tru R
    r20 = openmc.ZCylinder(r=0.5)
    r21 = openmc.ZCylinder(r=3.0)
    r22 = openmc.ZCylinder(r=5.0)
    r23 = openmc.ZCylinder(r=8.0)
    r24 = openmc.ZCylinder(r=10.0)

    # 1. Collimator Pb
    coll_pb = openmc.Cell(name="Collimator_Pb", fill=pb_mat, region=+z11 & -z12 & +r20 & -r21)

    # 2. Collimator bore (lo dan tia)
    coll_bore = openmc.Cell(name="Collimator_Bore", region=+z11 & -z12 & -r20)

    # 3. Mau thuy tinh (Sample slot)
    sample_cell = openmc.Cell(name="Sample", fill=sample_mat, region=+z12 & -z13 & -r22)

    # 4. Dau do F4 equivalent (Detector tally cell)
    detector_cell = openmc.Cell(name="Detector", region=+z15 & -z16 & -r22)

    # 5. Vo chi boc ngoai (Outer Pb Shield)
    shield_pb = openmc.Cell(name="Shield_Pb", fill=pb_mat, region=+z14 & -z17 & +r23 & -r24)

    # 6. Khoang trong ben trong vo chi (Chamber)
    chamber = openmc.Cell(
        name="Chamber",
        region=+z14 & -z17 & -r23 & ~coll_pb.region & ~coll_bore.region & ~sample_cell.region & ~detector_cell.region
    )

    # 7. Vung chan khong ngoai trong qua cau the gioi
    outer_void = openmc.Cell(
        name="OuterVoid",
        region=-s_world & ~shield_pb.region & ~chamber.region & ~coll_pb.region & ~coll_bore.region & ~sample_cell.region & ~detector_cell.region
    )

    geometry = openmc.Geometry([coll_pb, coll_bore, sample_cell, detector_cell, shield_pb, chamber, outer_void])
    return geometry, detector_cell


def build_settings(energy_kev: float, n_particles: int = 1000000, n_batches: int = 10) -> openmc.Settings:
    """Cau hinh Settings cho OpenMC: Fixed source photon transport."""
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.batches = int(n_batches)
    settings.particles = int(n_particles // n_batches)

    # Nguon photon don nang tai z = 0 chieu doc truc +z
    e0_ev = float(energy_kev) * 1.0e3
    src = openmc.IndependentSource()
    src.space = openmc.stats.Point((0.0, 0.0, 0.0))
    src.angle = openmc.stats.Monodirectional((0.0, 0.0, 1.0))
    src.energy = openmc.stats.Discrete([e0_ev], [1.0])
    src.particle = "photon"
    settings.source = src

    # Bat van chuyen photon (gamma/X-ray)
    settings.photon_transport = True
    settings.electron_treatment = "ttb"  # Thick target bremsstrahlung
    settings.cutoff = {"energy_photon": 1000.0}  # Cutoff 1 keV

    return settings


def build_tallies(detector_cell: openmc.Cell, energy_kev: float) -> openmc.Tallies:
    """
    Xay dung 2 Tallies trong vung detector:
    1. Tong thong luong photon (Total flux)
    2. Thong luong dinh photon chua va cham (Uncollided peak flux trong khoang E0 +/- 1%)
    """
    e0_ev = float(energy_kev) * 1.0e3
    emin_peak = e0_ev * 0.990
    emax_peak = e0_ev * 1.010

    cell_filter = openmc.CellFilter(detector_cell)
    particle_filter = openmc.ParticleFilter(["photon"])
    energy_filter_peak = openmc.EnergyFilter([emin_peak, emax_peak])

    # 1. Total Flux Tally
    tally_tot = openmc.Tally(name="total_flux")
    tally_tot.filters = [cell_filter, particle_filter]
    tally_tot.scores = ["flux"]

    # 2. Peak Flux Tally (Uncollided)
    tally_peak = openmc.Tally(name="peak_flux")
    tally_peak.filters = [cell_filter, particle_filter, energy_filter_peak]
    tally_peak.scores = ["flux"]

    tallies = openmc.Tallies([tally_tot, tally_peak])
    return tallies


def export_openmc_model(output_dir: str, mat_key: str, energy_kev: float, thickness_cm: float,
                        n_particles: int = 1000000, n_batches: int = 10):
    """Xuat tron bo 4 file XML (materials, geometry, settings, tallies) vao thu muc job."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        openmc.config['cross_sections'] = None
    except Exception:
        pass

    materials, pb_mat, sample_mat = build_materials(mat_key)
    geometry, detector_cell = build_geometry(thickness_cm, pb_mat, sample_mat)
    settings = build_settings(energy_kev, n_particles, n_batches)
    tallies = build_tallies(detector_cell, energy_kev)

    materials.export_to_xml(out_path / "materials.xml")
    geometry.export_to_xml(out_path / "geometry.xml")
    settings.export_to_xml(out_path / "settings.xml")
    tallies.export_to_xml(out_path / "tallies.xml")
