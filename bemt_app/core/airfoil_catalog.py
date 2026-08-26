"""
airfoil_catalog.py
------------------
Master airfoil catalog, slug conversion utilities, online polar fetcher,
coordinate loader, and polar CSV parser.

All functions are pure Python / NumPy / Pandas — no Streamlit imports.
"""

import glob
import os
import urllib.request
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Master Catalog
# ---------------------------------------------------------------------------

MASTER_CATALOG: List[str] = [
    # Selig (High-lift / low-Re UAV & Rotorcraft)
    "S1210", "S1223", "S1223 RTL", "S8035", "S8036", "S8037", "S8052", "S9000",
    "SD7003", "SD7032", "SD7037", "SD7062", "SD7080", "SD8000",
    # NACA 4, 5, 6-digit Series
    "NACA 0006", "NACA 0009", "NACA 0012", "NACA 0015", "NACA 0018",
    "NACA 0021", "NACA 0024",
    "NACA 1408", "NACA 1410", "NACA 1412",
    "NACA 2412", "NACA 2415", "NACA 2418", "NACA 2421", "NACA 2424",
    "NACA 4412", "NACA 4415", "NACA 4418", "NACA 4421", "NACA 4424",
    "NACA 23012", "NACA 23015", "NACA 23018", "NACA 23021", "NACA 23024",
    "NACA 63-212", "NACA 63-412", "NACA 64-212", "NACA 64-415",
    # Eppler Series
    "E193", "E205", "E214", "E387", "E420", "E423", "E560",
    # MH / RG / Pylon Series
    "MH 32", "MH 45", "MH 60", "MH 114", "MH 115", "MH 120", "RG 15",
    # General Aviation & Classic
    "Clark Y", "Clark YH", "Gottingen 398", "Gottingen 535",
    "USA 35B", "FX 63-137", "FX 74-CL5-140",
]


# ---------------------------------------------------------------------------
# 2. Name ↔ Slug Helpers
# ---------------------------------------------------------------------------

def to_airfoiltools_slug(name: str) -> str:
    """Map a display name to an airfoiltools.com database slug."""
    clean = name.lower().strip().replace(" ", "").replace("-", "").replace("_", "")

    known_mappings = {
        "clarky": "clarky-il",
        "clarkyh": "clarkyh-il",
        "gottingen398": "goe398-il",
        "gottingen535": "goe535-il",
        "goe398": "goe398-il",
        "goe535": "goe535-il",
        "usa35b": "usa35b-il",
        "s1223rtl": "s1223rtl-il",
        "fx63137": "fx63137-il",
        "fx74cl5140": "fx74cl5140-il",
    }
    if clean in known_mappings:
        return known_mappings[clean]

    if clean.startswith("naca"):
        digits = clean.replace("naca", "")
        return f"n{digits}-il"

    if clean.endswith("il") or clean.endswith("sa"):
        return clean
    return f"{clean}-il"


def normalize_name(name: str) -> str:
    return to_airfoiltools_slug(name).replace("-il", "").replace("-sa", "")


def filter_airfoil_catalog(query: str, catalog: List[str]) -> List[str]:
    """Prefix-first, then substring match on the catalog."""
    q = query.lower().replace(" ", "").replace("-", "")
    if not q:
        return catalog
    prefix = [af for af in catalog if af.lower().replace(" ", "").replace("-", "").startswith(q)]
    subs = [
        af for af in catalog
        if q in af.lower().replace(" ", "").replace("-", "") and af not in prefix
    ]
    return prefix + subs


def get_all_available_airfoils(airfoil_dir: str = "airfoils") -> List[str]:
    """Return de-duplicated catalog merged with any locally cached polars."""
    found_files = (
        glob.glob("*.csv")
        + glob.glob("*.txt")
        + glob.glob(f"{airfoil_dir}/*.csv")
        + glob.glob(f"{airfoil_dir}/*/*.csv")
    )
    disk_airfoils: set = set()
    rev_map = {
        "n0012": "NACA 0012", "n0015": "NACA 0015",
        "clarky": "Clark Y", "clarkyh": "Clark YH",
        "goe398": "Gottingen 398", "goe535": "Gottingen 535",
        "s1223": "S1223", "s1210": "S1210", "e423": "E423",
    }
    for f in found_files:
        fname = os.path.basename(f).lower()
        if fname.startswith("xf-"):
            p = fname.split("-")
            if len(p) >= 3:
                code = p[1].lower()
                disp_name = rev_map.get(code, p[1].upper())
                disk_airfoils.add(disp_name)

    seen_slugs: set = set()
    deduped: List[str] = []
    for item in (MASTER_CATALOG + list(disk_airfoils)):
        slug = to_airfoiltools_slug(item)
        if slug not in seen_slugs:
            seen_slugs.add(slug)
            deduped.append(item)
    return sorted(deduped)


# ---------------------------------------------------------------------------
# 3. Online Fetcher & Local Cache
# ---------------------------------------------------------------------------

def ensure_airfoil_data(airfoil_name: str, airfoil_dir: str = "airfoils") -> None:
    """Download coordinate and polar files from airfoiltools.com if not cached."""
    if airfoil_name in ("Knight & Hefner (1937)", "Knight & Hefner Analytical"):
        return  # analytical model — no files needed

    os.makedirs(airfoil_dir, exist_ok=True)
    slug = to_airfoiltools_slug(airfoil_name)
    target_dir = os.path.join(airfoil_dir, slug)
    os.makedirs(target_dir, exist_ok=True)

    # 1. Fetch coordinate file (.dat)
    dat_path = os.path.join(target_dir, f"{slug}.dat")
    if not os.path.exists(dat_path):
        try:
            url = f"http://airfoiltools.com/airfoil/seligdatfile?airfoil={slug}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                content = resp.read().decode("utf-8")
                if len(content) > 100 and "<html>" not in content.lower():
                    with open(dat_path, "w") as fh:
                        fh.write(content)
        except Exception:
            pass

    # 2. Fetch polars across standard Re spectrum
    re_list = [50000, 100000, 200000, 500000, 1000000]
    for re_v in re_list:
        for nc_str, nc_tag in [("", ""), ("-n5", "-n5")]:
            csv_path = os.path.join(target_dir, f"xf-{slug}-{re_v}{nc_tag}.csv")
            if not os.path.exists(csv_path):
                try:
                    url = f"http://airfoiltools.com/polar/csv?polar=xf-{slug}-{re_v}{nc_str}"
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        content = resp.read().decode("utf-8")
                        if "alpha" in content.lower() and "cl" in content.lower():
                            with open(csv_path, "w") as fh:
                                fh.write(content)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# 4. Coordinate File Parser
# ---------------------------------------------------------------------------

def load_airfoil_coords(
    airfoil_name: str,
    search_dirs: Tuple[str, ...] = ("airfoils", "."),
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x, y) coordinate arrays for the given airfoil."""
    ensure_airfoil_data(airfoil_name)
    slug = to_airfoiltools_slug(airfoil_name)
    slug_base = slug.replace("-il", "").replace("-sa", "")

    all_files: List[str] = []
    for d in search_dirs:
        if os.path.exists(d):
            all_files.extend(glob.glob(os.path.join(d, "*.dat")))
            all_files.extend(glob.glob(os.path.join(d, "*.txt")))
            all_files.extend(glob.glob(os.path.join(d, "*", "*.dat")))
            all_files.extend(glob.glob(os.path.join(d, "*", "*.txt")))

    for fpath in set(all_files):
        fname = os.path.basename(fpath).lower()
        if "xf-" in fname:
            continue
        if slug_base in fname or normalize_name(airfoil_name) in fname:
            raw_x, raw_y = [], []
            with open(fpath, "r") as fh:
                for line in fh.readlines()[1:]:
                    p = line.strip().split()
                    if len(p) == 2:
                        try:
                            raw_x.append(float(p[0]))
                            raw_y.append(float(p[1]))
                        except ValueError:
                            continue
            if len(raw_x) > 10:
                return np.array(raw_x), np.array(raw_y)

    # Fallback: NACA 0012 analytic
    beta = np.linspace(0, np.pi, 16)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = (
        5.0 * 0.12
        * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )
    )
    return np.concatenate([x[::-1], x[1:]]), np.concatenate([yt[::-1], -yt[1:]])


# ---------------------------------------------------------------------------
# 5. Polar CSV / TXT Parser
# ---------------------------------------------------------------------------

def parse_single_polar_file(
    filepath: str,
) -> Tuple[Optional[float], Optional[int], Optional[pd.DataFrame]]:
    """Parse an XFoil-style polar file and return (Re, Ncrit, DataFrame)."""
    try:
        with open(filepath, "r") as fh:
            lines = fh.readlines()

        re_val: Optional[float] = None
        ncrit_val: int = 9
        header_idx: Optional[int] = None

        for idx, line in enumerate(lines[:18]):
            lc = line.strip().lower()
            if "reynolds number" in lc:
                if "," in lc:
                    re_val = float(lc.split(",")[1].strip())
                elif "=" in lc:
                    p = lc.split("=")[1].split()[0]
                    re_val = float(p) * (1e6 if ("e6" in lc or "e 6" in lc) else 1.0)
            elif "re =" in lc:
                p = lc.split("re =")[1].split()[0]
                re_val = float(p) * (1e6 if ("e6" in lc or "e 6" in lc) else 1.0)
            if "ncrit" in lc:
                if "," in lc:
                    ncrit_val = int(float(lc.split(",")[1].strip()))
                elif "=" in lc:
                    ncrit_val = int(float(lc.split("=")[1].split()[0]))
            if "alpha" in lc and ("cl" in lc or "cd" in lc):
                header_idx = idx

        if re_val is None:
            fname = os.path.basename(filepath)
            for part in fname.replace(".csv", "").replace(".txt", "").split("-"):
                if part.isdigit() and int(part) >= 10000:
                    re_val = float(part)
                    break

        if "-n5" in os.path.basename(filepath).lower():
            ncrit_val = 5

        if header_idx is not None:
            if filepath.endswith(".csv"):
                df = pd.read_csv(filepath, skiprows=header_idx)
            else:
                df = pd.read_csv(filepath, skiprows=header_idx, sep=r"\s+", engine="python")
            df.columns = [c.strip().lower() for c in df.columns]
            if "alpha" in df.columns and "cl" in df.columns and "cd" in df.columns:
                df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
                df["cl"] = pd.to_numeric(df["cl"], errors="coerce")
                df["cd"] = pd.to_numeric(df["cd"], errors="coerce")
                df = df.dropna(subset=["alpha", "cl", "cd"])
                return (
                    re_val,
                    ncrit_val,
                    df[["alpha", "cl", "cd"]].rename(columns={"cl": "CL", "cd": "CD"}),
                )
    except Exception:
        pass
    return None, None, None
