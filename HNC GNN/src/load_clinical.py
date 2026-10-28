"""
M6a - Load and clean the real Head-Neck-PET-CT clinical spreadsheet.

Input : data/raw/INFOclinical_HN_Version2_30may2018.xlsx   (TCIA, CC BY 3.0)
Output: data/processed/clinical.csv      one clean row per patient
        results/table1_cohort.csv        cohort description (paper Table 1)
        results/clinical_cleaning_log.txt every correction made, for your methods section

Endpoints (Peng et al. 2024 use PFS as primary endpoint)
  PFS event = first of: locoregional recurrence (LR), distant metastasis (DM), death
  PFS time  = time to that first event; if no event, time to last follow-up
  OS  event = death;  OS time = time to death, else last follow-up
Times are converted from days to months (/30.44).

Run from the project root:   python -m src.load_clinical
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw" / "INFOclinical_HN_Version2_30may2018.xlsx"
CENTRES = ["CHUS", "HGJ", "CHUM", "HMR"]
DAYS_PER_MONTH = 30.44
log = []


def _norm_col(c):
    """Sheets spell headers inconsistently ('follow-up(days)' vs 'follow-up (days)', en-dash)."""
    c = " ".join(str(c).replace("\u2013", "-").split())
    return c.replace("follow-up(", "follow-up (")


def read_raw():
    xl = pd.ExcelFile(RAW)
    excluded = set(xl.parse("Excluded")["Patient #"].dropna().astype(str))
    excluded = {p for p in excluded if p.startswith("HN-")}
    frames = []
    for c in CENTRES:
        d = xl.parse(c)
        d.columns = [_norm_col(x) for x in d.columns]
        n0 = len(d)
        d = d[d["Patient #"].astype(str).str.startswith("HN-")]      # drop note rows
        log.append(f"{c}: {n0} rows -> {len(d)} patient rows (dropped footnote rows)")
        d.insert(1, "centre", c)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df[~df["Patient #"].isin(excluded)]
    log.append(f"Excluded-sheet patients {sorted(excluded)}: removed {before - len(df)} rows")
    return df


def _group(val, pattern):
    """'T4a' -> 'T4', 'N2c' -> 'N2', 'Tx'/'Mx'/missing -> NaN."""
    m = re.match(pattern, str(val).strip(), re.I)
    return m.group(0).upper() if m else np.nan


def clean(df):
    col = lambda s: [c for c in df.columns if c.startswith(s)][0]
    t_fu = df[col("Time - diagnosis to last follow-up")].astype(float)
    t_lr = df[col("Time - diagnosis to LR")].astype(float)
    t_dm = df[col("Time - diagnosis to DM")].astype(float)
    t_death = df[col("Time - diagnosis to Death")].astype(float)
    lr, dm, death = (df[c].astype(int) for c in ("Locoregional", "Distant", "Death"))

    # --- data corrections, each logged ------------------------------------------
    # DM recorded but no DM date: use the earliest other known event, else last follow-up
    for i in df.index[(dm == 1) & t_dm.isna()]:
        fallback = np.nanmin([t_death[i], t_lr[i], t_fu[i]])
        log.append(f"{df.at[i, 'Patient #']}: DM=1 without date -> DM time set to {fallback:.0f} d "
                   f"(earliest known event/follow-up; upper bound)")
        t_dm[i] = fallback
    n_late = int((death == 1).sum() - ((death == 1) & (t_death <= t_fu)).sum())
    log.append(f"{n_late} patients have death date after last clinic follow-up; "
               f"event dates are used as recorded")

    event_times = np.column_stack([t_lr.where(lr == 1), t_dm.where(dm == 1), t_death.where(death == 1)])
    any_event = (lr + dm + death) > 0
    pfs_days = np.where(any_event, np.nanmin(np.where(np.isnan(event_times), np.inf, event_times), 1), t_fu)
    os_days = np.where(death == 1, t_death, t_fu)

    hpv = df["HPV status"].map({"+": 1, "-": 0})
    out = pd.DataFrame({
        "patient_id": df["Patient #"].values,
        "centre": df["centre"].values,
        "pfs_time": np.round(pfs_days / DAYS_PER_MONTH, 2),
        "pfs_event": any_event.astype(int).values,
        "os_time": np.round(os_days / DAYS_PER_MONTH, 2),
        "os_event": death.values,
        "age": df["Age"].astype(float).values,
        "sex_male": (df["Sex"].str.upper() == "M").astype(int).values,
        "site": df["Primary Site"].str.lower().replace("unknown", np.nan).values,
        "t_stage": df["T-stage"].map(lambda v: _group(v, r"T[1-4]")).values,
        "n_stage": df["N-stage"].map(lambda v: _group(v, r"N[0-3]")).values,
        "hpv": hpv.values,                                   # NaN = not tested / unknown
        "chemo": (df["Therapy"] == "chemo radiation").astype(int).values,
        "followup_months": np.round(t_fu.values / DAYS_PER_MONTH, 2),
    })
    # Convenience aliases so M2-M5 (which expect `time`/`event`) run unchanged: primary endpoint = PFS
    out["time"], out["event"] = out["pfs_time"], out["pfs_event"]
    return out


def _med_iqr(s):
    q = s.quantile([0.25, 0.5, 0.75])
    return f"{q[0.5]:.0f} ({q[0.25]:.0f}-{q[0.75]:.0f})"


def _pct(mask, n):
    return f"{int(mask.sum())} ({100 * mask.sum() / n:.0f}%)"


def table1(c):
    """Cohort description by centre + test of whether centres differ."""
    from scipy.stats import chi2_contingency, kruskal
    rows = {}
    groups = {k: g for k, g in c.groupby("centre")}
    groups = {k: groups[k] for k in CENTRES} | {"All": c}
    for name, g in groups.items():
        n = len(g)
        rows[name] = {
            "Patients": n,
            "Age, median (IQR)": _med_iqr(g["age"]),
            "Male": _pct(g["sex_male"] == 1, n),
            "Oropharynx primary": _pct(g["site"] == "oropharynx", n),
            "HPV positive": _pct(g["hpv"] == 1, n),
            "HPV unknown": _pct(g["hpv"].isna(), n),
            "T3-T4": _pct(g["t_stage"].isin(["T3", "T4"]), n),
            "Node positive (N1-N3)": _pct(g["n_stage"].isin(["N1", "N2", "N3"]), n),
            "Chemoradiation": _pct(g["chemo"] == 1, n),
            "Follow-up months, median (IQR)": _med_iqr(g["followup_months"]),
            "PFS events": _pct(g["pfs_event"] == 1, n),
            "Deaths (OS events)": _pct(g["os_event"] == 1, n),
        }
    t = pd.DataFrame(rows)

    # p-values: do the four centres differ? (Kruskal-Wallis for continuous, chi-square for categorical)
    cc = c[c["centre"].isin(CENTRES)]
    p = {
        "Age, median (IQR)": kruskal(*[g["age"].dropna() for _, g in cc.groupby("centre")]).pvalue,
        "Follow-up months, median (IQR)": kruskal(*[g["followup_months"] for _, g in cc.groupby("centre")]).pvalue,
    }
    for label, series in {
        "Male": cc["sex_male"], "Oropharynx primary": cc["site"] == "oropharynx",
        "HPV positive": cc["hpv"] == 1, "HPV unknown": cc["hpv"].isna(),
        "T3-T4": cc["t_stage"].isin(["T3", "T4"]),
        "Node positive (N1-N3)": cc["n_stage"].isin(["N1", "N2", "N3"]),
        "Chemoradiation": cc["chemo"] == 1, "PFS events": cc["pfs_event"] == 1,
        "Deaths (OS events)": cc["os_event"] == 1,
    }.items():
        p[label] = chi2_contingency(pd.crosstab(cc["centre"], series))[1]
    t["p (centres differ)"] = pd.Series(p).map(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")
    return t.fillna("")


if __name__ == "__main__":
    if not RAW.exists():
        raise SystemExit(f"Put the spreadsheet here first: {RAW}")
    clin = clean(read_raw())
    (PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "results").mkdir(parents=True, exist_ok=True)
    clin.to_csv(PROJECT_ROOT / "data" / "processed" / "clinical.csv", index=False)
    t1 = table1(clin)
    t1.to_csv(PROJECT_ROOT / "results" / "table1_cohort.csv")
    (PROJECT_ROOT / "results" / "clinical_cleaning_log.txt").write_text("\n".join(log))

    print("Cleaning log:\n  " + "\n  ".join(log))
    pd.set_option("display.width", 200)
    print("\nTable 1 - cohort by centre\n")
    print(t1.to_string())
