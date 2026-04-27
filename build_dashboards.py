"""
build_dashboards.py

Pulls WHOOP biometric data from Supabase, computes 12 derived metrics,
and renders two Spanish HTML dashboards (Javier Nutriologo, Doctora Escobar Oncologa).

Usage:
    python build_dashboards.py
    python build_dashboards.py --days 365
    python build_dashboards.py --end-date 2026-04-25

Environment variables:
    SUPABASE_DB_URL    Required. Full postgres connection string.
                       Example: postgresql://postgres.xxxxx:PASSWORD@aws-1-us-east-2.pooler.supabase.com:6543/postgres

Inputs (in same directory):
    nutritionist_template.html    Template with __DATA_PLACEHOLDER__ token
    oncologist_template.html      Template with __DATA_PLACEHOLDER__ token

Outputs (in same directory):
    martinez_nutritionist_dashboard.html
    martinez_oncologist_dashboard.html
"""

import argparse
import json
import math
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean, stdev

import psycopg2
import psycopg2.extras


# =============================================================================
# CONFIGURATION
# =============================================================================

PATIENT_NAME = "Fernando Martinez"
HEIGHT_M = 1.69
WEIGHT_KG = 69.65
MAX_HR = 196
DEFAULT_LOOKBACK_DAYS = 365

# HR zone anchors (% of max HR). Standard 5-zone model.
HR_ZONES = {
    "Z1": (0.50, 0.60),
    "Z2": (0.60, 0.70),
    "Z3": (0.70, 0.80),
    "Z4": (0.80, 0.90),
    "Z5": (0.90, 1.00),
}

SCRIPT_DIR = Path(__file__).parent.resolve()


# =============================================================================
# DATABASE
# =============================================================================

def get_connection():
    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        sys.exit("ERROR: SUPABASE_DB_URL environment variable not set")
    return psycopg2.connect(db_url)


def fetch_data(conn, start_date, end_date):
    """Pull all 5 WHOOP tables filtered to the date window."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Cycles: daily strain and energy
    cur.execute("""
        SELECT
            cycle_start::date AS d,
            day_strain AS strain,
            kilojoule,
            average_heart_rate,
            max_heart_rate
        FROM whoop_cycles
        WHERE cycle_start::date BETWEEN %s AND %s
        ORDER BY d
    """, (start_date, end_date))
    cycles = cur.fetchall()

    # Recovery: HRV, RHR, SpO2, skin temp
    cur.execute("""
        SELECT
            cycle_start::date AS d,
            recovery_score,
            hrv_rmssd_milli AS hrv,
            resting_heart_rate AS rhr,
            spo2_percentage AS spo2,
            skin_temp_celsius AS skin_temp
        FROM whoop_recovery
        WHERE cycle_start::date BETWEEN %s AND %s
        ORDER BY d
    """, (start_date, end_date))
    recovery = cur.fetchall()

    # Sleep: stages, performance, efficiency
    cur.execute("""
        SELECT
            sleep_start::date AS d,
            total_in_bed_time_milli AS in_bed_ms,
            total_awake_time_milli AS awake_ms,
            total_light_sleep_time_milli AS light_ms,
            total_slow_wave_sleep_time_milli AS deep_ms,
            total_rem_sleep_time_milli AS rem_ms,
            sleep_performance_percentage AS performance,
            sleep_efficiency_percentage AS efficiency,
            sleep_end AS end_ts
        FROM whoop_sleep
        WHERE sleep_start::date BETWEEN %s AND %s
          AND nap = false
        ORDER BY d
    """, (start_date, end_date))
    sleep = cur.fetchall()

    # Workouts: sport, strain, HR, duration
    cur.execute("""
        SELECT
            workout_start::date AS d,
            sport_id,
            sport_name,
            strain,
            kilojoule,
            average_heart_rate,
            max_heart_rate,
            workout_start AS start_ts,
            workout_end AS end_ts,
            zone_zero_milli,
            zone_one_milli,
            zone_two_milli,
            zone_three_milli,
            zone_four_milli,
            zone_five_milli
        FROM whoop_workouts
        WHERE workout_start::date BETWEEN %s AND %s
        ORDER BY workout_start
    """, (start_date, end_date))
    workouts = cur.fetchall()

    cur.close()
    return {
        "cycles": cycles,
        "recovery": recovery,
        "sleep": sleep,
        "workouts": workouts,
    }


# =============================================================================
# HELPERS
# =============================================================================

def kj_to_kcal(kj):
    """WHOOP reports energy in kilojoules. Convert to kilocalories."""
    if kj is None:
        return None
    return round(kj / 4.184, 1)


def ms_to_min(ms):
    if ms is None:
        return None
    return round(ms / 60000, 1)


def safe_round(x, digits=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(x, digits)


def rolling_mean(values, window):
    """Rolling mean with center alignment. Returns list same length as input."""
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        sub = [v for v in values[start:i+1] if v is not None]
        out.append(sum(sub) / len(sub) if sub else None)
    return out


def rolling_baseline_30d(values):
    """30-day rolling mean, used as baseline for skin temp deviation."""
    return rolling_mean(values, 30)


# =============================================================================
# 12 DERIVED METRICS
# =============================================================================

def compute_rmr(weight_kg, height_m, age=40, sex="male"):
    """
    Mifflin-St Jeor RMR estimate. Single value applied as constant baseline.
    Conservative default age 40, can be parameterized.
    """
    h_cm = height_m * 100
    if sex == "male":
        return 10 * weight_kg + 6.25 * h_cm - 5 * age + 5
    return 10 * weight_kg + 6.25 * h_cm - 5 * age - 161


def compute_neat_kcal_per_day():
    """
    NEAT (non-exercise activity thermogenesis) approximation.
    For a moderately active office worker, ~400-600 kcal/day above RMR.
    Using 500 as a midrange constant.
    """
    return 500


def compute_workout_kcal_per_day(workouts_by_day, all_dates):
    """Metric 4 component: total workout kcal per day."""
    out = []
    for d in all_dates:
        day_total = sum(
            kj_to_kcal(w["kilojoule"]) or 0
            for w in workouts_by_day.get(d, [])
            if w["kilojoule"] is not None
        )
        out.append(round(day_total, 1) if day_total > 0 else 0)
    return out


def compute_skin_temp_deviation(skin_temps):
    """Metric 1: skin temp minus 30-day rolling baseline."""
    baseline = rolling_baseline_30d(skin_temps)
    deviations = []
    for v, b in zip(skin_temps, baseline):
        if v is None or b is None:
            deviations.append(None)
        else:
            deviations.append(round(v - b, 2))
    return deviations


def compute_autonomic_stress_index(rhr_list, hrv_list):
    """Metric 2: RHR divided by HRV. Higher = more autonomic stress."""
    out = []
    for rhr, hrv in zip(rhr_list, hrv_list):
        if rhr is None or hrv is None or hrv == 0:
            out.append(None)
        else:
            out.append(round(rhr / hrv, 2))
    return out


def compute_acwr(strain_list):
    """
    Metric 3: Acute:Chronic Workload Ratio.
    Acute = 7-day rolling mean strain.
    Chronic = 28-day rolling mean strain.
    Sweet spot: 0.8-1.3. Over: >1.5. Under: <0.8.
    """
    acute = rolling_mean(strain_list, 7)
    chronic = rolling_mean(strain_list, 28)
    out = []
    for a, c in zip(acute, chronic):
        if a is None or c is None or c == 0:
            out.append(None)
        else:
            out.append(round(a / c, 2))
    return out


def compute_rmr_neat_baseline():
    """Metric 4 component: RMR + NEAT, daily baseline burn."""
    rmr = compute_rmr(WEIGHT_KG, HEIGHT_M)
    neat = compute_neat_kcal_per_day()
    return round(rmr + neat, 0)


def compute_restorative_sleep_pct(sleep_rows):
    """
    Metric 5: (deep + REM) / total sleep.
    Returns pct (0-100).
    """
    out = []
    for s in sleep_rows:
        deep = s["deep_ms"] or 0
        rem = s["rem_ms"] or 0
        light = s["light_ms"] or 0
        total = deep + rem + light
        if total == 0:
            out.append(None)
        else:
            out.append(round(100 * (deep + rem) / total, 1))
    return out


def compute_sleep_regularity_weekly(sleep_rows, all_dates):
    """
    Metric 6: weekly SD of wake hour (hour of day, fractional).
    Returns one value per ISO week aligned to all_dates, but for
    rendering simplicity we return a per-day list where each day
    carries its week's SD.
    """
    wake_by_date = {}
    for s in sleep_rows:
        if s["end_ts"] is None:
            continue
        dt = s["end_ts"]
        wake_hour = dt.hour + dt.minute / 60 + dt.second / 3600
        wake_by_date[s["d"].isoformat() if hasattr(s["d"], "isoformat") else str(s["d"])] = wake_hour

    week_sds = {}
    week_buckets = {}
    for d_str in all_dates:
        d = date.fromisoformat(d_str)
        iso_year, iso_week, _ = d.isocalendar()
        key = f"{iso_year}-W{iso_week:02d}"
        if d_str in wake_by_date:
            week_buckets.setdefault(key, []).append(wake_by_date[d_str])

    for key, vals in week_buckets.items():
        if len(vals) >= 2:
            week_sds[key] = round(stdev(vals), 2)
        else:
            week_sds[key] = None

    out = []
    for d_str in all_dates:
        d = date.fromisoformat(d_str)
        iso_year, iso_week, _ = d.isocalendar()
        key = f"{iso_year}-W{iso_week:02d}"
        out.append(week_sds.get(key))
    return out


def compute_wake_hour(sleep_rows, all_dates):
    """Metric 11 helper: wake hour per day (for circadian charts)."""
    wake_by_date = {}
    for s in sleep_rows:
        if s["end_ts"] is None:
            continue
        dt = s["end_ts"]
        wake_hour = dt.hour + dt.minute / 60
        d_str = s["d"].isoformat() if hasattr(s["d"], "isoformat") else str(s["d"])
        wake_by_date[d_str] = round(wake_hour, 2)
    return [wake_by_date.get(d_str) for d_str in all_dates]


def compute_workout_zone_distribution(workout):
    """
    Metric 7: identify dominant zone for a workout.
    Returns the zone label (Z1..Z5) where the most time was spent.
    """
    zones = {
        "Z1": workout.get("zone_one_milli") or 0,
        "Z2": workout.get("zone_two_milli") or 0,
        "Z3": workout.get("zone_three_milli") or 0,
        "Z4": workout.get("zone_four_milli") or 0,
        "Z5": workout.get("zone_five_milli") or 0,
    }
    if sum(zones.values()) == 0:
        return None
    return max(zones, key=zones.get)


def compute_workout_duration_min(workout):
    """Metric 12: workout duration in minutes from start/end timestamps."""
    if workout["start_ts"] is None or workout["end_ts"] is None:
        return None
    delta = workout["end_ts"] - workout["start_ts"]
    return round(delta.total_seconds() / 60, 1)


# =============================================================================
# DATA ASSEMBLY
# =============================================================================

def build_daily_series(raw, start_date, end_date):
    """Assemble per-day arrays aligned to a continuous date axis."""
    days = []
    d = start_date
    while d <= end_date:
        days.append(d.isoformat())
        d += timedelta(days=1)

    cycles_by_date = {
        (c["d"].isoformat() if hasattr(c["d"], "isoformat") else str(c["d"])): c
        for c in raw["cycles"]
    }
    recovery_by_date = {
        (r["d"].isoformat() if hasattr(r["d"], "isoformat") else str(r["d"])): r
        for r in raw["recovery"]
    }
    sleep_by_date = {
        (s["d"].isoformat() if hasattr(s["d"], "isoformat") else str(s["d"])): s
        for s in raw["sleep"]
    }
    workouts_by_date = {}
    for w in raw["workouts"]:
        key = w["d"].isoformat() if hasattr(w["d"], "isoformat") else str(w["d"])
        workouts_by_date.setdefault(key, []).append(w)

    strain = [cycles_by_date[d]["strain"] if d in cycles_by_date else None for d in days]
    kcal_total = [
        kj_to_kcal(cycles_by_date[d]["kilojoule"]) if d in cycles_by_date else None
        for d in days
    ]
    workout_kcal = compute_workout_kcal_per_day(workouts_by_date, days)
    rmr_neat = [compute_rmr_neat_baseline()] * len(days)

    rec_score = [recovery_by_date[d]["recovery_score"] if d in recovery_by_date else None for d in days]
    hrv = [recovery_by_date[d]["hrv"] if d in recovery_by_date else None for d in days]
    rhr = [recovery_by_date[d]["rhr"] if d in recovery_by_date else None for d in days]
    spo2 = [recovery_by_date[d]["spo2"] if d in recovery_by_date else None for d in days]
    skin_temp = [recovery_by_date[d]["skin_temp"] if d in recovery_by_date else None for d in days]

    skin_temp_dev = compute_skin_temp_deviation(skin_temp)
    autonomic = compute_autonomic_stress_index(rhr, hrv)
    acwr = compute_acwr(strain)

    sleep_min = []
    sleep_perf = []
    sleep_eff = []
    deep_min = []
    rem_min = []
    sleep_rows_aligned = []
    for d in days:
        s = sleep_by_date.get(d)
        sleep_rows_aligned.append(s)
        if s is None:
            sleep_min.append(None)
            sleep_perf.append(None)
            sleep_eff.append(None)
            deep_min.append(None)
            rem_min.append(None)
        else:
            in_bed = s["in_bed_ms"] or 0
            awake = s["awake_ms"] or 0
            sleep_min.append(ms_to_min(in_bed - awake))
            sleep_perf.append(s["performance"])
            sleep_eff.append(s["efficiency"])
            deep_min.append(ms_to_min(s["deep_ms"]))
            rem_min.append(ms_to_min(s["rem_ms"]))

    restorative_pct = compute_restorative_sleep_pct(
        [s for s in sleep_rows_aligned if s is not None]
    )
    restorative_aligned = []
    rp_idx = 0
    for s in sleep_rows_aligned:
        if s is None:
            restorative_aligned.append(None)
        else:
            restorative_aligned.append(restorative_pct[rp_idx])
            rp_idx += 1

    wake_hour = compute_wake_hour(raw["sleep"], days)

    return {
        "d": days,
        "st": strain,
        "kc": kcal_total,
        "wk": workout_kcal,
        "rm": rmr_neat,
        "ac": acwr,
        "rs": rec_score,
        "hv": hrv,
        "hr": rhr,
        "so": spo2,
        "sk": skin_temp,
        "sd": skin_temp_dev,
        "as": autonomic,
        "sl": sleep_min,
        "sp": sleep_perf,
        "se": sleep_eff,
        "sw": deep_min,
        "sr": rem_min,
        "rp": restorative_aligned,
        "wh": wake_hour,
    }


def build_workout_series(raw):
    """Assemble flat workout arrays for the workouts chart."""
    out = {"d": [], "sp": [], "st": [], "mn": [], "hr": [], "kc": [], "z": []}
    for w in raw["workouts"]:
        d_str = w["d"].isoformat() if hasattr(w["d"], "isoformat") else str(w["d"])
        out["d"].append(d_str)
        out["sp"].append(w.get("sport_name") or "Unknown")
        out["st"].append(safe_round(w.get("strain"), 1))
        out["mn"].append(compute_workout_duration_min(w))
        out["hr"].append(w.get("average_heart_rate"))
        out["kc"].append(kj_to_kcal(w.get("kilojoule")))
        out["z"].append(compute_workout_zone_distribution(w))
    return out


def build_payload(raw, start_date, end_date):
    """Compose final payload matching the dashboard template schema."""
    daily = build_daily_series(raw, start_date, end_date)
    workouts = build_workout_series(raw)
    return {
        "meta": {
            "patient": PATIENT_NAME,
            "gen": date.today().isoformat(),
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "days": len(daily["d"]),
            "h": HEIGHT_M,
            "w": WEIGHT_KG,
            "mhr": MAX_HR,
            "bmi": round(WEIGHT_KG / (HEIGHT_M ** 2), 1),
        },
        "daily": daily,
        "workouts": workouts,
    }


# =============================================================================
# RENDER
# =============================================================================

def render_template(template_path, payload, output_path):
    """Inject JSON payload into template's __DATA_PLACEHOLDER__ token."""
    if not template_path.exists():
        sys.exit(f"ERROR: template not found: {template_path}")
    template = template_path.read_text()
    if "__DATA_PLACEHOLDER__" not in template:
        sys.exit(f"ERROR: __DATA_PLACEHOLDER__ token missing in {template_path.name}")
    payload_json = json.dumps(payload, separators=(",", ":"), default=str)
    rendered = template.replace("__DATA_PLACEHOLDER__", payload_json)
    output_path.write_text(rendered)
    print(f"  wrote {output_path.name} ({len(rendered):,} bytes)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build WHOOP dashboards from Supabase")
    parser.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS,
                        help=f"Lookback window in days (default: {DEFAULT_LOOKBACK_DAYS})")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    end_date = (
        date.fromisoformat(args.end_date) if args.end_date else date.today()
    )
    start_date = end_date - timedelta(days=args.days - 1)

    print(f"Building dashboards for {start_date} to {end_date} ({args.days} days)")

    print("Connecting to Supabase...")
    conn = get_connection()
    print("Fetching WHOOP tables...")
    raw = fetch_data(conn, start_date, end_date)
    conn.close()
    print(f"  cycles: {len(raw['cycles'])}, recovery: {len(raw['recovery'])}, "
          f"sleep: {len(raw['sleep'])}, workouts: {len(raw['workouts'])}")

    print("Computing 12 derived metrics...")
    payload = build_payload(raw, start_date, end_date)

    print("Rendering dashboards...")
    render_template(
        SCRIPT_DIR / "nutritionist_template.html",
        payload,
        SCRIPT_DIR / "martinez_nutritionist_dashboard.html",
    )
    render_template(
        SCRIPT_DIR / "oncologist_template.html",
        payload,
        SCRIPT_DIR / "martinez_oncologist_dashboard.html",
    )

    print("Done.")


if __name__ == "__main__":
    main()
