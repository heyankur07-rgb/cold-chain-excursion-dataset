# SPDX-License-Identifier: CC0-1.0
# License: CC0 1.0 Universal (Public Domain Dedication) - https://creativecommons.org/publicdomain/zero/1.0/
# Applies to this script and to the dataset it generates (data.csv). Original synthetic data, created by the submitter.
"""
Synthetic generator for the "Cold-Chain Shipment Temperature Excursion" dataset.

Reproducible: run `python generate_data.py --seed <SEED> --out data.csv` and you get byte-identical
output (no wall-clock or environment dependence). The seed used for the published dataset
is withheld from solvers so the test labels cannot be regenerated; reviewers can request it.

Generative story (simplified pharma cold-chain physics)
-------------------------------------------------------
A consignment (1-6 shippers packed together, same product, packaging, route and carrier)
leaves an origin site and travels through 0-3 air hubs to a destination country.
Each shipper is a passive or active insulated box with a nominal "holdover" duration:
how long it can keep its payload in range at a 30 C ambient profile.

Whether a shipper has a temperature EXCURSION depends on:
  * actual transit time relative to holdover (actual time is NOT known at dispatch;
    only the planned time and expected customs dwell are),
  * heat exposure on route (season, hemisphere, hot transit hubs),
  * cold exposure for 2-8 C goods in passive boxes during winter (freezing is also an excursion),
  * how well the box was pre-conditioned (precool reading at pack-out),
  * packing events recorded in free-text handler notes,
  * carrier quality, active-container power faults,
  * a shared consignment-level shock (same truck / same tarmac wait) -> correlated rows.

Deliberate pitfalls planted for the benchmark (documented in DATASET.md):
  * post-arrival LEAKAGE columns (actual_transit_h, logger_peak_status, logger_alarm_level, ...)
  * mixed timestamp formats depending on origin site
  * precool reading in Celsius OR Fahrenheit depending on logger model
  * pipe-delimited multi-value route column
  * holdover hours embedded inside the packaging code string
  * missing-not-at-random expected_customs_h (blank = domestic, not unknown)
  * pure-noise / identifier columns
  * grouped/correlated rows (consignment_id) and time drift
"""
import argparse
import hashlib

import numpy as np
import pandas as pd

SEED = None  # set from --seed at runtime (withheld for the published dataset)

# ---------------------------------------------------------------- reference tables
ORIGINS = {
    # site: (country, climate_base_c, seasonal_amp, southern_hemisphere, ts_format, weight)
    "BSL": ("CH", 11.0, 9.0, False, "dmy", 0.12),
    "GNT": ("BE", 11.0, 7.5, False, "dmy", 0.10),
    "ORK": ("IE", 10.5, 5.0, False, "dmy", 0.08),
    "LYS": ("FR", 13.0, 9.5, False, "dmy", 0.08),
    "PNQ": ("IN", 26.0, 4.5, False, "dmy", 0.14),
    "HYD": ("IN", 27.5, 5.0, False, "dmy", 0.10),
    "RDU": ("US", 16.0, 10.0, False, "iso", 0.12),
    "IND": ("US", 12.5, 12.5, False, "iso", 0.10),
    "SIN": ("SG", 28.0, 1.0, False, "iso", 0.08),
    "GRU": ("BR", 20.5, 3.5, True, "dmy", 0.08),
}

DESTS = {
    # country: (climate_base_c, seasonal_amp, southern, international_hub_bias, weight)
    "DE": (10.5, 9.0, False, "EU", 0.08), "FR": (12.5, 8.5, False, "EU", 0.06),
    "GB": (11.0, 6.5, False, "EU", 0.07), "ES": (16.5, 8.0, False, "EU", 0.05),
    "PL": (9.0, 10.5, False, "EU", 0.04), "US": (14.5, 10.5, False, "AM", 0.10),
    "CA": (7.0, 13.0, False, "AM", 0.04), "MX": (21.0, 4.0, False, "AM", 0.04),
    "BR": (22.0, 3.5, True, "AM", 0.04), "IN": (26.5, 5.0, False, "ME", 0.08),
    "AE": (28.5, 8.0, False, "ME", 0.05), "SA": (27.5, 8.5, False, "ME", 0.04),
    "EG": (23.0, 7.0, False, "ME", 0.03), "KE": (19.5, 2.0, True, "AF", 0.03),
    "NG": (27.0, 2.0, False, "AF", 0.03), "ZA": (17.5, 5.0, True, "AF", 0.03),
    "SG": (28.0, 1.0, False, "AS", 0.03), "JP": (16.0, 10.0, False, "AS", 0.04),
    "AU": (18.0, 6.0, True, "AS", 0.04), "VN": (27.0, 3.5, False, "AS", 0.03),
    "TH": (28.5, 2.5, False, "AS", 0.03),
}

HUBS = {
    # hub: (climate_base_c, seasonal_amp, southern, region)
    "FRA": (10.5, 9.0, False, "EU"), "AMS": (10.5, 7.0, False, "EU"),
    "CDG": (12.0, 8.0, False, "EU"), "IST": (15.0, 9.0, False, "EU"),
    "DXB": (29.0, 8.0, False, "ME"), "DOH": (28.5, 8.5, False, "ME"),
    "BOM": (27.5, 3.0, False, "ME"), "DEL": (25.5, 8.5, False, "ME"),
    "SIN": (28.0, 1.0, False, "AS"), "HKG": (23.5, 6.0, False, "AS"),
    "JFK": (13.0, 10.5, False, "AM"), "ORD": (10.5, 13.5, False, "AM"),
    "MIA": (25.5, 3.5, False, "AM"), "NBO": (19.5, 2.0, True, "AF"),
    "JNB": (16.5, 5.0, True, "AF"),
}
HUBS_BY_REGION = {}
for h, v in HUBS.items():
    HUBS_BY_REGION.setdefault(v[3], []).append(h)

PRODUCTS = {
    # product_class: (temp_range, target_mid_c, weight)
    "VACCINE": ("2-8C", 5.0, 0.34),
    "INSULIN": ("2-8C", 5.0, 0.18),
    "MAB_BIOLOGIC": ("2-8C", 5.0, 0.20),
    "CELL_THERAPY": ("-20C", -20.0, 0.06),
    "MRNA_BULK": ("-20C", -20.0, 0.06),
    "DIAGNOSTIC_KIT": ("15-25C", 20.0, 0.16),
}

PACKAGING = {
    # code: (holdover_h, is_active, insulation_quality 0-1, allowed ranges)
    "EPS-GEL-24H": (24, False, 0.35, ("2-8C", "15-25C")),
    "PCM28-48H": (48, False, 0.60, ("2-8C",)),
    "PCM28-72H": (72, False, 0.70, ("2-8C",)),      # introduced mid-2025 (drift)
    "PCM28-96H": (96, False, 0.80, ("2-8C",)),
    "CRT-PCM-72H": (72, False, 0.65, ("15-25C",)),
    "DRYICE-120H": (120, False, 0.75, ("-20C",)),
    "ACTIVE-RKN": (240, True, 0.95, ("2-8C", "-20C", "15-25C")),
}

CARRIERS = {
    # carrier: (risk_offset, delay_sigma, weight)
    "AEROLINK": (0.00, 0.12, 0.26),
    "BLUESKY_CARGO": (0.10, 0.16, 0.22),
    "TRANSCOLD": (-0.15, 0.10, 0.18),
    "NORTHWIND_EXP": (0.05, 0.14, 0.14),
    "QUICKFREIGHT": (0.55, 0.24, 0.12),             # the "cheap" carrier
    "MERIDIAN_LOG": (-0.05, 0.13, 0.08),
}

# empty shipper (tare) weight in kg by packaging type
PACK_TARE_KG = {"EPS-GEL-24H": 3.0, "PCM28-48H": 5.5, "PCM28-72H": 7.5, "PCM28-96H": 9.5,
                "CRT-PCM-72H": 7.0, "DRYICE-120H": 11.0, "ACTIVE-RKN": 14.0}

SERVICE = {"PRIORITY": 0.75, "STANDARD": 1.0, "ECONOMY": 1.35}

LOGGERS = {"TL-20": "C", "TempTrak-US": "F", "SenseQ-4": "C"}

NOTE_EVENTS = {
    # event: (logit effect, phrasings)
    "routine": (0.0, ["packed per SOP", "std pack-out", "ok", "routine pack out, no issues",
                      "packed and sealed", "pack-out complete"]),
    "short_condition": (0.95, ["gel packs conditioned short ({h}h)", "PCM bricks only {h}h in chamber",
                               "short conditioning {h}h - supervisor approved",
                               "bricks conditioned {h}h (below spec)"]),
    "full_condition": (-0.35, ["gel packs conditioned full {h}h", "PCM conditioned {h}h per spec",
                               "bricks conditioned {h}h ok"]),
    "lid_damage": (1.20, ["lid tape damaged - retaped", "box lid corner crushed, retaped",
                          "lid seal torn, re-taped at dock", "minor lid damage, taped"]),
    "dock_wait": (0.70, ["staged on dock {m} min before pickup", "pallet waited on dock {m}min",
                         "dock dwell {m} min (truck late)"]),
    "relabel": (0.0, ["relabelled - wrong address label", "label reprinted", "AWB label replaced"]),
    "extra_coolant": (-0.55, ["extra coolant added", "added 2 extra gel packs", "extra PCM brick added"]),
    "customs_docs": (0.45, ["permit docs pending at dispatch", "import licence copy missing - sent by email",
                            "customs docs incomplete at pickup"]),
}


def ambient_c(base, amp, southern, month, rng_noise):
    """Monthly mean daily max temperature; month 1..12. Peak in July (north)."""
    phase = (month - 7) / 12.0 * 2 * np.pi
    seasonal = amp * np.cos(phase)
    if southern:
        seasonal = -seasonal
    return base + seasonal + rng_noise


def fmt_ts(ts: pd.Timestamp, style: str) -> str:
    if style == "dmy":
        return ts.strftime("%d/%m/%Y %H:%M")
    return ts.strftime("%Y-%m-%d %H:%M")


def pick(rng, d, idx=-1):
    keys = list(d.keys())
    w = np.array([d[k][idx] if isinstance(d[k], tuple) else d[k] for k in keys], dtype=float)
    return keys[rng.choice(len(keys), p=w / w.sum())]


def generate(n_consignments: int = 26000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2026-06-30")
    total_days = (end - start).days

    # business growth: more consignments later -> sample day with increasing density
    u = rng.random(n_consignments)
    day_offsets = np.floor(total_days * u ** 0.85).astype(int)
    day_offsets.sort()

    rows = []
    ship_counter = 0
    for ci in range(n_consignments):
        dispatch = start + pd.Timedelta(days=int(day_offsets[ci])) + pd.Timedelta(
            minutes=int(rng.integers(5 * 60, 22 * 60)))
        month = dispatch.month
        origin = pick(rng, ORIGINS)
        o_country, o_base, o_amp, o_south, ts_style, _ = ORIGINS[origin]
        dest = pick(rng, DESTS)
        while dest == o_country and rng.random() < 0.6:   # keep domestic share modest
            dest = pick(rng, DESTS)
        d_base, d_amp, d_south, d_region, _ = DESTS[dest]
        domestic = dest == o_country

        product = pick(rng, PRODUCTS)
        trange, target_mid, _ = PRODUCTS[product]

        # packaging choice depends on range and on date (PCM28-72H only after 2025-06)
        options = [p for p, v in PACKAGING.items() if trange in v[3]]
        if dispatch < pd.Timestamp("2025-06-01"):
            options = [p for p in options if p != "PCM28-72H"]
        pw = np.array([0.6 if PACKAGING[p][1] else 1.0 for p in options])
        if trange == "2-8C" and not domestic:
            pw = pw * np.array([0.25 if PACKAGING[p][0] <= 24 else 1.0 for p in options])
        packaging = options[rng.choice(len(options), p=pw / pw.sum())]
        holdover, is_active, insul, _ = PACKAGING[packaging]

        carrier = pick(rng, CARRIERS)
        c_risk, c_sigma, _ = CARRIERS[carrier]
        service = rng.choice(list(SERVICE.keys()), p=[0.3, 0.5, 0.2])

        # route
        if domestic:
            n_hubs = int(rng.choice([0, 1], p=[0.8, 0.2]))
        else:
            n_hubs = int(rng.choice([0, 1, 2, 3], p=[0.12, 0.45, 0.33, 0.10]))
        hub_pool = list(HUBS.keys())
        pref = HUBS_BY_REGION.get(d_region, [])
        hubs = []
        for _ in range(n_hubs):
            pool = pref if (pref and rng.random() < 0.5) else hub_pool
            h = pool[rng.integers(len(pool))]
            if h not in hubs:
                hubs.append(h)
        n_hubs = len(hubs)

        base_h = (6 if domestic else 14) + 7.5 * n_hubs + rng.gamma(2.0, 4.0)
        planned_h = round(base_h * SERVICE[service], 1)
        exp_customs = np.nan if domestic else round(float(rng.gamma(2.0, 3.0 + (2.5 if d_region in ("AF", "ME") else 0))), 1)

        # ambient exposure along legs (latent)
        temps = [ambient_c(o_base, o_amp, o_south, month, rng.normal(0, 1.5)),
                 ambient_c(d_base, d_amp, d_south, month, rng.normal(0, 1.5))]
        for h in hubs:
            hb, ha, hs, _ = HUBS[h]
            temps.append(ambient_c(hb, ha, hs, month, rng.normal(0, 1.5)) + 2.0)  # tarmac penalty
        temps = np.array(temps)
        fc_origin = round(float(temps[0] + rng.normal(0, 1.2)), 1)
        fc_dest = round(float(temps[1] + rng.normal(0, 1.2)), 1)

        # shared consignment shock (truck breakdown, tarmac hold...)
        cons_shock = rng.normal(0, 0.55)
        cons_delay_mult = np.exp(rng.normal(0, c_sigma + 0.05 * n_hubs))

        n_ship = int(rng.choice([1, 2, 3, 4, 5, 6], p=[0.38, 0.22, 0.15, 0.11, 0.08, 0.06]))
        cons_id = f"CN{ci + 100000:06d}"
        note_event_cons = None
        if rng.random() < 0.12:
            note_event_cons = "customs_docs" if not domestic else "dock_wait"

        for _ in range(n_ship):
            ship_counter += 1
            sid = f"SHP{hashlib.md5(f'{SEED}-{ship_counter}'.encode()).hexdigest()[:8].upper()}"
            payload = int(rng.integers(20, 400))
            weight = round(4.0 + 0.012 * payload + PACK_TARE_KG[packaging] + rng.normal(0, 0.8), 2)

            logger = rng.choice(list(LOGGERS.keys()), p=[0.45, 0.30, 0.25])
            # pre-conditioning quality
            if trange == "-20C":
                precool_c = target_mid + rng.normal(0, 2.0)
            else:
                precool_c = target_mid + rng.normal(0, 1.1)
            bad_precool = rng.random() < 0.07
            if bad_precool:
                precool_c += rng.choice([-1, 1]) * rng.uniform(2.5, 6.0)
            precool_reading = precool_c * 9 / 5 + 32 if LOGGERS[logger] == "F" else precool_c

            # handler notes (dispatch-time)
            events = []
            if rng.random() < 0.55:
                events.append("routine")
            else:
                pool = ["short_condition", "full_condition", "lid_damage", "dock_wait",
                        "relabel", "extra_coolant"]
                pw2 = np.array([0.20, 0.30, 0.10, 0.15, 0.15, 0.10])
                k = 1 if rng.random() < 0.8 else 2
                events += list(rng.choice(pool, size=k, replace=False, p=pw2))
            if note_event_cons and rng.random() < 0.85:
                events.append(note_event_cons)
            note_parts, note_effect = [], 0.0
            for ev in events:
                eff, phr = NOTE_EVENTS[ev]
                note_effect += eff
                t = phr[rng.integers(len(phr))]
                t = t.format(h=int(rng.integers(3, 7)) if ev == "short_condition" else int(rng.integers(12, 25)),
                             m=int(rng.integers(30, 120)))
                if rng.random() < 0.25:
                    t = t.upper() if rng.random() < 0.3 else t.lower()
                note_parts.append(t)
            notes = "; ".join(note_parts) if rng.random() > 0.04 else ""

            # latent actual transit
            delay_mult = cons_delay_mult * np.exp(rng.normal(0, 0.06))
            customs_actual = 0.0 if domestic else exp_customs * np.exp(rng.normal(0, 0.45))
            if "customs_docs" in events:
                customs_actual += rng.gamma(2.0, 6.0)
            actual_h = planned_h * delay_mult + customs_actual
            if "dock_wait" in events:
                actual_h += rng.uniform(0.5, 2.0)

            # thermal stress
            heat = np.clip(temps - 22.0, 0, None)
            cold = np.clip(4.0 - temps, 0, None)
            heat_idx = (heat.mean() + 0.5 * heat.max()) / 8.0 * (1.0 - insul)
            cold_idx = (cold.mean() + 0.5 * cold.max()) / 6.0 * (1.0 - insul) if trange == "2-8C" else 0.0
            ratio = actual_h / holdover
            precool_dev = abs(precool_c - target_mid)

            logit = (-4.75
                     + 3.0 * max(ratio - 0.85, 0) + 0.6 * ratio
                     + 2.6 * heat_idx * (1.0 + 0.6 * ratio)
                     + 1.8 * cold_idx
                     + 0.38 * max(precool_dev - 1.0, 0)
                     + c_risk
                     + note_effect
                     + cons_shock
                     + rng.normal(0, 0.35))
            if trange == "15-25C":
                logit -= 0.9
            if is_active:
                logit = -4.8 + c_risk + 0.5 * note_effect + 0.5 * cons_shock + rng.normal(0, 0.3)
                if rng.random() < 0.025 * (1 + ratio / 3):   # power fault
                    logit += 4.5
            p = 1.0 / (1.0 + np.exp(-logit))
            excursion = int(rng.random() < p)

            # ---------------- post-arrival (LEAKAGE) columns
            if trange == "-20C":
                base_max = -20 + rng.normal(0, 1.5)
                max_t = base_max + (rng.uniform(6, 18) if excursion else rng.uniform(0, 4.5))
                min_t = -20 - abs(rng.normal(0, 3))
            elif trange == "15-25C":
                max_t = 20 + (rng.uniform(5.5, 14) if excursion else rng.uniform(0, 4.6))
                min_t = 20 - rng.uniform(0, 4.5)
            else:
                if excursion and cold_idx > heat_idx and rng.random() < 0.7:
                    max_t = 5 + rng.uniform(0, 2.8)
                    min_t = 2 - rng.uniform(0.5, 4.0)
                elif excursion:
                    max_t = 8 + rng.uniform(0.4, 9)
                    min_t = 2 + rng.uniform(0, 2.5)
                else:
                    max_t = 5 + rng.uniform(0, 2.9)
                    min_t = 2 + rng.uniform(0, 2.5)
            alarms = int(rng.poisson(3.0) + 1) if excursion else int(rng.random() < 0.04)
            qa = ("QUARANTINED" if excursion and rng.random() < 0.9 else
                  ("QUARANTINED" if rng.random() < 0.02 else "RELEASED"))
            claim = int(excursion and rng.random() < 0.55)
            # summarise the logger trace as categories (raw temps are multi-modal by range)
            lo, hi = {"2-8C": (2.0, 8.0), "-20C": (-60.0, -15.0), "15-25C": (15.0, 25.0)}[trange]
            peak_status = "ABOVE_RANGE" if max_t > hi else ("BELOW_RANGE" if min_t < lo else "IN_RANGE")
            alarm_level = "NONE" if alarms == 0 else ("SINGLE" if alarms == 1 else "MULTIPLE")

            rows.append({
                "shipment_id": sid,
                "consignment_id": cons_id,
                "dispatch_ts": fmt_ts(dispatch, ts_style),
                "origin_site": origin,
                "destination_country": dest,
                "route_hubs": "|".join(hubs),
                "carrier": carrier,
                "service_level": service,
                "product_class": product,
                "temp_range": trange,
                "packaging_code": packaging,
                "payload_units": payload,
                "gross_weight_kg": weight,
                "logger_model": logger,
                "precool_reading": round(float(precool_reading), 1),
                "planned_transit_h": planned_h,
                "expected_customs_h": exp_customs,
                "forecast_max_temp_origin_c": fc_origin,
                "forecast_max_temp_dest_c": fc_dest,
                "handler_notes": notes,
                # noise columns
                "label_printer_id": f"LP-{int(rng.integers(1, 40)):02d}",
                "record_checksum": hashlib.sha1(f"{sid}-{SEED}".encode()).hexdigest()[:16],
                # leakage (recorded after arrival)
                "actual_transit_h": round(float(actual_h), 1),
                "logger_peak_status": peak_status,
                "logger_alarm_level": alarm_level,
                "qa_disposition": qa,
                "insurance_claim_filed": "YES" if claim else "NO",
                # target
                "excursion": excursion,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="random seed (published dataset's seed is withheld)")
    ap.add_argument("--out", default="data.csv")
    args = ap.parse_args()
    SEED = args.seed
    df = generate()
    df.to_csv(args.out, index=False)
    print(df.shape, "excursion rate:", round(df["excursion"].mean(), 4))
