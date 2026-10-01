"""
build_cfb_benchmarks.py — build cfb_benchmarks.csv, the FBS comparison table
the Team Profiles page uses.

Run it once per season (locally, not on Streamlit Cloud) and commit the CSV:

    python build_cfb_benchmarks.py --season 2022 \
        --pbp https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/data/parquet/pbp_players_pos_2022.parquet

--pbp accepts a local .parquet/.csv/.csv.gz path or a URL. For a newer season,
export play-by-play from your cfbfastR pipeline in R, e.g.:

    pbp <- cfbfastR::load_cfb_pbp(seasons = 2025)
    arrow::write_parquet(pbp, "pbp_2025.parquet")

and point --pbp at that file. Only the columns in profiles.CFBFASTR_COLUMNS
are read, and every metric uses the same definitions as the app.
"""
from __future__ import annotations

import argparse

import pandas as pd

import profiles as pr


def read_pbp(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path, columns=pr.CFBFASTR_COLUMNS)
    return pd.read_csv(path, usecols=pr.CFBFASTR_COLUMNS, low_memory=False)


def build(pbp: pd.DataFrame, season: int) -> pd.DataFrame:
    tables = pr.standardize_cfbfastr(pbp)
    rows = []
    for team in pr.fbs_teams(pbp):
        metrics = pr.team_metrics(tables, team)
        for unit, values in metrics.items():
            for key, (value, n) in values.items():
                rows.append({"season": season, "unit": unit, "metric": key, "team": team, "value": value, "n": n})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pbp", required=True, help="cfbfastR play-by-play file or URL (.parquet or .csv)")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--out", default=pr.BENCHMARK_PATH)
    args = ap.parse_args()
    bench = build(read_pbp(args.pbp), args.season)
    bench.to_csv(args.out, index=False)
    teams = bench["team"].nunique()
    print(f"Wrote {args.out}: {teams} FBS teams, {bench['metric'].nunique()} metrics, season {args.season}")
    summary = bench.groupby(["unit", "metric"])["value"].median().round(3)
    print(summary.to_string())


if __name__ == "__main__":
    main()
