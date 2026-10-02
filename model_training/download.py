"""Download cfbfastR play-by-play (2014-2025) from the sportsdataverse release into DATA_DIR."""
import os, urllib.request
from prep import DATA_DIR

URL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/cfbfastR_cfb_pbp/play_by_play_{}.parquet"
os.makedirs(DATA_DIR, exist_ok=True)
for yr in range(2014, 2026):
    path = os.path.join(DATA_DIR, f"pbp_{yr}.parquet")
    if not os.path.exists(path):
        print("downloading", yr, flush=True)
        urllib.request.urlretrieve(URL.format(yr), path)
