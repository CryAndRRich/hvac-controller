#!/usr/bin/env python3
"""Fetch Climate.OneBuilding TMYx weather for every station and pack it.

The shipped weather CSVs (``smart_buildings/weather/csv``) are the
Climate.OneBuilding ``TMYx.2011-2025`` EPW files of their stations, row i
relabelled as 2023-01-01 00:00 + i hours. ``verify`` rebuilds all six from the
EPW files and they are byte-identical. This script fetches the same product for
every station (14,331 on 2026-10-17), keeps the four columns the shipped CSVs
carry (dry-bulb, dew point, relative humidity, wind speed) and packs them into
one ``weathertmyx.zip`` (the Kaggle dataset ``weathertmyx``) of 1,000-station shards plus a station table.

Work directory layout (``--work-dir``, default ``weather_tmyx/``, git-ignored):

    raw/index/              saved region and country index pages
    stations_2011-2025.tsv  stem -> zip URL
    stations/<stem>.npz     fetch cache, one file per station: x100[8760, 4] + meta;
                            only needed until ``pack`` has run, delete it afterwards
    fetch_log.txt           failures and progress
    packed/weathertmyx.zip  ext_weather_tmyx_NN.npz shards + ext_weather_stations.csv

Values are stored as integers x100: EPW carries at most two decimals, so the
round trip is exact.

Usage (from the repo root):
    python -m hvac_controller.weather index    # crawl index pages
    python -m hvac_controller.weather list     # index pages -> station list
    python -m hvac_controller.weather fetch    # download + convert, resumable
    python -m hvac_controller.weather pack     # cache -> packed/weathertmyx.zip
    python -m hvac_controller.weather verify   # rebuild the six shipped CSVs from the pack
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import io
import json
import re
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np

DEFAULT_WORK_DIR = Path("weather_tmyx")
SHIPPED_CSV_DIR = Path(__file__).resolve().parents[1] / "smart_buildings" / "weather" / "csv"
SITE = "https://climate.onebuilding.org"
REGIONS = (
    "WMO_Region_1_Africa",
    "WMO_Region_2_Asia",
    "WMO_Region_3_South_America",
    "WMO_Region_4_North_and_Central_America",
    "WMO_Region_5_Southwest_Pacific",
    "WMO_Region_6_Europe",
    "WMO_Region_7_Antarctica",
)
VERSION = "TMYx.2011-2025"
HOURS = 8760
COLUMNS = ("temp_c", "dewpoint_c", "rh_pct", "wind_ms")
# EPW missing-value sentinels for the four kept columns.
MISSING = (99.9, 99.9, 999.0, 999.0)
SHIPPED_STEMS = {
    "atlanta": "USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190",
    "denver": "USA_CO_Denver.Intl.AP.725650",
    "fairbanks": "USA_AK_Fairbanks.Intl.AP.702610",
    "minneapolis": "USA_MN_Minneapolis-St.Paul.Intl.AP.726580",
    "phoenix": "USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780",
    "singapore": "SGP_SG_Singapore-Changi.Intl.AP.486980",
}


class Paths:
    def __init__(self, work_dir: Path):
        self.index = work_dir / "raw" / "index"
        self.list = work_dir / f"stations_{VERSION.split('.', 1)[1]}.tsv"
        self.cache = work_dir / "stations"
        self.log = work_dir / "fetch_log.txt"
        self.packed = work_dir / "packed"
        self.zip = self.packed / "weathertmyx.zip"


def curl(url: str) -> bytes:
    # curl rather than urllib: python.org builds on macOS ship without a CA bundle.
    return subprocess.run(
        ["curl", "-sSf", "--max-time", "120", url], capture_output=True, check=True
    ).stdout


def crawl_index(paths: Paths) -> None:
    """Save every region page and every country page under it."""
    paths.index.mkdir(parents=True, exist_ok=True)
    for region in REGIONS:
        page = paths.index / f"{region}.html"
        if not page.exists():
            page.write_bytes(curl(f"{SITE}/{region}/default.html"))
        countries = sorted(
            set(re.findall(r'href="([A-Z][A-Za-z0-9_.-]*)/index\.html"', page.read_text(errors="replace")))
        )
        for country in countries:
            out = paths.index / f"{region}__{country}.html"
            if not out.exists():
                out.write_bytes(curl(f"{SITE}/{region}/{country}/index.html"))
    print(f"{len(list(paths.index.glob('*__*.html')))} country pages in {paths.index}")


def build_list(paths: Paths) -> None:
    rows: dict[str, str] = {}
    pattern = r'href="([^"]*_' + re.escape(VERSION) + r'\.zip)"'
    for page in sorted(paths.index.glob("*__*.html")):
        region, country = page.stem.split("__", 1)
        for href in re.findall(pattern, page.read_text(errors="replace")):
            stem = href.rsplit("/", 1)[-1][: -len(f"_{VERSION}.zip")]
            rows.setdefault(stem, f"{SITE}/{region}/{country}/{href}")
    paths.list.write_text("".join(f"{s}\t{u}\n" for s, u in sorted(rows.items())))
    print(f"{len(rows)} stations -> {paths.list}")


def parse_epw(text: str) -> tuple[np.ndarray, dict]:
    lines = text.splitlines()
    loc = next(l for l in lines if l.startswith("LOCATION,")).split(",")
    comment = next((l for l in lines if l.startswith("COMMENTS 1,")), "")
    data = [r.split(",") for r in lines if r[:1].isdigit()]
    if len(data) != HOURS:
        raise ValueError(f"{len(data)} data rows")
    # EPW fields 6, 7, 8, 21: dry-bulb, dew point, relative humidity, wind speed.
    arr = np.array([[float(r[6]), float(r[7]), float(r[8]), float(r[21])] for r in data])
    months = dict(re.findall(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)=(\d{4})", comment))
    meta = {
        "name": loc[1], "state": loc[2], "country": loc[3], "source": loc[4], "wmo": loc[5],
        "lat": float(loc[6]), "lon": float(loc[7]), "tz": float(loc[8]), "elev": float(loc[9]),
        "month_years": months,
    }
    return arr, meta


def to_x100(arr: np.ndarray) -> np.ndarray:
    x = np.round(arr * 100).astype(np.int32)
    if not np.array_equal(x / 100, arr):
        raise ValueError("value with more than two decimals")
    return x


def fetch_one(paths: Paths, stem: str, url: str) -> tuple[str, str]:
    out = paths.cache / f"{stem}.npz"
    if out.exists():
        return stem, "cached"
    err = ""
    for attempt in range(3):
        try:
            zf = zipfile.ZipFile(io.BytesIO(curl(url)))
            member = next(n for n in zf.namelist() if n.lower().endswith(".epw"))
            arr, meta = parse_epw(zf.read(member).decode("latin-1"))
            meta.update(stem=stem, url=url, version=VERSION)
            tmp = out.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, x100=to_x100(arr), meta=np.array(json.dumps(meta)))
            tmp.rename(out)
            return stem, "ok"
        except Exception as e:  # noqa: BLE001 - logged and retried
            err = f"{type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))
    return stem, f"FAIL {err}"


def fetch(paths: Paths, workers: int) -> None:
    paths.cache.mkdir(parents=True, exist_ok=True)
    todo = [l.split("\t") for l in paths.list.read_text().splitlines()]
    t0, done, failed = time.time(), 0, 0
    with cf.ThreadPoolExecutor(workers) as ex, paths.log.open("a") as log:
        for stem, status in ex.map(lambda su: fetch_one(paths, *su), todo):
            done += 1
            if status.startswith("FAIL"):
                failed += 1
                log.write(f"{stem}\t{status}\n")
            if done % 200 == 0 or done == len(todo):
                log.write(f"# {done}/{len(todo)} failed={failed} {time.time() - t0:.0f}s\n")
                log.flush()
    print(f"{done} processed, {failed} failed, {time.time() - t0:.0f}s")


def load_station(cache: Path, stem: str) -> tuple[np.ndarray, dict]:
    z = np.load(cache / f"{stem}.npz")
    return z["x100"], json.loads(str(z["meta"]))


class Packed:
    """Read stations from the packed data: the zip, or the folder Kaggle unpacks it into."""

    def __init__(self, path: Path):
        path = Path(path)
        if path.is_dir():
            self._open = lambda name: (path / name).open("rb")
        else:
            zf = zipfile.ZipFile(path)
            self._open = zf.open
        with self._open("ext_weather_stations.csv") as f:
            self.table = {r["stem"]: r for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))}
        self._shard: tuple[int, np.ndarray] | None = None

    def x100(self, stem: str) -> np.ndarray:
        shard, idx = int(self.table[stem]["shard"]), int(self.table[stem]["idx"])
        if self._shard is None or self._shard[0] != shard:
            with self._open(f"ext_weather_tmyx_{shard:02d}.npz") as f:
                self._shard = (shard, np.load(io.BytesIO(f.read()))["x100"])
        return self._shard[1][idx]


def find_packed(root: Path) -> Path:
    """The packed weather under ``root``: the folder Kaggle unpacks the dataset into."""
    for table in sorted(Path(root).rglob("ext_weather_stations.csv")):
        return table.parent
    raise FileNotFoundError(f"no packed TMYx weather under {root}")


# Shipped convection coefficients and the closest wind fit (EXTERNAL_DATA.md §1): no simple
# statistic reproduces them exactly, so this is an estimate for stations outside the six.
H_RANGE = (8.8, 26.7)


def estimate_h(wind_mean_ms: float) -> float:
    return float(np.clip(1.7 + 5.3 * wind_mean_ms, *H_RANGE))


def write_preset(packed: Packed, stem: str, weather_dir: Path, start_timestamp: str, h: float, name: str) -> Path:
    """Write one station as a weather preset (shipped CSV layout + yaml) the scenario generator loads."""
    row = packed.table[stem]
    csv_dir = Path(weather_dir) / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / f"{name}.csv").write_text(to_weather_csv(packed.x100(stem), row["name"], row["country"]))
    yaml_path = Path(weather_dir) / f"{name}.yaml"
    yaml_path.write_text(
        f"name: {name}\nweather_csv: csv/{name}.csv\nconvection_coefficient: {h:.1f}\n"
        f"start_timestamp: {start_timestamp}\n"
    )
    return yaml_path


def to_weather_csv(x100: np.ndarray, name: str, country: str) -> str:
    """One station-year in the shipped CSV layout, labelled 2023 like the shipped files."""
    out = [",Time,StationName,Location,TempC,DewPointC,WindspeedKmph,Humidity,TempF"]
    loc = f'"{name}, {country}"'
    t0 = np.datetime64("2023-01-01T00:00")
    for i, (t, d, h, w) in enumerate(x100.tolist()):
        ts = str(t0 + np.timedelta64(i, "h"))
        temp = t / 100
        out.append(
            f"{i},{ts[0:4]}{ts[5:7]}{ts[8:10]}-{ts[11:13]}{ts[14:16]},{name},{loc},"
            f"{temp},{d / 100},{w / 100 * 3.6},{h / 100},{temp * 9 / 5 + 32}"
        )
    return "\n".join(out) + "\n"


def verify(paths: Paths) -> None:
    packed = Packed(paths.zip)
    for city, stem in SHIPPED_STEMS.items():
        row = packed.table[stem]
        ours = to_weather_csv(packed.x100(stem), row["name"], row["country"])
        theirs = (SHIPPED_CSV_DIR / f"{city}.csv").read_text()
        if ours == theirs:
            print(f"{city:12s} identical")
            continue
        a, b = ours.splitlines(), theirs.splitlines()
        bad = [i for i in range(min(len(a), len(b))) if a[i] != b[i]]
        first = f"ours={a[bad[0]]!r} theirs={b[bad[0]]!r}" if bad else f"rows {len(a)} vs {len(b)}"
        print(f"{city:12s} DIFF rows={len(bad)} {first}")


def pack(paths: Paths, shard_size: int) -> None:
    # Shards go straight into the zip (stored, the npz are already compressed); peak memory
    # stays near 150 MB per 1,000 stations.
    stems = sorted(p.name[: -len(".npz")] for p in paths.cache.glob("*.npz") if not p.name.endswith(".tmp.npz"))
    paths.packed.mkdir(parents=True, exist_ok=True)
    tmp_zip = paths.zip.with_suffix(".tmp.zip")
    rows = []
    with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_STORED) as zf:
        for k in range(0, len(stems), shard_size):
            chunk = stems[k:k + shard_size]
            x = np.empty((len(chunk), HOURS, len(COLUMNS)), np.int32)
            for j, stem in enumerate(chunk):
                x100, meta = load_station(paths.cache, stem)
                x[j] = x100
                arr = x100 / 100.0
                n_missing = int(sum((arr[:, c] == MISSING[c]).sum() for c in range(len(COLUMNS))))
                temp, wind = arr[:, 0], arr[:, 3]
                rows.append([
                    k + j, k // shard_size, j, stem, meta["name"], meta["state"], meta["country"], meta["wmo"],
                    meta["lat"], meta["lon"], meta["tz"], meta["elev"], f"{temp.mean():.2f}", f"{temp.min():.1f}",
                    f"{temp.max():.1f}", f"{wind[wind < 900].mean():.3f}", n_missing,
                    ";".join(f"{m}={y}" for m, y in meta["month_years"].items()),
                ])
            buf = io.BytesIO()
            np.savez_compressed(buf, x100=x, columns=np.array(COLUMNS))
            zf.writestr(f"ext_weather_tmyx_{k // shard_size:02d}.npz", buf.getvalue())
        head = ("row,shard,idx,stem,name,state,country,wmo,lat,lon,tz,elev,temp_mean,temp_min,temp_max,"
                "wind_mean_ms,n_missing,month_years")
        lines = [head] + [
            ",".join(f'"{v}"' if isinstance(v, str) and "," in v else str(v) for v in r) for r in rows
        ]
        zf.writestr("ext_weather_stations.csv", "\n".join(lines) + "\n")
    tmp_zip.rename(paths.zip)
    print(f"packed {len(stems)} stations into {(len(stems) + shard_size - 1) // shard_size} shards in {paths.zip}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("index", "list", "fetch", "verify", "pack"))
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR,
                        help=f"Cache and output directory (default: {DEFAULT_WORK_DIR}).")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel downloads for fetch (default: 4; the site is a free service, keep it low).")
    parser.add_argument("--shard-size", type=int, default=1000, help="Stations per packed shard (default: 1000).")
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    paths = Paths(args.work_dir)
    if args.command == "index":
        crawl_index(paths)
    elif args.command == "list":
        build_list(paths)
    elif args.command == "fetch":
        fetch(paths, args.workers)
    elif args.command == "verify":
        verify(paths)
    else:
        pack(paths, args.shard_size)


if __name__ == "__main__":
    main()
