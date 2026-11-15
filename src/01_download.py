"""Module 1: download and cache price data.

    python src/01_download.py            # download anything not cached yet
    python src/01_download.py --force    # re-download everything
    python src/01_download.py --check    # offline check only (no internet)

Saves one CSV per symbol in data/raw/ (date, open, high, low, close, volume),
plus data/raw/_manifest.csv. Safe to re-run: cached symbols are skipped, so
if Yahoo rate-limits you, wait a few minutes and run it again.

Tip: one person runs this and shares the data/raw folder (zip it) so the
whole team works from identical prices.
"""
import argparse
import sys
import time

import pandas as pd

from common import (ALL_SYMBOLS, HOLDOUT_TICKERS, END, MANIFEST_PATH, MARKET_TICKERS, PRICE_COLS,
                    RAW_DIR, START, TICKER_SECTOR, csv_path, load_raw)


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance frame for one symbol -> date + lowercase OHLCV, no empty rows."""
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    df = df.dropna(subset=["close"])
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df[["date"] + PRICE_COLS]


def split_batch(raw: pd.DataFrame, symbols: list) -> dict:
    """Split a multi-symbol yf.download result into {symbol: tidy frame}."""
    out = {}
    for s in symbols:
        try:
            part = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        part = tidy(part)
        if len(part):
            out[s] = part
    return out


def download(symbols, start, end, batch_size, pause, retries):
    import yfinance as yf  # imported here so --check works without it

    todo = list(symbols)
    for attempt in range(1, retries + 1):
        if not todo:
            break
        print(f"\nAttempt {attempt}: {len(todo)} symbols to fetch")
        failed = []
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            try:
                raw = yf.download(batch, start=start, end=end, auto_adjust=True,
                                  group_by="ticker", threads=True, progress=False)
                got = split_batch(raw, batch)
            except Exception as e:  # network errors, rate limits
                print(f"  batch {batch[0]}..{batch[-1]} failed: {e}")
                got = {}
            for s in batch:
                if s in got:
                    got[s].to_csv(csv_path(s), index=False)
                    print(f"  ok   {s:6s} {len(got[s]):5d} rows")
                else:
                    failed.append(s)
                    print(f"  FAIL {s}")
            time.sleep(pause)
        todo = failed
        if todo and attempt < retries:
            wait = 30 * attempt
            print(f"{len(todo)} failed; waiting {wait}s before retrying (rate limit?)")
            time.sleep(wait)
    return todo


def check(symbols, start, manifest_path=MANIFEST_PATH) -> bool:
    """Load every symbol from disk only and write the manifest."""
    rows, problems = [], []
    first_ok = pd.Timestamp(start) + pd.Timedelta(days=10)
    for s in symbols:
        try:
            df = load_raw(s)
        except FileNotFoundError:
            problems.append(f"{s}: missing")
            continue
        if df.empty:
            problems.append(f"{s}: empty file")
            continue
        dups = int(df["date"].duplicated().sum())
        bad = int((df["close"] <= 0).sum())
        if dups:
            problems.append(f"{s}: {dups} duplicate dates")
        if bad:
            problems.append(f"{s}: {bad} non-positive closes")
        if df["date"].min() > first_ok:
            problems.append(f"{s}: history starts late ({df['date'].min().date()})")
        rows.append({
            "ticker": s,
            "sector": TICKER_SECTOR.get(s, "Market" if s in MARKET_TICKERS else ""),
            "rows": len(df),
            "first_date": df["date"].min().date(),
            "last_date": df["date"].max().date(),
            "missing_close": int(df["close"].isna().sum()),
        })
    manifest = pd.DataFrame(rows)
    if len(manifest):
        manifest.to_csv(manifest_path, index=False)

    print(f"\nLoaded {len(manifest)}/{len(symbols)} symbols from disk (no internet used).")
    if len(manifest):
        print(f"Date range: {manifest['first_date'].min()} -> {manifest['last_date'].max()}")
        print(f"Rows per symbol: min {manifest['rows'].min()}, max {manifest['rows'].max()}")
    for p in problems:
        print("  WARNING", p)
    return len(manifest) == len(symbols)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END, help="exclusive")
    ap.add_argument("--force", action="store_true", help="re-download cached symbols")
    ap.add_argument("--check", action="store_true", help="offline check only")
    ap.add_argument("--holdout", action="store_true",
                    help="download/check the 50 fresh holdout stocks instead")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between batches")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    symbols = HOLDOUT_TICKERS if args.holdout else ALL_SYMBOLS

    if not args.check:
        todo = symbols if args.force else [s for s in symbols
                                           if not csv_path(s).exists()]
        print(f"{len(symbols) - len(todo)} cached, {len(todo)} to download "
              f"({args.start} to {args.end})")
        failed = download(todo, args.start, args.end, args.batch_size,
                          args.pause, args.retries) if todo else []
        if failed:
            print(f"\nStill failing: {failed}\nRe-run later; cached ones are kept.")

    ok = check(symbols, args.start,
               MANIFEST_PATH.with_name('_manifest_holdout.csv') if args.holdout else MANIFEST_PATH)
    print("\nDONE: every symbol loads from disk." if ok else "\nNOT DONE yet (see above).")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
