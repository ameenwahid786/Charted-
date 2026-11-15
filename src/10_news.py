"""Module 10: news around big drops + recent headlines, saved for the app (needs internet).

    python src/10_news.py          # ~30 min: GDELT allows one request every ~5 seconds

Source: the GDELT Project DOC 2.0 API (free, no key, global news back to 2017).
For each stock we take its worst single-day fall since 2023 (if 5% or worse) and the
S&P 500's worst days, then fetch English headlines published from the day before to the
day after. Headlines are what was published around a drop; they are NOT proof of cause.

Each drop also gets facts computed from prices only: the S&P 500 and same-sector moves
that day (market-wide vs company-specific), whether model v3 was on Warning the day
before, and the move over the next 20 trading days.

Responses are cached in data/news_cache/ (git-ignored) so re-runs resume.
Output: data/app/app_news.json
"""
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from common import DATA_DIR, load_live

APP_DIR = DATA_DIR / "app"
CACHE = DATA_DIR / "news_cache"
API = "https://api.gdeltproject.org/api/v2/doc/doc"
PAUSE = 10.0
UA = {"User-Agent": "Mozilla/5.0 (compatible; charted-student-project)"}
BATCH = 6               # companies per query for the recent-headlines search
DROP_MIN = -0.05        # a stock's worst day must be at least this bad to be explained
MARKET_DROPS = 6


def short_name(name):
    name = re.sub(r"\s*\(.*\)", "", name)
    return re.sub(r"\b(Inc\.|Corp|Company|Companies|Group|Holdings)\b", "", name).strip(" .,")


def gdelt(query, start, end, n=10):
    """One GDELT artlist query (cached). start/end: Timestamps."""
    params = {"query": f"{query} sourcelang:english", "mode": "artlist", "format": "json",
              "maxrecords": str(n), "sort": "hybridrel",
              "startdatetime": start.strftime("%Y%m%d000000"),
              "enddatetime": end.strftime("%Y%m%d235959")}
    url = API + "?" + urllib.parse.urlencode(params)
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / (hashlib.md5(url.encode()).hexdigest() + ".json")
    if f.exists():
        return json.loads(f.read_text())
    for attempt in range(5):
        time.sleep(PAUSE)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                body = r.read().decode("utf-8", "replace")
            data = json.loads(body) if body.strip().startswith("{") else None
        except Exception as e:
            print(f"    retry {attempt + 1}: {str(e)[:60]}", flush=True)
            data = None
        if data is not None:
            arts = data.get("articles", [])
            f.write_text(json.dumps(arts))
            return arts
        time.sleep(60 * (attempt + 1))  # rate-limited (429) or a hiccup: back off properly
    return []


def headlines(arts, must=None, k=5):
    out, seen = [], set()
    for a in arts:
        title = (a.get("title") or "").strip()
        key = re.sub(r"\W+", " ", title.lower()).strip()
        if not title or key in seen or "archives" in key or len(title) < 25:
            continue
        if must and not any(m.lower() in title.lower() for m in must):
            continue
        seen.add(key)
        out.append({"title": title, "url": a.get("url"), "source": a.get("domain"),
                    "date": pd.to_datetime(a.get("seendate"), format="%Y%m%dT%H%M%SZ")
                    .strftime("%Y-%m-%d")})
        if len(out) == k:
            break
    return out


def main():
    meta = json.loads((APP_DIR / "app_meta.json").read_text())
    px = pd.read_parquet(APP_DIR / "app_prices.parquet").sort_values(["ticker", "date"])
    hist = pd.read_parquet(APP_DIR / "app_history.parquet")
    px["ret"] = px.groupby("ticker")["close"].pct_change()
    spx = load_live("^GSPC").set_index("date")["close"]
    spx_ret = spx.pct_change()
    px = px[px["date"] >= "2023-01-03"]
    px["sector"] = px["ticker"].map(meta["sector"])
    sector_ret = px.groupby(["date", "sector"])["ret"].mean()
    latest = pd.Timestamp(meta["as_of"])
    levels = hist.set_index(["ticker", "date"]).sort_index()

    def context(t, d, r):
        m = float(spx_ret.get(d, np.nan))
        sec = float(sector_ret.get((d, meta["sector"][t]), np.nan))
        if m <= -0.02:
            kind = "Market-wide sell-off"
        elif sec <= -0.025 and r - sec > -0.04:
            kind = "Sector-wide fall"
        else:
            kind = "Company-specific"
        g = levels.loc[t]
        before = g[g.index < d].tail(1)
        p = px[(px["ticker"] == t)].set_index("date")["close"]
        after = p[p.index >= d]
        nxt = float(after.iloc[min(20, len(after) - 1)] / after.iloc[0] - 1) if len(after) > 1 else None
        return {"mkt_ret": m, "sector_ret": sec, "kind": kind,
                "level_before": before["level"].iloc[0] if len(before) else None,
                "p_before": float(before["p_v3"].iloc[0]) if len(before) else None,
                "next20": nxt, "next20_days": int(min(20, len(after) - 1))}

    out = {"source": "GDELT Project (gdeltproject.org), DOC 2.0 API",
           "fetched": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "as_of": meta["as_of"],
           "market": [], "stocks": {}}

    # --- S&P 500's worst days ------------------------------------------------------------
    worst = spx_ret[spx_ret.index >= "2023-01-01"].sort_values().head(40)
    picked = []
    for d, r in worst.items():
        if all(abs((d - p).days) > 10 for p in picked):
            picked.append(d)
        if len(picked) == MARKET_DROPS:
            break
    for d in sorted(picked):
        arts = gdelt('("stock market" OR "Wall Street" OR "S&P 500") (stocks OR selloff OR plunge)',
                     d - pd.Timedelta(days=1), d + pd.Timedelta(days=1))
        out["market"].append({"date": d.strftime("%Y-%m-%d"), "ret": float(spx_ret[d]),
                              "headlines": headlines(arts, k=6)})
        print(f"market {d.date()} {spx_ret[d]:+.1%}: {len(out['market'][-1]['headlines'])} headlines")

    # --- recent headlines: several companies per query, matched back by name -----------------
    tickers = sorted(meta["names"])
    names = {t: short_name(meta["names"][t]) for t in tickers}
    musts = {t: ([names[t].split()[0], t] if len(names[t].split()[0]) > 2 else [names[t], t])
             for t in tickers}
    recent_arts = {t: [] for t in tickers}
    for i in range(0, len(tickers), BATCH):
        group = tickers[i:i + BATCH]
        q = "(" + " OR ".join(f'"{names[t]}"' for t in group) + ") (shares OR stock OR investors)"
        arts = gdelt(q, latest - pd.Timedelta(days=7), latest + pd.Timedelta(days=1), n=75)
        for t in group:
            recent_arts[t] = [a for a in arts
                              if any(m.lower() in (a.get("title") or "").lower() for m in musts[t])]
        print(f"recent batch {i // BATCH + 1}: {len(arts)} articles", flush=True)

    # --- each stock: worst day ----------------------------------------------------------------
    for i, t in enumerate(tickers, 1):
        name, must = names[t], musts[t]
        q = f'"{name}" (shares OR stock OR investors)'
        g = px[px["ticker"] == t].dropna(subset=["ret"])
        drops = []
        w = g.nsmallest(1, "ret")
        for _, row in w.iterrows():
            if row["ret"] > DROP_MIN:
                continue
            d = row["date"]
            arts = gdelt(q, d - pd.Timedelta(days=1), d + pd.Timedelta(days=1))
            drops.append({"date": d.strftime("%Y-%m-%d"), "ret": float(row["ret"]),
                          **context(t, d, float(row["ret"])), "headlines": headlines(arts, must)})
        recent = headlines(recent_arts[t], must, k=6)
        out["stocks"][t] = {"drops": drops, "recent": recent}
        print(f"[{i}/{len(tickers)}] {t}: {len(drops)} drop(s), "
              f"{sum(len(x['headlines']) for x in drops)} drop headlines, {len(recent)} recent",
              flush=True)

    (APP_DIR / "app_news.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"Saved {APP_DIR / 'app_news.json'}")


if __name__ == "__main__":
    main()
