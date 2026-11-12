# pumpfun-top5-forecast-01 entry: forecast the 5 largest pump.fun coins by market cap
# at each answer read, from the house's 10-minute marks.
#
# Model (standard library only, fully deterministic):
#   * each coin's log10 market cap (in SOL) is modelled as a random walk with a
#     shrunk per-coin drift (from its own marks) and a shrunk per-coin volatility,
#     projected to every answer read in task.json; the drift is applied for at
#     most 2 hours, curve coins decay steadily, pool coins decay slowly after 2h;
#     the volatility grows with the square root of the horizon.
#   * the coins that will top the chart at a read are mostly coins launched
#     minutes before that read, so every read also simulates NEWBORN contenders:
#     caps resampled from the empirical cap distribution of coins aged under
#     60 minutes in the history (all top-5 debuts in the dev window were coins
#     1-9 minutes old).  A Monte Carlo over candidates + newborns, per read,
#     gives each place's coin probabilities and each place's cap distribution.
#   * the single forecast is the mean over the reads of the per-read
#     distributions (the score is the mean over the reads, and the mean of the
#     per-read distributions is what maximises it).
#   * probabilities are Laplace-smoothed (3% spread over the named coins) so
#     no named coin ever scores the floor; sizes are mixed 90/10 with a broad
#     normal around the place's mean cap so no bin is ever empty.
#
# Determinism: one fixed seed (20261112), candidates iterated in sorted mint
# order, no set iteration, no wall clock, no network, no randomness outside
# the seeded generator.  Two runs on the same history write byte-identical
# output.

import json
import math
import os
import random
import sys
from datetime import datetime, timezone

SEED = 20261112
SCENARIOS = 6000
TOP_NEWBORNS = 3         # newborn contenders simulated per read: the top 3
                          # young coins (age < 60 min) at the read.  Their
                          # caps are the top order statistics of the young
                          # coins visible per mark, sampled from the empirical
                          # young-coin cap distribution via Beta quantiles
SMOOTH_PLACE = 0.03       # Laplace share spread over the named coins
SMOOTH_SIZE_PRIOR = 0.10  # weight of the broad normal mixed into sizes
SIZE_PRIOR_SD = 0.60      # sd (dex) of the broad normal mixed into sizes
MAX_CANDIDATES = 150
MAX_ROWS_PER_PLACE = 100
DRIFT_HOURS = 2.0         # drift is applied for at most this many hours
DRIFT_CAP = 0.15          # the drift may move a coin's log-cap at most this far
CURVE_DECAY = 0.08        # dex per hour the log-cap of a curve coin falls
POOL_DECAY = 0.03         # dex per hour a pool coin falls after 2h
SIGMA0 = 0.10             # base sd (dex) at the cut
VOL_HOURLY_CAP = 0.50     # cap on per-hour volatility (dex)
SIGMA_CAP = 0.90          # cap on total sd (dex) at any horizon


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def floor6(x):
    """Round down to 6 decimals (never -0), like the house starter."""
    r = math.floor(x * 1_000_000) / 1_000_000
    return 0.0 if r == 0 else r


def erf_approx(z):
    return math.erf(z)


def normal_cdf(x, mu, sd):
    if sd <= 0:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1.0 + erf_approx((x - mu) / (sd * math.sqrt(2.0))))


def bin_of(value, edges):
    lo, hi = 0, len(edges)
    while lo < hi:
        mid = (lo + hi) // 2
        if edges[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

    with open(os.path.join(in_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    out_name = task.get("output", "forecast.json")
    n_bins = len(edges) + 1

    marks = []
    with open(os.path.join(in_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list):
                marks.append(m)
    if not marks:
        raise ValueError("the history holds no ok mark")

    mark_times = [parse_time(m["at"]) for m in marks]
    cut_time = mark_times[-1]
    read_times = []
    for r in task.get("reads", []) or []:
        read_times.append(parse_time(r))
    if not read_times:
        base = parse_time(task["history_until"]) if task.get("history_until") else cut_time
        for h in range(1, 8):
            read_times.append(base.replace(tzinfo=timezone.utc))
        # hourly reads starting 1h after the cut, fallback when task has no reads
        read_times = []
        for h in range(1, 8):
            read_times.append(datetime.fromtimestamp(cut_time.timestamp() + 3600.0 * h, tz=timezone.utc))
    horizons = [(rt - cut_time).total_seconds() / 3600.0 for rt in read_times]
    horizons = [h if h > 0 else 0.05 for h in horizons]

    # ---- per-coin series from the marks -----------------------------------
    # mint -> {times (hours since first mark), logcaps, venue, complete, created}
    series = {}
    for mi, m in enumerate(marks):
        t_h = (mark_times[mi] - mark_times[0]).total_seconds() / 3600.0
        for row in m["rows"]:
            mint = row["mint"]
            cap_lamports = int(row["market_cap_lamports"])
            if cap_lamports <= 0:
                continue
            x = math.log10(cap_lamports) - 9.0  # log10 of the cap in SOL
            e = series.get(mint)
            if e is None:
                e = {"t": [], "x": [], "venue": row.get("venue", "curve"),
                     "complete": bool(row.get("complete", False)),
                     "created": row.get("created_at")}
                series[mint] = e
            e["t"].append(t_h)
            e["x"].append(x)
            e["venue"] = row.get("venue", e["venue"])
            e["complete"] = bool(row.get("complete", e["complete"]))

    last_rows = marks[-1]["rows"]
    last_by_mint = {}
    for row in last_rows:
        cap_lamports = int(row["market_cap_lamports"])
        if cap_lamports > 0:
            last_by_mint[row["mint"]] = math.log10(cap_lamports) - 9.0

    # newborn empirical cap distribution: log-caps of coins aged under 60
    # minutes at any mark (they are the population the next reads draw from)
    newborn_caps = []
    for mi, m in enumerate(marks):
        at = mark_times[mi]
        for row in m["rows"]:
            created = row.get("created_at")
            if not created:
                continue
            age_h = (at - parse_time(created)).total_seconds() / 3600.0
            if 0.0 <= age_h < 1.0:
                cap_lamports = int(row["market_cap_lamports"])
                if cap_lamports > 0:
                    newborn_caps.append(math.log10(cap_lamports) - 9.0)
    newborn_caps.sort()
    if not newborn_caps:
        newborn_caps = [math.log10(int(r["market_cap_lamports"])) - 9.0
                        for r in last_rows if int(r["market_cap_lamports"]) > 0]
        newborn_caps.sort()
    # young coins visible per mark: the size of the population the top
    # newborn order statistics are drawn from
    n_young_per_mark = max(50, len(newborn_caps) // max(1, len(marks)))

    # ---- candidate selection -------------------------------------------------
    # relevance: current log-cap plus recent momentum; keep the strongest set
    def momentum(e):
        if len(e["x"]) < 2:
            return 0.0
        return e["x"][-1] - e["x"][0]

    scored = []
    for mint, e in series.items():
        x = e["x"][-1]
        scored.append((x + 0.35 * momentum(e), mint))
    scored.sort(key=lambda kv: (-kv[0], kv[1]))
    # top by relevance, plus every coin in the last mark's top 20 by cap
    cand_mints = [mint for _, mint in scored[:MAX_CANDIDATES]]
    top20 = sorted(last_by_mint.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
    for mint, _ in top20:
        if mint not in cand_mints:
            cand_mints.append(mint)
    cand_mints.sort()

    # ---- per-coin drift and volatility ---------------------------------------
    venue_prior_vol = {"pool": 0.34, "curve": 0.094, "migrating": 0.20}
    coins = []
    for mint in cand_mints:
        e = series[mint]
        xs = e["x"]
        ts = e["t"]
        x = xs[-1]
        # OLS slope of log-cap on time (dex per hour), shrunk toward 0;
        # a two-point slope is mostly noise, so the shrinkage is strong,
        # and the total drift a coin may carry is capped (DRIFT_CAP)
        n = len(xs)
        if n >= 2:
            mt = sum(ts) / n
            mx = sum(xs) / n
            num = sum((t - mt) * (xv - mx) for t, xv in zip(ts, xs))
            den = sum((t - mt) ** 2 for t in ts)
            slope = num / den if den > 0 else 0.0
            drift = slope * (n / (n + 6.0))
        else:
            drift = 0.0
        # clamp drift: a coin cannot keep compounding forever
        drift = max(-0.25, min(0.25, drift))
        # realised 10-minute volatility, shrunk toward the venue prior
        diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        if diffs:
            mean_d = sum(diffs) / len(diffs)
            var = sum((d - mean_d) ** 2 for d in diffs) / len(diffs)
            vol10 = math.sqrt(var)
            prior = venue_prior_vol.get(e["venue"], 0.20)
            vol10 = (len(diffs) * vol10 + 2.0 * prior) / (len(diffs) + 2.0)
        else:
            vol10 = venue_prior_vol.get(e["venue"], 0.20)
        vol_hourly = min(VOL_HOURLY_CAP, max(0.15, vol10 * math.sqrt(6.0)))
        coins.append({
            "mint": mint,
            "x": x,
            "drift": drift,
            "vol_hourly": vol_hourly,
            "venue": e["venue"],
            "complete": e["complete"],
        })

    # ---- Monte Carlo over the reads ------------------------------------------
    rng = random.Random(SEED)
    n_cand = len(coins)
    # place_counts[read][place] -> {mint: count}; size_counts[read][place] -> [bin counts]
    place_counts = []
    size_counts = []
    newborn_wins = [0] * len(horizons)
    for ri, h in enumerate(horizons):
        pc = [dict() for _ in range(places)]
        sc = [[0] * n_bins for _ in range(places)]
        # per-coin forecast at this horizon
        mus = []
        sds = []
        for c in coins:
            raw_drift = c["drift"] * min(h, DRIFT_HOURS)
            drift_part = max(-DRIFT_CAP, min(DRIFT_CAP, raw_drift))
            if c["venue"] == "curve":
                decay = CURVE_DECAY * h
            else:
                decay = POOL_DECAY * max(0.0, h - DRIFT_HOURS)
            mu = c["x"] + drift_part - decay
            sd = math.sqrt(SIGMA0 * SIGMA0 + (c["vol_hourly"] ** 2) * h)
            mus.append(mu)
            sds.append(min(SIGMA_CAP, sd))
        for _s in range(SCENARIOS):
            # candidate caps
            caps = []
            for i in range(n_cand):
                caps.append((rng.gauss(mus[i], sds[i]), i))
            # newborn contenders: the top 3 young coins at this read.  The
            # j-th largest of K draws from a distribution F has F(x) ~
            # Beta(K-j+1, j); sample the Beta quantile via gamma variates and
            # map it through the empirical young-coin caps.
            nb_top = []
            for j in range(1, TOP_NEWBORNS + 1):
                g1 = rng.gammavariate(n_young_per_mark - j + 1, 1.0)
                g2 = rng.gammavariate(j, 1.0)
                v = g1 / (g1 + g2) if g1 + g2 > 0 else 0.5
                pos = v * (len(newborn_caps) - 1)
                lo = int(pos)
                hi = min(lo + 1, len(newborn_caps) - 1)
                nb_top.append(newborn_caps[lo] + (newborn_caps[hi] - newborn_caps[lo]) * (pos - lo))
            # rank candidates and newborns together; newborns are not named,
            # they only take places (recorded as newborn wins)
            caps.sort(key=lambda kv: -kv[0])
            ci = 0
            ni = 0
            placed = 0
            while placed < places:
                if ni < len(nb_top) and (ci >= len(caps) or nb_top[ni] > caps[ci][0]):
                    cap_val = nb_top[ni]
                    ni += 1
                    newborn_wins[ri] += 1
                else:
                    cap_val, i = caps[ci]
                    ci += 1
                    pc[placed][coins[i]["mint"]] = pc[placed].get(coins[i]["mint"], 0) + 1
                b = bin_of(cap_val, edges)
                sc[placed][b] += 1
                placed += 1
        place_counts.append(pc)
        size_counts.append(sc)

    n_reads = len(horizons)
    # ---- average over reads, build the forecast ------------------------------
    probabilities = []
    sizes = []
    top5 = []
    for k in range(places):
        # mean frequency per candidate at this place
        freq = {}
        for r in range(n_reads):
            for mint, cnt in place_counts[r][k].items():
                freq[mint] = freq.get(mint, 0.0) + cnt / (SCENARIOS * n_reads)
        order = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        # Laplace smoothing: 3% spread evenly over the coins named here
        named = [mint for mint, _ in order[:MAX_ROWS_PER_PLACE]]
        n_named = len(named)
        for mint in named:
            p = 0.97 * freq.get(mint, 0.0) + SMOOTH_PLACE / n_named if n_named else 0.0
            p = floor6(p)
            if p > 0:
                probabilities.append({"mint": mint, "place": k + 1, "p": p})
        # top-5 point guess: highest frequency at each place, distinct coins
        for mint, _ in order:
            if mint not in top5:
                top5.append(mint)
                break
        # sizes: mean MC histogram, mixed 90/10 with a broad normal.
        # The place's mean cap comes from the mass-weighted bin centres.
        tot = [0.0] * n_bins
        for r in range(n_reads):
            for b in range(n_bins):
                tot[b] += size_counts[r][k][b] / (SCENARIOS * n_reads)
        num = 0.0
        den = 0.0
        for b in range(n_bins):
            lo = -10.0 if b == 0 else edges[b - 1]
            hi = 10.0 if b == n_bins - 1 else edges[b]
            centre = (lo + hi) / 2.0
            num += tot[b] * centre
            den += tot[b]
        mu_size = num / den if den > 0 else 0.0
        for b in range(n_bins):
            lo = -float("inf") if b == 0 else edges[b - 1]
            hi = float("inf") if b == n_bins - 1 else edges[b]
            prior = normal_cdf(hi, mu_size, SIZE_PRIOR_SD) - normal_cdf(lo, mu_size, SIZE_PRIOR_SD)
            p = 0.9 * tot[b] + SMOOTH_SIZE_PRIOR * prior
            p = floor6(p)
            if p > 0:
                sizes.append({"place": k + 1, "bin": b, "p": p})
    if len(top5) < places:
        # fill from the most frequent coins overall
        overall = sorted(
            {mint: sum(freq for freq in [0]) for mint in cand_mints}.items(),
            key=lambda kv: kv[0],
        )
        for mint, _ in overall:
            if mint not in top5:
                top5.append(mint)
            if len(top5) == places:
                break

    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["mint"]))
    sizes.sort(key=lambda r: (r["place"], r["bin"]))
    doc = {"probabilities": probabilities, "top5": top5[:places], "sizes": sizes}
    text = json.dumps(doc, sort_keys=True) + "\n"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as f:
        f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
