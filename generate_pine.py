"""
Scheduled level generator — run by GitHub Actions.

Writes, for each configured instrument:
  levels/gex_<LABEL>.pine   ready-to-paste Pine script, levels baked in
  levels/gex_<LABEL>.csv    the same levels as a table
  levels/README.md          a quick summary you can read on your phone

Run locally:  python generate_pine.py
"""

import datetime as dt
import pathlib
import sys
import traceback

import gex_core as core

# Which instruments to generate on each run, and with what expiration window.
# Metals need a longer window — GLD and SLV rarely list weeklies.
TARGETS = [
    ("Nasdaq 100  ·  QQQ → NQ / MNQ", 7),
    ("Gold  ·  GLD → GC / MGC", 30),
]

OUT = pathlib.Path("levels")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc)
    summary = [
        "# Latest GEX levels",
        "",
        f"Generated {stamp.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]
    failures = 0

    for asset_name, dte in TARGETS:
        label = core.ASSETS[asset_name]["label"]
        etf = core.ASSETS[asset_name]["etf"]
        has_index = core.ASSETS[asset_name]["index"] is not None
        mode = "Ratio + basis" if has_index else "Pure ratio"

        try:
            res = core.analyze(asset_name, max_dte=dte, conv_mode=mode)
        except Exception:
            traceback.print_exc()
            summary += [f"## {label}", "", "Generation raised an exception.", ""]
            failures += 1
            continue

        if "error" in res:
            print(f"{label}: {res['error']}", file=sys.stderr)
            summary += [f"## {label}", "", res["error"], ""]
            failures += 1
            continue

        (OUT / f"gex_{label}.pine").write_text(core.build_pine(res), encoding="utf-8")

        tbl = core.levels_table(res)
        tbl.to_csv(OUT / f"gex_{label}.csv", index=False)

        regime = "−Γ negative gamma" if res["negative_gamma"] else "+Γ positive gamma"
        summary += [
            f"## {label}  ·  {regime}",
            "",
            f"{etf} {res['etf_px']:,.2f} · {label} {res['fut_px']:,.2f} · "
            f"net GEX ${res['total_gex']/1e9:,.2f}B per 1% · ≤{dte} DTE",
            "",
            tbl.to_markdown(index=False),
            "",
            f"Script: [`gex_{label}.pine`](gex_{label}.pine)",
            "",
        ]
        print(f"{label}: wrote {len(tbl)} levels")

    (OUT / "README.md").write_text("\n".join(summary), encoding="utf-8")

    # Don't fail the whole run if one instrument was unavailable; only if all were.
    return 1 if failures == len(TARGETS) else 0


if __name__ == "__main__":
    sys.exit(main())
