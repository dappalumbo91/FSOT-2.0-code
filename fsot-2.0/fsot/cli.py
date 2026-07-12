#!/usr/bin/env python3
"""FSOT 2.0 monolith CLI — run | predict | verify | domains."""

from __future__ import annotations

import argparse
import json
import sys

from fsot import __version__
from fsot.adapter import load_compute
from fsot.predict import predict
from fsot.verify import run_portable_verify


def cmd_run(args: argparse.Namespace) -> int:
    mod = load_compute()
    argv = []
    if args.validate:
        argv.append("--validate")
    if args.wave is not None:
        argv.extend(["--wave", str(args.wave)])
    if args.seeds:
        argv.append("--seeds")
    if args.full:
        pass
    elif not argv:
        argv.append("--validate")
    old = sys.argv
    try:
        sys.argv = ["fsot"] + argv
        mod.main()
    finally:
        sys.argv = old
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    out = predict(
        args.measured,
        property_name=args.property,
        domain=args.domain,
        formula=args.formula,
    )
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"measured:  {out['measured']}")
        print(f"computed:  {out['computed']}")
        print(f"error_pct: {out['error_pct']}%")
        print(f"domain:    {out['fsot_domain']} (scalar={out['fsot_scalar']})")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    report = run_portable_verify(verbose=not args.quiet)
    if args.json:
        print(json.dumps(report, indent=2))
    if not report["ok"]:
        return 1
    return 0


def cmd_domains(args: argparse.Namespace) -> int:
    mod = load_compute()
    rows = []
    for name in sorted(mod.DOMAINS):
        cfg = mod.DOMAINS[name]
        scalar = float(mod.domain_scalar(name))
        rows.append(
            {
                "name": name,
                "D_eff": cfg.D_eff,
                "observed": cfg.observed,
                "scalar": round(scalar, 8),
            }
        )
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            print(f"{row['name']:28s} D_eff={row['D_eff']:2d}  S={row['scalar']:+.6f}")
        print(f"\n{len(rows)} domains")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fsot",
        description="FSOT 2.0 — interpretable monolithic runtime (Python + mpmath)",
    )
    parser.add_argument("--version", action="version", version=f"FSOT {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run compute engine (validation suite, waves, full report)")
    run_p.add_argument("--validate", action="store_true", help="Run 28-check validation suite")
    run_p.add_argument("--wave", type=int, default=None, help="Run a single wave (1-10)")
    run_p.add_argument("--seeds", action="store_true", help="Print foundational seeds")
    run_p.add_argument("--full", action="store_true", help="Full report (default if no flags)")
    run_p.set_defaults(func=cmd_run)

    pred_p = sub.add_parser("predict", help="Predict one observable from measured value")
    pred_p.add_argument("measured", type=float, help="Measured / observed value")
    pred_p.add_argument("--property", default="value", help="Property name for routing")
    pred_p.add_argument("--domain", default="Biology", help="FSOT domain name")
    pred_p.add_argument("--formula", default=None, help="Chemical formula for MW prediction")
    pred_p.add_argument("--json", action="store_true")
    pred_p.set_defaults(func=cmd_predict)

    ver_p = sub.add_parser("verify", help="Portable verify — constants + bundled benchmarks")
    ver_p.add_argument("--json", action="store_true")
    ver_p.add_argument("--quiet", action="store_true")
    ver_p.set_defaults(func=cmd_verify)

    dom_p = sub.add_parser("domains", help="List all 35 FSOT domains and scalars")
    dom_p.add_argument("--json", action="store_true")
    dom_p.set_defaults(func=cmd_domains)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())