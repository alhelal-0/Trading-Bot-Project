"""
Measurement harness for Nas100PriceActionBot.

Runs a backtest on 5m CSV data and breaks the results down by setup,
day type, entry hour, direction, and weekday, so it's obvious which
parts of the strategy earn and which lose.

Usage:
    python backtest_report.py --data data/nas100_5m.csv
    python backtest_report.py --data data/nas100_5m.csv --ny-only
    python backtest_report.py --data data/nas100_5m.csv --cash 25000 --commission 0.0002

Expected CSV columns (header row optional, see --separator/--dtformat):
    datetime, open, high, low, close, volume
    e.g. 2026-03-05 09:30:00,24801.2,24855.0,24790.1,24830.6,1234
"""

import argparse
import importlib
from collections import defaultdict

import backtrader as bt


def load_strategy(module_name):
    """Import a strategy module by name and return its bt.Strategy subclass."""
    mod = importlib.import_module(module_name)
    classes = [
        obj for obj in vars(mod).values()
        if isinstance(obj, type) and issubclass(obj, bt.Strategy) and obj is not bt.Strategy
    ]
    if len(classes) != 1:
        names = ", ".join(c.__name__ for c in classes) or "none"
        raise SystemExit(f"Expected exactly one strategy class in {module_name}, found: {names}")
    return classes[0]


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
def load_data(args):
    return bt.feeds.GenericCSVData(
        dataname=args.data,
        dtformat=args.dtformat,
        separator=args.separator,
        timeframe=bt.TimeFrame.Minutes,
        compression=args.compression,
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1,
        headers=args.headers,
    )


# ------------------------------------------------------------
# Reporting
# ------------------------------------------------------------
def bucket_stats(records):
    wins = [r for r in records if r["pnlcomm"] > 0]
    losses = [r for r in records if r["pnlcomm"] <= 0]

    gross_win = sum(r["pnlcomm"] for r in wins)
    gross_loss = abs(sum(r["pnlcomm"] for r in losses))

    return dict(
        trades=len(records),
        win_pct=100.0 * len(wins) / len(records) if records else 0.0,
        net_pnl=sum(r["pnlcomm"] for r in records),
        avg_r=sum(r["r_multiple"] for r in records) / len(records) if records else 0.0,
        total_r=sum(r["r_multiple"] for r in records),
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
    )


def print_breakdown(title, records, key_fn):
    groups = defaultdict(list)
    for r in records:
        groups[key_fn(r)].append(r)

    print(f"\n--- {title} ---")
    header = f"{'group':<16}{'trades':>7}{'win%':>8}{'netPnL':>12}{'avgR':>8}{'totR':>8}{'PF':>8}"
    print(header)
    print("-" * len(header))

    for key in sorted(groups, key=lambda k: str(k)):
        s = bucket_stats(groups[key])
        pf = f"{s['profit_factor']:.2f}" if s["profit_factor"] != float("inf") else "inf"
        print(
            f"{str(key):<16}{s['trades']:>7}{s['win_pct']:>7.1f}%"
            f"{s['net_pnl']:>12.2f}{s['avg_r']:>8.2f}{s['total_r']:>8.2f}{pf:>8}"
        )


def print_report(records, start_cash, end_value, drawdown):
    print("\n" + "=" * 64)
    print("BACKTEST REPORT")
    print("=" * 64)

    if not records:
        print("No trades were taken. Nothing to report.")
        return

    s = bucket_stats(records)
    pf = f"{s['profit_factor']:.2f}" if s["profit_factor"] != float("inf") else "inf"

    print(f"Start cash:      {start_cash:,.2f}")
    print(f"End value:       {end_value:,.2f}")
    print(f"Net PnL:         {end_value - start_cash:,.2f}")
    print(f"Max drawdown:    {drawdown.max.drawdown:.2f}%  ({drawdown.max.moneydown:,.2f})")
    print(f"Trades:          {s['trades']}")
    print(f"Win rate:        {s['win_pct']:.1f}%")
    print(f"Expectancy:      {s['avg_r']:.2f} R per trade")
    print(f"Total R:         {s['total_r']:.2f}")
    print(f"Profit factor:   {pf}")

    print_breakdown("By setup", records, lambda r: r["setup"])
    print_breakdown("By day type", records, lambda r: r["day_type"])
    print_breakdown("By entry hour", records, lambda r: f"{r['entry_dt'].hour:02d}:00")
    print_breakdown("By direction", records, lambda r: r["direction"])
    print_breakdown("By weekday", records, lambda r: r["entry_dt"].strftime("%a"))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Backtest a strategy module and report per-setup stats")
    parser.add_argument("--data", required=True, help="Path to 5m OHLCV CSV")
    parser.add_argument("--module", default="nas100_price_action_bot",
                        help="Strategy module to test, e.g. 1_2_RRtrade")
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--commission", type=float, default=0.0, help="Commission as a fraction, e.g. 0.0002")
    parser.add_argument("--leverage", type=float, default=100.0,
                        help="Account leverage (CFD/futures style); without it longs get margin-rejected")
    parser.add_argument("--dtformat", default="%Y-%m-%d %H:%M:%S")
    parser.add_argument("--separator", default=",")
    parser.add_argument("--compression", type=int, default=5, help="Bar size in minutes")
    parser.add_argument("--headers", action="store_true", help="CSV has a header row")
    parser.add_argument("--ny-only", action="store_true", help="Restrict to NY session (trade_all_hours=False)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-bar strategy logs")
    args = parser.parse_args()

    strategy_cls = load_strategy(args.module)
    print(f"Strategy: {strategy_cls.__name__} (from {args.module})")

    skwargs = dict(verbose=not args.quiet)
    # Only strategies that declare session control get the NY-only flag
    if hasattr(strategy_cls.params, "trade_all_hours"):
        skwargs["trade_all_hours"] = not args.ny_only

    cerebro = bt.Cerebro()
    cerebro.adddata(load_data(args))
    cerebro.addstrategy(strategy_cls, **skwargs)
    cerebro.broker.setcash(args.cash)
    cerebro.broker.setcommission(commission=args.commission, leverage=args.leverage)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

    results = cerebro.run()
    strat = results[0]

    print_report(
        records=strat.trade_records,
        start_cash=args.cash,
        end_value=cerebro.broker.getvalue(),
        drawdown=strat.analyzers.drawdown.get_analysis(),
    )


if __name__ == "__main__":
    main()
