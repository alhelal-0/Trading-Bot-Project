# Bots on TradeLocker are implemented in Python using the Backtrader framework.
# You can find more examples and the complete documentation at https://www.backtrader.com/

# EMA crossover trend following:
#   - Fast EMA (9) and slow EMA (21) on closing prices; the bar interval
#     comes from the chart/feed the bot runs on (built for 5m candles).
#   - LONG on the candle where the fast EMA crosses ABOVE the slow EMA;
#     SHORT on the candle of the cross below. Signals fire only on the
#     crossing candle itself, never on every fast>slow candle.
#   - Minimum separation filter: the EMAs must be at least `min_sep_pct`
#     percent of price apart after the cross -- skips whipsaw crosses in
#     choppy/flat markets.
#   - Stop loss: `stop_atr_mult` x ATR from entry.
#   - Exits: with `use_trailing` on (default), winners are NOT capped --
#     once the trade reaches `trail_trigger_r` (1R) of profit the stop
#     trails the best price by `trail_atr_mult` x ATR and only tightens,
#     letting trend days run. With trailing off, a fixed `rr_ratio` (1:2)
#     take-profit applies instead.
#   - Position size risks `risk_per_trade_pct` (1%) of account equity.

import math

import backtrader as bt


class EmaCrossoverTrendStrategy(bt.Strategy):
    params = {
        "fast_period": 9,
        "slow_period": 21,
        "min_sep_pct": 0.004,        # EMA distance must exceed this % of price.
                                     # NOTE: on the crossing candle the EMAs are
                                     # nearly equal by definition, so this is a
                                     # small number -- calibrate per instrument
        "atr_period": 14,
        "stop_atr_mult": 2.0,
        "rr_ratio": 2.0,             # take profit at this multiple of risk
        "use_trailing": True,        # trail after the trade reaches trigger R
        "trail_trigger_r": 1.0,      # start trailing at this many R of profit
        "trail_atr_mult": 1.5,       # trail distance in ATRs
        "risk_per_trade_pct": 1.0,
        "allow_long": True,
        "allow_short": True,
        "min_size": 0.01,
        "verbose": False,
    }

    def __init__(self) -> None:
        self.ema_fast = bt.indicators.ExponentialMovingAverage(
            self.data.close, period=self.p.fast_period)
        self.ema_slow = bt.indicators.ExponentialMovingAverage(
            self.data.close, period=self.p.slow_period)
        self.crossover = bt.indicators.CrossOver(self.ema_fast, self.ema_slow)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)

        self.order = None
        self.order_ctx = None
        self.trade_meta = None
        self.stop_price = None
        self.target_price = None
        self.extreme = None          # best price since entry
        self.trail_active = False

        self.trade_records = []

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def log(self, txt):
        if self.p.verbose:
            print(f"{self.data.datetime.datetime(0)} | {txt}")

    def _tick_size(self):
        try:
            tick = float(self.data.tick_size)
            if tick > 0:
                return tick
        except Exception:
            pass
        return 0.0001

    def _lot_step(self):
        try:
            step = float(self.data.lot_step)
            if step > 0:
                return step
        except Exception:
            pass
        return 0.01

    def _contract_size(self):
        try:
            size = float(self.data.lot_size)
            if size > 0:
                return size
        except Exception:
            pass
        return 1.0

    def _calc_size(self, stop_distance):
        equity = float(self.broker.getvalue())
        risk_cash = equity * (self.p.risk_per_trade_pct / 100.0)
        unit_risk = max(stop_distance, self._tick_size()) * self._contract_size()
        if unit_risk <= 0:
            return 0.0
        step = self._lot_step()
        size = math.floor((risk_cash / unit_risk) / step) * step
        return size if size >= self.p.min_size else 0.0

    def _warmed_up(self):
        needed = max(self.p.slow_period, self.p.atr_period) + 1
        return len(self.data) > needed

    # ------------------------------------------------------------
    # Order / trade bookkeeping
    # ------------------------------------------------------------
    def notify_order(self, order):
        # Partial fills stay in flight; act only on the final status
        if order.status in [order.Submitted, order.Accepted, order.Partial]:
            return

        if self.order is None or order.ref != self.order.ref:
            return

        ctx = self.order_ctx

        if order.status == order.Completed:
            fill_price = float(order.executed.price)

            if ctx and ctx["kind"] == "entry":
                direction = ctx["direction"]
                stop_distance = ctx["stop_distance"]
                size = abs(float(order.executed.size))

                self.trade_meta = dict(
                    direction=direction,
                    entry_dt=self.data.datetime.datetime(0),
                    entry_price=fill_price,
                    size=size,
                    stop_distance=stop_distance,
                    risk_cash=stop_distance * size * self._contract_size(),
                )
                if direction == 1:
                    self.stop_price = fill_price - stop_distance
                else:
                    self.stop_price = fill_price + stop_distance

                # With the trailing stop on, winners are NOT capped by a
                # fixed target -- the trail is the exit. The fixed R:R
                # target applies only when trailing is disabled.
                if self.p.use_trailing:
                    self.target_price = None
                else:
                    self.target_price = (
                        fill_price + stop_distance * self.p.rr_ratio
                        if direction == 1
                        else fill_price - stop_distance * self.p.rr_ratio
                    )
                self.extreme = fill_price
                self.trail_active = False

                self.log(
                    f"{'LONG' if direction == 1 else 'SHORT'} OPEN | "
                    f"Entry:{fill_price:.5f} Stop:{self.stop_price:.5f} "
                    f"Target:{'trail' if self.target_price is None else f'{self.target_price:.5f}'} "
                    f"Size:{size:.4f}"
                )

            elif ctx and ctx["kind"] == "exit" and self.trade_meta is not None:
                meta = self.trade_meta
                pnl = (
                    (fill_price - meta["entry_price"])
                    * meta["direction"] * meta["size"] * self._contract_size()
                )
                risk = meta["risk_cash"]
                self.trade_records.append(dict(
                    setup="ema-cross",
                    day_type="-",
                    direction="long" if meta["direction"] == 1 else "short",
                    entry_dt=meta["entry_dt"],
                    exit_dt=self.data.datetime.datetime(0),
                    pnl=pnl,
                    pnlcomm=pnl,
                    risk_cash=risk,
                    r_multiple=pnl / risk if risk > 0 else 0.0,
                    reason=ctx.get("reason", "exit"),
                ))
                self.log(f"CLOSED | {ctx.get('reason')} | PnL:{pnl:.2f}")
                self.trade_meta = None
                self.stop_price = None
                self.target_price = None
                self.extreme = None
                self.trail_active = False

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"ORDER FAILED | Status: {order.getstatusname()}")
            if ctx and ctx["kind"] == "entry":
                self.trade_meta = None

        self.order = None
        self.order_ctx = None

    # ------------------------------------------------------------
    # Trading logic, executed on each candle
    # ------------------------------------------------------------
    def next(self) -> None:
        if not self._warmed_up():
            return
        if self.order is not None:
            return

        h = float(self.data.high[0])
        l = float(self.data.low[0])
        c = float(self.data.close[0])
        atr_value = float(self.atr[0])
        cross = int(self.crossover[0])   # +1 / -1 only on the crossing candle

        # --------------------------------------------------------
        # Manage the open position
        # --------------------------------------------------------
        if self.position.size != 0 and self.trade_meta is not None:
            meta = self.trade_meta
            d = meta["direction"]

            if self.p.use_trailing:
                if d == 1:
                    self.extreme = max(self.extreme, h)
                    favorable = self.extreme - meta["entry_price"]
                    if (
                        not self.trail_active
                        and favorable >= meta["stop_distance"] * self.p.trail_trigger_r
                    ):
                        self.trail_active = True
                        self.log("TRAIL ACTIVE | 1R reached, stop now trails")
                    if self.trail_active:
                        self.stop_price = max(
                            self.stop_price,
                            self.extreme - self.p.trail_atr_mult * atr_value)
                else:
                    self.extreme = min(self.extreme, l)
                    favorable = meta["entry_price"] - self.extreme
                    if (
                        not self.trail_active
                        and favorable >= meta["stop_distance"] * self.p.trail_trigger_r
                    ):
                        self.trail_active = True
                        self.log("TRAIL ACTIVE | 1R reached, stop now trails")
                    if self.trail_active:
                        self.stop_price = min(
                            self.stop_price,
                            self.extreme + self.p.trail_atr_mult * atr_value)

            if d == 1:
                if l <= self.stop_price:
                    reason = "trail-stop" if self.trail_active else "atr-stop"
                    self.order_ctx = dict(kind="exit", reason=reason)
                elif self.target_price is not None and h >= self.target_price:
                    self.order_ctx = dict(kind="exit", reason="rr-target")
            else:
                if h >= self.stop_price:
                    reason = "trail-stop" if self.trail_active else "atr-stop"
                    self.order_ctx = dict(kind="exit", reason=reason)
                elif self.target_price is not None and l <= self.target_price:
                    self.order_ctx = dict(kind="exit", reason="rr-target")
            if self.order_ctx:
                self.order = self.close()
            return

        # --------------------------------------------------------
        # Entries: only on the crossing candle
        # --------------------------------------------------------
        if cross == 0 or atr_value <= 0:
            return

        # Minimum separation: skip whipsaw crosses in flat markets
        separation_pct = 100.0 * abs(float(self.ema_fast[0]) - float(self.ema_slow[0])) / c
        if separation_pct < self.p.min_sep_pct:
            self.log(
                f"SIGNAL BLOCKED | EMA separation {separation_pct:.4f}% "
                f"< {self.p.min_sep_pct}% (whipsaw filter)")
            return

        direction = 1 if cross > 0 else -1
        if direction == 1 and not self.p.allow_long:
            return
        if direction == -1 and not self.p.allow_short:
            return

        stop_distance = self.p.stop_atr_mult * atr_value
        size = self._calc_size(stop_distance)
        if size <= 0:
            return

        self.order_ctx = dict(kind="entry", direction=direction, stop_distance=stop_distance)
        self.log(
            f"{'LONG' if direction == 1 else 'SHORT'} SIGNAL | "
            f"EMA{self.p.fast_period}x{self.p.slow_period} cross | "
            f"Sep:{separation_pct:.4f}% Size:{size:.4f}")
        self.order = self.buy(size=size) if direction == 1 else self.sell(size=size)

    # ====================================================================================
    # params_metadata is optional -- it is used to configure the "Run Backtest/Bot" modal.
    params_metadata = {
        "fast_period": {
            "label": "Fast EMA",
            "helper_text": "Period of the fast EMA",
            "value_type": "int",
        },
        "slow_period": {
            "label": "Slow EMA",
            "helper_text": "Period of the slow EMA",
            "value_type": "int",
        },
        "min_sep_pct": {
            "label": "Min EMA separation %",
            "helper_text": "Skip crosses where the EMAs are closer than this % of price. "
                           "Keep it small: on the crossing candle the EMAs are nearly "
                           "equal by definition (e.g. 0.004 on NAS100 5m)",
            "value_type": "float",
        },
        "stop_atr_mult": {
            "label": "Stop (ATRs)",
            "helper_text": "Stop loss distance from entry, in ATRs",
            "value_type": "float",
        },
        "rr_ratio": {
            "label": "Reward:risk",
            "helper_text": "Fixed take-profit multiple, used only when the "
                           "trailing stop is off",
            "value_type": "float",
        },
        "use_trailing": {
            "label": "Trailing stop",
            "helper_text": "Trail after the trigger profit and leave winners "
                           "uncapped (recommended for trend following)",
            "value_type": "bool",
        },
        "trail_trigger_r": {
            "label": "Trail trigger (R)",
            "helper_text": "Start trailing after this many R of open profit",
            "value_type": "float",
        },
        "trail_atr_mult": {
            "label": "Trail (ATRs)",
            "helper_text": "Trailing stop distance from the best price, in ATRs",
            "value_type": "float",
        },
        "risk_per_trade_pct": {
            "label": "Risk per trade %",
            "helper_text": "Percent of account equity risked on each trade",
            "value_type": "float",
        },
    }
