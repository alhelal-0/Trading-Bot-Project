# Bots on TradeLocker are implemented in Python using the Backtrader framework.
# You can find more examples and the complete documentation at https://www.backtrader.com/

# MACD signal-line cross:
#   - MACD line = EMA(12) - EMA(26); signal line = EMA(9) of MACD.
#   - LONG when MACD crosses above the signal line; SHORT on the cross
#     below.
#   - Histogram-strength filter: the cross only counts if the histogram
#     magnitude exceeds `hist_min_atr` x ATR -- weak, near-zero crosses
#     are skipped.
#   - Trend filter: longs only above the `trend_period` (200) EMA,
#     shorts only below it.
#   - Divergence detection (price higher high vs MACD lower high, and
#     the mirror): always logged as a warning; set `divergence_block`
#     to also veto signals that fight a fresh divergence.
#   - Stop loss `stop_atr_mult` x ATR from entry; take profit at
#     `rr_ratio` x the risk; optional exit on the opposite cross.
#   - Position size risks `risk_per_trade_pct` (1%) of account equity.

import math

import backtrader as bt


class MacdSignalCrossStrategy(bt.Strategy):
    params = {
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "hist_min_atr": 0.05,        # histogram magnitude must exceed this x ATR
        "trend_period": 200,
        "use_trend_filter": True,
        "divergence_lookback": 30,   # bars scanned for divergence pivots
        "divergence_block": False,   # True: veto signals against a divergence
        "atr_period": 14,
        "stop_atr_mult": 2.0,
        "rr_ratio": 2.0,
        "exit_on_opposite_cross": True,
        "risk_per_trade_pct": 1.0,
        "allow_long": True,
        "allow_short": True,
        "min_size": 0.01,
        "verbose": False,
    }

    def __init__(self) -> None:
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal,
        )
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
        self.ema_trend = bt.indicators.ExponentialMovingAverage(
            self.data.close, period=self.p.trend_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)

        self.order = None
        self.order_ctx = None
        self.trade_meta = None
        self.stop_price = None
        self.target_price = None

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
        needed = max(
            self.p.trend_period,
            self.p.macd_slow + self.p.macd_signal,
            self.p.atr_period,
            self.p.divergence_lookback + 2,
        ) + 1
        return len(self.data) > needed

    # ------------------------------------------------------------
    # Divergence detection on 1-bar pivots
    # ------------------------------------------------------------
    def _last_two_pivot_highs(self):
        pivots = []
        for i in range(2, self.p.divergence_lookback):
            ph = float(self.data.high[-i])
            if (ph >= float(self.data.high[-(i - 1)])
                    and ph >= float(self.data.high[-(i + 1)])):
                pivots.append(i)
                if len(pivots) == 2:
                    break
        return pivots  # nearest first

    def _last_two_pivot_lows(self):
        pivots = []
        for i in range(2, self.p.divergence_lookback):
            pl = float(self.data.low[-i])
            if (pl <= float(self.data.low[-(i - 1)])
                    and pl <= float(self.data.low[-(i + 1)])):
                pivots.append(i)
                if len(pivots) == 2:
                    break
        return pivots

    def _bearish_divergence(self):
        # Price higher high while MACD makes a lower high
        pivots = self._last_two_pivot_highs()
        if len(pivots) < 2:
            return False
        i2, i1 = pivots  # i2 = most recent pivot
        return (
            float(self.data.high[-i2]) > float(self.data.high[-i1])
            and float(self.macd.macd[-i2]) < float(self.macd.macd[-i1])
        )

    def _bullish_divergence(self):
        # Price lower low while MACD makes a higher low
        pivots = self._last_two_pivot_lows()
        if len(pivots) < 2:
            return False
        i2, i1 = pivots
        return (
            float(self.data.low[-i2]) < float(self.data.low[-i1])
            and float(self.macd.macd[-i2]) > float(self.macd.macd[-i1])
        )

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
                    risk_cash=stop_distance * size * self._contract_size(),
                )
                if direction == 1:
                    self.stop_price = fill_price - stop_distance
                    self.target_price = fill_price + stop_distance * self.p.rr_ratio
                else:
                    self.stop_price = fill_price + stop_distance
                    self.target_price = fill_price - stop_distance * self.p.rr_ratio

                self.log(
                    f"{'LONG' if direction == 1 else 'SHORT'} OPEN | "
                    f"Entry:{fill_price:.5f} Stop:{self.stop_price:.5f} "
                    f"Target:{self.target_price:.5f} Size:{size:.4f}"
                )

            elif ctx and ctx["kind"] == "exit" and self.trade_meta is not None:
                meta = self.trade_meta
                pnl = (
                    (fill_price - meta["entry_price"])
                    * meta["direction"] * meta["size"] * self._contract_size()
                )
                risk = meta["risk_cash"]
                self.trade_records.append(dict(
                    setup="macd-cross",
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
        cross = int(self.crossover[0])
        histogram = float(self.macd.macd[0]) - float(self.macd.signal[0])

        # ------------------------------------------------------------
        # Manage the open position
        # ------------------------------------------------------------
        if self.position.size != 0 and self.trade_meta is not None:
            d = self.trade_meta["direction"]
            if d == 1:
                if l <= self.stop_price:
                    self.order_ctx = dict(kind="exit", reason="atr-stop")
                elif h >= self.target_price:
                    self.order_ctx = dict(kind="exit", reason="rr-target")
                elif self.p.exit_on_opposite_cross and cross < 0:
                    self.order_ctx = dict(kind="exit", reason="opposite-cross")
            else:
                if h >= self.stop_price:
                    self.order_ctx = dict(kind="exit", reason="atr-stop")
                elif l <= self.target_price:
                    self.order_ctx = dict(kind="exit", reason="rr-target")
                elif self.p.exit_on_opposite_cross and cross > 0:
                    self.order_ctx = dict(kind="exit", reason="opposite-cross")
            if self.order_ctx:
                self.order = self.close()
            return

        # ------------------------------------------------------------
        # Entries: signal-line cross + filters
        # ------------------------------------------------------------
        if cross == 0 or atr_value <= 0:
            return

        # Histogram-strength filter: skip weak, near-zero crosses
        if abs(histogram) < self.p.hist_min_atr * atr_value:
            self.log(
                f"SIGNAL BLOCKED | weak cross, |hist| {abs(histogram):.5f} "
                f"< {self.p.hist_min_atr} x ATR {atr_value:.5f}")
            return

        direction = 1 if cross > 0 else -1
        if direction == 1 and not self.p.allow_long:
            return
        if direction == -1 and not self.p.allow_short:
            return

        # Trend filter: longs above the long EMA, shorts below it
        if self.p.use_trend_filter:
            trend_ema = float(self.ema_trend[0])
            if direction == 1 and c <= trend_ema:
                self.log(f"SIGNAL BLOCKED | long below EMA{self.p.trend_period}")
                return
            if direction == -1 and c >= trend_ema:
                self.log(f"SIGNAL BLOCKED | short above EMA{self.p.trend_period}")
                return

        # Divergence: warn always, veto only when divergence_block is on
        if direction == 1 and self._bearish_divergence():
            self.log("DIVERGENCE WARNING | bearish (price HH, MACD LH) against this long")
            if self.p.divergence_block:
                return
        if direction == -1 and self._bullish_divergence():
            self.log("DIVERGENCE WARNING | bullish (price LL, MACD HL) against this short")
            if self.p.divergence_block:
                return

        stop_distance = self.p.stop_atr_mult * atr_value
        size = self._calc_size(stop_distance)
        if size <= 0:
            return

        self.order_ctx = dict(kind="entry", direction=direction, stop_distance=stop_distance)
        self.log(
            f"{'LONG' if direction == 1 else 'SHORT'} SIGNAL | "
            f"MACD cross | hist:{histogram:.5f} Size:{size:.4f}")
        self.order = self.buy(size=size) if direction == 1 else self.sell(size=size)

    # ====================================================================================
    # params_metadata is optional -- it is used to configure the "Run Backtest/Bot" modal.
    params_metadata = {
        "macd_fast": {
            "label": "MACD fast EMA",
            "helper_text": "Fast EMA period of the MACD line",
            "value_type": "int",
        },
        "macd_slow": {
            "label": "MACD slow EMA",
            "helper_text": "Slow EMA period of the MACD line",
            "value_type": "int",
        },
        "macd_signal": {
            "label": "Signal period",
            "helper_text": "EMA period of the signal line",
            "value_type": "int",
        },
        "hist_min_atr": {
            "label": "Histogram threshold (ATRs)",
            "helper_text": "Skip crosses whose histogram magnitude is below this x ATR",
            "value_type": "float",
        },
        "trend_period": {
            "label": "Trend EMA",
            "helper_text": "Longs only above this EMA, shorts only below",
            "value_type": "int",
        },
        "use_trend_filter": {
            "label": "Trend filter",
            "helper_text": "Enable the long-period EMA trend filter",
            "value_type": "bool",
        },
        "divergence_block": {
            "label": "Block on divergence",
            "helper_text": "Veto signals that fight a fresh price/MACD divergence "
                           "(warnings are always logged)",
            "value_type": "bool",
        },
        "stop_atr_mult": {
            "label": "Stop (ATRs)",
            "helper_text": "Stop loss distance from entry, in ATRs",
            "value_type": "float",
        },
        "rr_ratio": {
            "label": "Reward:risk",
            "helper_text": "Take profit at this multiple of the risk",
            "value_type": "float",
        },
        "exit_on_opposite_cross": {
            "label": "Exit on opposite cross",
            "helper_text": "Also close the position when MACD crosses back",
            "value_type": "bool",
        },
        "risk_per_trade_pct": {
            "label": "Risk per trade %",
            "helper_text": "Percent of account equity risked on each trade",
            "value_type": "float",
        },
    }
