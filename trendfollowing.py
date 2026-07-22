# Bots on TradeLocker are implemented in Python using the Backtrader framework.
# You can find more examples and the complete documentation at https://www.backtrader.com/

# Trend following via EMA pullbacks (9/15):
#   - Trend regime: fast EMA above slow EMA AND the slow EMA rising
#     steeply enough (`slope_min`, slope normalized by ATR) = uptrend;
#     mirror for downtrend. No regime, no trades.
#   - Entry: price pulls back into the 9/15 EMA zone and prints a strong
#     signal candle (body >= `signal_body_min`, closing near its extreme)
#     in the trend direction -- the classic "buy the pullback on a trend
#     day" entry, repeatable on every pullback of the trend.
#   - Chop guard: no entries while the EMAs are intertwined (closer than
#     `chop_ema_dist_atr` x ATR).
#   - Stop: beyond the signal bar / recent swing. Winners are not capped:
#     after `trail_trigger_r` (1R) of profit the stop trails the best
#     price by `trail_atr_mult` x ATR and only tightens. With trailing
#     disabled, a fixed `rr_ratio` take-profit applies instead.
#   - Position size risks `risk_per_trade_pct` (1%) of account equity.

import math

import backtrader as bt


class TrendFollowingStrategy(bt.Strategy):
    params = {
        "fast_period": 9,
        "slow_period": 15,
        "slope_lookback": 6,
        "slope_min": 0.08,           # slow-EMA slope per bar, normalized by ATR
        "ema_touch_atr": 0.25,       # pullback must come this close to the EMA zone
        "signal_body_min": 0.50,     # signal candle body vs range
        "signal_close_pct": 0.30,    # close within this fraction of its extreme
        "chop_ema_dist_atr": 0.15,   # EMAs closer than this = chop, stand aside
        "swing_lookback": 3,
        "stop_pad_ticks": 2,
        "atr_period": 14,
        "use_trailing": True,        # trail after trigger R, winners uncapped
        "trail_trigger_r": 1.0,
        "trail_atr_mult": 1.5,
        "rr_ratio": 2.0,             # fixed target used only when trailing is off
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
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)

        self.order = None
        self.order_ctx = None
        self.trade_meta = None
        self.stop_price = None
        self.target_price = None
        self.extreme = None
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
        needed = max(self.p.slow_period, self.p.atr_period,
                     self.p.slope_lookback, self.p.swing_lookback) + 2
        return len(self.data) > needed

    def _slope_per_atr(self, atr_value):
        lb = int(self.p.slope_lookback)
        if atr_value <= 0:
            return 0.0
        return (float(self.ema_slow[0]) - float(self.ema_slow[-lb])) / (atr_value * lb)

    def _swing_low(self):
        return min(float(self.data.low[-i]) for i in range(0, self.p.swing_lookback + 1))

    def _swing_high(self):
        return max(float(self.data.high[-i]) for i in range(0, self.p.swing_lookback + 1))

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

                if self.p.use_trailing:
                    self.target_price = None  # winners uncapped, trail is the exit
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
                    setup="ema-pullback-trend",
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

        o = float(self.data.open[0])
        h = float(self.data.high[0])
        l = float(self.data.low[0])
        c = float(self.data.close[0])
        atr_value = float(self.atr[0])

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
                else:
                    self.extreme = min(self.extreme, l)
                    favorable = meta["entry_price"] - self.extreme

                if (
                    not self.trail_active
                    and favorable >= meta["stop_distance"] * self.p.trail_trigger_r
                ):
                    self.trail_active = True
                    self.log("TRAIL ACTIVE | trigger R reached, stop now trails")

                if self.trail_active:
                    if d == 1:
                        self.stop_price = max(
                            self.stop_price,
                            self.extreme - self.p.trail_atr_mult * atr_value)
                    else:
                        self.stop_price = min(
                            self.stop_price,
                            self.extreme + self.p.trail_atr_mult * atr_value)

            if d == 1:
                if l <= self.stop_price:
                    reason = "trail-stop" if self.trail_active else "swing-stop"
                    self.order_ctx = dict(kind="exit", reason=reason)
                elif self.target_price is not None and h >= self.target_price:
                    self.order_ctx = dict(kind="exit", reason="rr-target")
            else:
                if h >= self.stop_price:
                    reason = "trail-stop" if self.trail_active else "swing-stop"
                    self.order_ctx = dict(kind="exit", reason=reason)
                elif self.target_price is not None and l <= self.target_price:
                    self.order_ctx = dict(kind="exit", reason="rr-target")
            if self.order_ctx:
                self.order = self.close()
            return

        # --------------------------------------------------------
        # Entries: pullback to the EMA zone inside an established trend
        # --------------------------------------------------------
        if atr_value <= 0:
            return

        ema_fast = float(self.ema_fast[0])
        ema_slow = float(self.ema_slow[0])

        # Chop guard: EMAs intertwined = no trend worth following
        if abs(ema_fast - ema_slow) < self.p.chop_ema_dist_atr * atr_value:
            return

        slope = self._slope_per_atr(atr_value)
        touch = self.p.ema_touch_atr * atr_value

        rng = h - l
        body_ratio = abs(c - o) / rng if rng > 0 else 0.0
        closes_high = rng > 0 and (h - c) <= self.p.signal_close_pct * rng
        closes_low = rng > 0 and (c - l) <= self.p.signal_close_pct * rng

        uptrend = ema_fast > ema_slow and slope >= self.p.slope_min
        downtrend = ema_fast < ema_slow and slope <= -self.p.slope_min

        pad = self._tick_size() * self.p.stop_pad_ticks

        if self.p.allow_long and uptrend:
            pulled = min(l, float(self.data.low[-1])) <= max(ema_fast, ema_slow) + touch
            signal = c > o and body_ratio >= self.p.signal_body_min and closes_high
            if pulled and signal:
                stop_price = min(l, self._swing_low()) - pad
                stop_distance = c - stop_price
                size = self._calc_size(stop_distance) if stop_distance > 0 else 0.0
                if size > 0:
                    self.order_ctx = dict(kind="entry", direction=1,
                                          stop_distance=stop_distance)
                    self.log(
                        f"LONG PULLBACK | Slope:{slope:.3f} C:{c:.5f} "
                        f"Stop:{stop_price:.5f} Size:{size:.4f}")
                    self.order = self.buy(size=size)
                    return

        if self.p.allow_short and downtrend:
            pulled = max(h, float(self.data.high[-1])) >= min(ema_fast, ema_slow) - touch
            signal = c < o and body_ratio >= self.p.signal_body_min and closes_low
            if pulled and signal:
                stop_price = max(h, self._swing_high()) + pad
                stop_distance = stop_price - c
                size = self._calc_size(stop_distance) if stop_distance > 0 else 0.0
                if size > 0:
                    self.order_ctx = dict(kind="entry", direction=-1,
                                          stop_distance=stop_distance)
                    self.log(
                        f"SHORT PULLBACK | Slope:{slope:.3f} C:{c:.5f} "
                        f"Stop:{stop_price:.5f} Size:{size:.4f}")
                    self.order = self.sell(size=size)

    # ====================================================================================
    # params_metadata is optional -- it is used to configure the "Run Backtest/Bot" modal.
    params_metadata = {
        "fast_period": {
            "label": "Fast EMA",
            "helper_text": "Period of the fast EMA (9 in the base style)",
            "value_type": "int",
        },
        "slow_period": {
            "label": "Slow EMA",
            "helper_text": "Period of the slow EMA (15 in the base style)",
            "value_type": "int",
        },
        "slope_min": {
            "label": "Min trend slope",
            "helper_text": "Slow-EMA slope per bar normalized by ATR; higher = only "
                           "steeper trends qualify",
            "value_type": "float",
        },
        "ema_touch_atr": {
            "label": "Pullback touch (ATRs)",
            "helper_text": "How close the pullback must come to the EMA zone",
            "value_type": "float",
        },
        "signal_body_min": {
            "label": "Signal candle body",
            "helper_text": "Minimum body/range ratio of the entry candle",
            "value_type": "float",
        },
        "use_trailing": {
            "label": "Trailing stop",
            "helper_text": "Trail after the trigger profit and leave winners uncapped "
                           "(recommended for trend following)",
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
        "rr_ratio": {
            "label": "Reward:risk",
            "helper_text": "Fixed take-profit multiple, used only when trailing is off",
            "value_type": "float",
        },
        "risk_per_trade_pct": {
            "label": "Risk per trade %",
            "helper_text": "Percent of account equity risked on each trade",
            "value_type": "float",
        },
    }
