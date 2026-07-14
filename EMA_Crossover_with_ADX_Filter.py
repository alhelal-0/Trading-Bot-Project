# Bots on TradeLocker are implemented in Python using the Backtrader framework.
# You can find more examples and the complete documentation at https://www.backtrader.com/

# EMA crossover trend following with ADX and volume filters:
#   - Signal: fast EMA (20) crossing slow EMA (50); long on cross up,
#     short on cross down.
#   - Filters: ADX must be above `adx_threshold` (a trend actually
#     exists) and volume above its `vol_period` average (participation
#     confirms the move). Disable the volume filter for feeds without
#     volume data.
#   - Risk `risk_per_trade_pct` of equity per trade against the initial
#     ATR stop; exit on an ATR chandelier trailing stop (only tightens)
#     or the opposite crossover, whichever comes first.

import math

import backtrader as bt


class EmaAdxTrendStrategy(bt.Strategy):
    params = {
        "fast_period": 20,
        "slow_period": 50,
        "adx_period": 14,
        "adx_threshold": 25.0,
        "vol_period": 20,
        "use_volume_filter": True,
        "atr_period": 14,
        "trail_atr_mult": 2.0,
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
        self.adx = bt.indicators.AverageDirectionalMovementIndex(
            self.data, period=self.p.adx_period)
        self.vol_sma = bt.indicators.SimpleMovingAverage(
            self.data.volume, period=self.p.vol_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)

        self.order = None
        self.order_ctx = None
        self.trade_meta = None
        self.trail_stop = None
        self.extreme = None

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
        needed = max(self.p.slow_period, self.p.adx_period * 2,
                     self.p.vol_period, self.p.atr_period) + 1
        return len(self.data) > needed

    def _filters_pass(self):
        if float(self.adx.adx[0]) <= self.p.adx_threshold:
            return False
        if self.p.use_volume_filter:
            vol = float(self.data.volume[0])
            avg = float(self.vol_sma[0])
            if avg > 0 and vol <= avg:
                return False
        return True

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
                self.extreme = fill_price
                self.trail_stop = (
                    fill_price - stop_distance if direction == 1
                    else fill_price + stop_distance
                )
                self.log(
                    f"{'LONG' if direction == 1 else 'SHORT'} OPEN | "
                    f"Entry:{fill_price:.5f} Trail:{self.trail_stop:.5f} Size:{size:.4f}"
                )

            elif ctx and ctx["kind"] == "exit" and self.trade_meta is not None:
                meta = self.trade_meta
                pnl = (
                    (fill_price - meta["entry_price"])
                    * meta["direction"] * meta["size"] * self._contract_size()
                )
                risk = meta["risk_cash"]
                self.trade_records.append(dict(
                    setup="ema-adx-cross",
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
                self.trail_stop = None
                self.extreme = None

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
        atr_value = float(self.atr[0])
        cross = int(self.crossover[0])

        # Manage the open position
        if self.position.size != 0 and self.trade_meta is not None:
            if self.position.size > 0:
                self.extreme = max(self.extreme, h)
                self.trail_stop = max(
                    self.trail_stop, self.extreme - self.p.trail_atr_mult * atr_value)

                if l <= self.trail_stop:
                    self.order_ctx = dict(kind="exit", reason="atr-trail")
                    self.order = self.close()
                elif cross < 0:
                    self.order_ctx = dict(kind="exit", reason="opposite-cross")
                    self.order = self.close()
            else:
                self.extreme = min(self.extreme, l)
                self.trail_stop = min(
                    self.trail_stop, self.extreme + self.p.trail_atr_mult * atr_value)

                if h >= self.trail_stop:
                    self.order_ctx = dict(kind="exit", reason="atr-trail")
                    self.order = self.close()
                elif cross > 0:
                    self.order_ctx = dict(kind="exit", reason="opposite-cross")
                    self.order = self.close()
            return

        # Entries: EMA crossover gated by ADX and volume
        if cross == 0 or atr_value <= 0:
            return
        if not self._filters_pass():
            self.log(
                f"SIGNAL BLOCKED | cross:{cross} "
                f"ADX:{float(self.adx.adx[0]):.1f} <= {self.p.adx_threshold}"
            )
            return

        direction = 1 if cross > 0 else -1
        if direction == 1 and not self.p.allow_long:
            return
        if direction == -1 and not self.p.allow_short:
            return

        stop_distance = self.p.trail_atr_mult * atr_value
        size = self._calc_size(stop_distance)
        if size <= 0:
            return

        self.order_ctx = dict(kind="entry", direction=direction, stop_distance=stop_distance)
        self.log(
            f"{'LONG' if direction == 1 else 'SHORT'} SIGNAL | "
            f"EMA{self.p.fast_period}x{self.p.slow_period} "
            f"ADX:{float(self.adx.adx[0]):.1f} Size:{size:.4f}"
        )
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
        "adx_threshold": {
            "label": "ADX threshold",
            "helper_text": "Only trade when ADX is above this level",
            "value_type": "float",
        },
        "trail_atr_mult": {
            "label": "Trail (ATRs)",
            "helper_text": "Trailing stop distance in ATRs",
            "value_type": "float",
        },
        "risk_per_trade_pct": {
            "label": "Risk per trade %",
            "helper_text": "Percent of account equity risked on each trade",
            "value_type": "float",
        },
        "use_volume_filter": {
            "label": "Volume filter",
            "helper_text": "Require volume above its average (disable if the feed has no volume)",
            "value_type": "bool",
        },
    }
