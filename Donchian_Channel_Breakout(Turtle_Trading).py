# Bots on TradeLocker are implemented in Python using the Backtrader framework.
# You can find more examples and the complete documentation at https://www.backtrader.com/

# Donchian Channel Breakout (Turtle Trading, System 1):
#   - Enter LONG when price closes above the highest high of the previous
#     `entry_lookback` bars; SHORT when it closes below the lowest low.
#   - Position size risks `risk_per_trade_pct` of account equity, where the
#     risk distance is `atr_stop_mult` x ATR (the Turtles' "N").
#   - Exit on the FIRST of: the opposite `exit_lookback`-bar Donchian
#     breakout, or a chandelier trailing stop that only tightens.

import math

import backtrader as bt


class DonchianTurtleStrategy(bt.Strategy):
    params = {
        "entry_lookback": 20,
        "exit_lookback": 10,
        "atr_period": 20,
        "atr_stop_mult": 2.0,
        "risk_per_trade_pct": 1.0,
        "allow_long": True,
        "allow_short": True,
        "min_size": 0.01,
        "verbose": False,
    }

    def __init__(self) -> None:
        self.entry_high = bt.indicators.Highest(self.data.high, period=self.p.entry_lookback)
        self.entry_low = bt.indicators.Lowest(self.data.low, period=self.p.entry_lookback)
        self.exit_high = bt.indicators.Highest(self.data.high, period=self.p.exit_lookback)
        self.exit_low = bt.indicators.Lowest(self.data.low, period=self.p.exit_lookback)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)

        self.order = None
        self.order_ctx = None        # what the in-flight order is doing
        self.trade_meta = None       # open-trade context
        self.trail_stop = None
        self.extreme = None          # highest high (long) / lowest low (short) since entry

        self.trade_records = []      # one dict per closed trade

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
        needed = max(self.p.entry_lookback, self.p.exit_lookback, self.p.atr_period) + 1
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
                    setup="donchian-breakout",
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

        c = float(self.data.close[0])
        h = float(self.data.high[0])
        l = float(self.data.low[0])
        atr_value = float(self.atr[0])

        # Manage the open position
        if self.position.size != 0 and self.trade_meta is not None:
            if self.position.size > 0:
                self.extreme = max(self.extreme, h)
                self.trail_stop = max(
                    self.trail_stop, self.extreme - self.p.atr_stop_mult * atr_value)

                if l <= self.trail_stop:
                    self.order_ctx = dict(kind="exit", reason="atr-trail")
                    self.order = self.close()
                elif c < float(self.exit_low[-1]):
                    self.order_ctx = dict(kind="exit", reason="donchian-exit")
                    self.order = self.close()
            else:
                self.extreme = min(self.extreme, l)
                self.trail_stop = min(
                    self.trail_stop, self.extreme + self.p.atr_stop_mult * atr_value)

                if h >= self.trail_stop:
                    self.order_ctx = dict(kind="exit", reason="atr-trail")
                    self.order = self.close()
                elif c > float(self.exit_high[-1]):
                    self.order_ctx = dict(kind="exit", reason="donchian-exit")
                    self.order = self.close()
            return

        # Entries: breakout of the previous bars' channel
        if atr_value <= 0:
            return

        stop_distance = self.p.atr_stop_mult * atr_value

        if self.p.allow_long and c > float(self.entry_high[-1]):
            size = self._calc_size(stop_distance)
            if size <= 0:
                return
            self.order_ctx = dict(kind="entry", direction=1, stop_distance=stop_distance)
            self.log(f"LONG BREAKOUT | C:{c:.5f} Size:{size:.4f}")
            self.order = self.buy(size=size)

        elif self.p.allow_short and c < float(self.entry_low[-1]):
            size = self._calc_size(stop_distance)
            if size <= 0:
                return
            self.order_ctx = dict(kind="entry", direction=-1, stop_distance=stop_distance)
            self.log(f"SHORT BREAKOUT | C:{c:.5f} Size:{size:.4f}")
            self.order = self.sell(size=size)

    # ====================================================================================
    # params_metadata is optional -- it is used to configure the "Run Backtest/Bot" modal.
    params_metadata = {
        "entry_lookback": {
            "label": "Entry lookback",
            "helper_text": "Enter on a breakout of the previous N bars' high/low",
            "value_type": "int",
        },
        "exit_lookback": {
            "label": "Exit lookback",
            "helper_text": "Exit on a breakout of the opposite N-bar channel",
            "value_type": "int",
        },
        "atr_period": {
            "label": "ATR period",
            "helper_text": "Period of the ATR used for stops and sizing (Turtle N)",
            "value_type": "int",
        },
        "atr_stop_mult": {
            "label": "ATR stop multiple",
            "helper_text": "Trailing stop distance in ATRs",
            "value_type": "float",
        },
        "risk_per_trade_pct": {
            "label": "Risk per trade %",
            "helper_text": "Percent of account equity risked on each trade",
            "value_type": "float",
        },
    }
