# Bots on TradeLocker are implemented in Python using the Backtrader framework.
# You can find more examples and the complete documentation at https://www.backtrader.com/

# RSI mean reversion:
#   - LONG when RSI has been below the oversold threshold and then CLOSES
#     back above it (the reversal is confirmed -- never catch the falling
#     knife). SHORT is the mirror: RSI back below the overbought level.
#   - Extremity requirement: during the oversold episode RSI must have
#     reached `rsi_extreme_long` (28) -- a dip to 29.9 does not count.
#   - Range filter: signals only when the market is NOT trending strongly
#     (ADX below `adx_max`, default 40 = block only strong trends),
#     because mean reversion fails in strong trends.
#   - Defaults are calibrated for roughly 2-3 signals per day on 5m NAS100
#     style data; tighten `rsi_extreme_*` / lower `adx_max` for fewer,
#     higher-quality signals.
#   - Stop beyond the recent swing low/high.
#   - Take profit either when RSI reaches the midline (50), or at a fixed
#     `rr_ratio` multiple of the risk (set `use_midline_exit` to choose).
#   - Position size risks `risk_per_trade_pct` (1%) of account equity.

import math

import backtrader as bt


class RsiMeanReversionStrategy(bt.Strategy):
    params = {
        "rsi_period": 14,
        "oversold": 30.0,
        "overbought": 70.0,
        "rsi_extreme_long": 28.0,    # episode must reach this before a LONG
        "rsi_extreme_short": 72.0,   # episode must reach this before a SHORT
        "adx_period": 14,
        "adx_max": 40.0,             # block only strongly trending markets
        "use_adx_filter": True,
        "swing_lookback": 10,        # bars for the swing high/low stop
        "stop_pad_ticks": 2,
        "use_midline_exit": True,    # True: exit at RSI midline; False: fixed R:R
        "midline": 50.0,
        "rr_ratio": 2.0,             # used when use_midline_exit is False
        "risk_per_trade_pct": 1.0,
        "allow_long": True,
        "allow_short": True,
        "min_size": 0.01,
        "verbose": False,
    }

    def __init__(self) -> None:
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(
            self.data, period=self.p.adx_period)

        self.order = None
        self.order_ctx = None
        self.trade_meta = None
        self.stop_price = None
        self.target_price = None     # None in midline mode

        # Oversold / overbought episode tracking
        self.os_active = False       # currently (or recently) below oversold
        self.os_min = 100.0          # lowest RSI seen during the episode
        self.ob_active = False
        self.ob_max = 0.0

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
        needed = max(self.p.rsi_period, self.p.adx_period * 2,
                     self.p.swing_lookback) + 1
        return len(self.data) > needed

    def _range_filter_ok(self):
        if not self.p.use_adx_filter:
            return True
        return float(self.adx.adx[0]) < self.p.adx_max

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
                stop_price = ctx["stop_price"]
                size = abs(float(order.executed.size))
                stop_distance = abs(fill_price - stop_price)

                self.trade_meta = dict(
                    direction=direction,
                    entry_dt=self.data.datetime.datetime(0),
                    entry_price=fill_price,
                    size=size,
                    risk_cash=stop_distance * size * self._contract_size(),
                )
                self.stop_price = stop_price
                if self.p.use_midline_exit:
                    self.target_price = None  # exit on the RSI midline instead
                else:
                    self.target_price = (
                        fill_price + stop_distance * self.p.rr_ratio
                        if direction == 1
                        else fill_price - stop_distance * self.p.rr_ratio
                    )

                self.log(
                    f"{'LONG' if direction == 1 else 'SHORT'} OPEN | "
                    f"Entry:{fill_price:.5f} Stop:{self.stop_price:.5f} "
                    f"Target:{'RSI-midline' if self.target_price is None else f'{self.target_price:.5f}'} "
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
                    setup="rsi-reversion",
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

        rsi_now = float(self.rsi[0])

        # --------------------------------------------------------
        # Track oversold / overbought episodes on every bar.
        # An episode "confirms" on the bar RSI closes back through the
        # threshold; it is consumed whether or not a trade is taken.
        # --------------------------------------------------------
        long_confirm = False
        short_confirm = False

        if rsi_now < self.p.oversold:
            self.os_active = True
            self.os_min = min(self.os_min, rsi_now)
        elif self.os_active:
            long_confirm = self.os_min <= self.p.rsi_extreme_long
            if not long_confirm:
                self.log(
                    f"LONG SKIPPED | episode min RSI {self.os_min:.1f} never "
                    f"reached extremity {self.p.rsi_extreme_long}")
            self.os_active = False
            self.os_min = 100.0

        if rsi_now > self.p.overbought:
            self.ob_active = True
            self.ob_max = max(self.ob_max, rsi_now)
        elif self.ob_active:
            short_confirm = self.ob_max >= self.p.rsi_extreme_short
            if not short_confirm:
                self.log(
                    f"SHORT SKIPPED | episode max RSI {self.ob_max:.1f} never "
                    f"reached extremity {self.p.rsi_extreme_short}")
            self.ob_active = False
            self.ob_max = 0.0

        if self.order is not None:
            return

        h = float(self.data.high[0])
        l = float(self.data.low[0])
        c = float(self.data.close[0])

        # --------------------------------------------------------
        # Manage the open position
        # --------------------------------------------------------
        if self.position.size != 0 and self.trade_meta is not None:
            d = self.trade_meta["direction"]
            if d == 1:
                if l <= self.stop_price:
                    self.order_ctx = dict(kind="exit", reason="swing-stop")
                elif self.target_price is not None and h >= self.target_price:
                    self.order_ctx = dict(kind="exit", reason="rr-target")
                elif self.target_price is None and rsi_now >= self.p.midline:
                    self.order_ctx = dict(kind="exit", reason="rsi-midline")
            else:
                if h >= self.stop_price:
                    self.order_ctx = dict(kind="exit", reason="swing-stop")
                elif self.target_price is not None and l <= self.target_price:
                    self.order_ctx = dict(kind="exit", reason="rr-target")
                elif self.target_price is None and rsi_now <= self.p.midline:
                    self.order_ctx = dict(kind="exit", reason="rsi-midline")
            if self.order_ctx:
                self.order = self.close()
            return

        # --------------------------------------------------------
        # Entries: confirmed RSI reversal + range filter
        # --------------------------------------------------------
        if not (long_confirm or short_confirm):
            return
        if not self._range_filter_ok():
            self.log(
                f"SIGNAL BLOCKED | trending market, "
                f"ADX:{float(self.adx.adx[0]):.1f} >= {self.p.adx_max}")
            return

        pad = self._tick_size() * self.p.stop_pad_ticks

        if long_confirm and self.p.allow_long:
            stop_price = self._swing_low() - pad
            stop_distance = c - stop_price
            if stop_distance <= 0:
                return
            size = self._calc_size(stop_distance)
            if size <= 0:
                return
            self.order_ctx = dict(kind="entry", direction=1, stop_price=stop_price)
            self.log(
                f"LONG SIGNAL | RSI back above {self.p.oversold} "
                f"(episode min {self.p.rsi_extreme_long} met) | "
                f"C:{c:.5f} Stop:{stop_price:.5f} Size:{size:.4f}")
            self.order = self.buy(size=size)

        elif short_confirm and self.p.allow_short:
            stop_price = self._swing_high() + pad
            stop_distance = stop_price - c
            if stop_distance <= 0:
                return
            size = self._calc_size(stop_distance)
            if size <= 0:
                return
            self.order_ctx = dict(kind="entry", direction=-1, stop_price=stop_price)
            self.log(
                f"SHORT SIGNAL | RSI back below {self.p.overbought} "
                f"(episode max {self.p.rsi_extreme_short} met) | "
                f"C:{c:.5f} Stop:{stop_price:.5f} Size:{size:.4f}")
            self.order = self.sell(size=size)

    # ====================================================================================
    # params_metadata is optional -- it is used to configure the "Run Backtest/Bot" modal.
    params_metadata = {
        "rsi_period": {
            "label": "RSI period",
            "helper_text": "Period of the RSI",
            "value_type": "int",
        },
        "oversold": {
            "label": "Oversold level",
            "helper_text": "LONG confirms when RSI closes back above this",
            "value_type": "float",
        },
        "overbought": {
            "label": "Overbought level",
            "helper_text": "SHORT confirms when RSI closes back below this",
            "value_type": "float",
        },
        "rsi_extreme_long": {
            "label": "Long extremity",
            "helper_text": "RSI must have reached this low during the episode",
            "value_type": "float",
        },
        "rsi_extreme_short": {
            "label": "Short extremity",
            "helper_text": "RSI must have reached this high during the episode",
            "value_type": "float",
        },
        "adx_max": {
            "label": "Max ADX",
            "helper_text": "Trade only when ADX is below this (not trending)",
            "value_type": "float",
        },
        "use_adx_filter": {
            "label": "ADX range filter",
            "helper_text": "Block signals in strongly trending markets",
            "value_type": "bool",
        },
        "swing_lookback": {
            "label": "Swing lookback",
            "helper_text": "Bars used to find the swing high/low for the stop",
            "value_type": "int",
        },
        "use_midline_exit": {
            "label": "Exit at RSI midline",
            "helper_text": "Take profit when RSI reaches 50; off = fixed R:R target",
            "value_type": "bool",
        },
        "rr_ratio": {
            "label": "Reward:risk",
            "helper_text": "Fixed R:R target used when the midline exit is off",
            "value_type": "float",
        },
        "risk_per_trade_pct": {
            "label": "Risk per trade %",
            "helper_text": "Percent of account equity risked on each trade",
            "value_type": "float",
        },
    }
