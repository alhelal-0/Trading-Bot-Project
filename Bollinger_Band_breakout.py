# Bots on TradeLocker are implemented in Python using the Backtrader framework.
# You can find more examples and the complete documentation at https://www.backtrader.com/

# Bollinger Band breakout:
#   - Bands: `bb_period` (20) moving average +/- `bb_dev` (2.0) standard
#     deviations.
#   - LONG when price has CLOSED above the upper band for
#     `confirm_closes` consecutive bars (false-breakout filter) and the
#     confirming candle's volume exceeds `vol_mult` x its rolling
#     average. SHORT is the mirror below the lower band.
#   - Squeeze detector: when band width normalized by price drops below
#     `squeeze_pct`, the next breakout within `squeeze_lookback` bars is
#     tagged as a post-squeeze breakout (higher quality); set
#     `require_squeeze` to trade ONLY those.
#   - Stop: the middle band (mean) at entry, or `stop_atr_mult` x ATR
#     when `stop_at_mean` is off.
#   - Take profit: fixed `rr_ratio` R:R, or -- with `use_reentry_exit` --
#     a trailing exit that closes when price re-enters the bands.
#   - Position size risks `risk_per_trade_pct` (1%) of account equity.

import math

import backtrader as bt


class BollingerBreakoutStrategy(bt.Strategy):
    params = {
        "bb_period": 20,
        "bb_dev": 2.0,
        "vol_period": 20,
        "vol_mult": 1.2,             # volume must exceed this x rolling average
        "use_volume_filter": True,
        "confirm_closes": 2,         # consecutive closes outside the band
        "squeeze_pct": 0.30,         # band width as % of price counting as a squeeze
        "squeeze_lookback": 10,      # breakout within N bars of a squeeze = post-squeeze
        "require_squeeze": False,    # True: only trade post-squeeze breakouts
        "stop_at_mean": True,        # stop at the middle band; False = ATR stop
        "atr_period": 14,
        "stop_atr_mult": 2.0,
        "use_reentry_exit": False,   # True: exit when price closes back inside the bands
        "rr_ratio": 2.0,             # fixed R:R target when reentry exit is off
        "risk_per_trade_pct": 1.0,
        "allow_long": True,
        "allow_short": True,
        "min_size": 0.01,
        "verbose": False,
    }

    def __init__(self) -> None:
        self.bb = bt.indicators.BollingerBands(
            self.data.close, period=self.p.bb_period, devfactor=self.p.bb_dev)
        self.vol_sma = bt.indicators.SimpleMovingAverage(
            self.data.volume, period=self.p.vol_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)

        self.order = None
        self.order_ctx = None
        self.trade_meta = None
        self.stop_price = None
        self.target_price = None     # None in reentry-exit mode

        # Breakout / squeeze state
        self.count_above = 0         # consecutive closes above the upper band
        self.count_below = 0
        self.bars_since_squeeze = None

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
        needed = max(self.p.bb_period, self.p.vol_period, self.p.atr_period) + 1
        return len(self.data) > needed

    def _volume_ok(self):
        if not self.p.use_volume_filter:
            return True
        avg = float(self.vol_sma[0])
        if avg <= 0:
            return True  # feed has no volume data
        return float(self.data.volume[0]) > avg * self.p.vol_mult

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
                    setup=ctx["setup"],
                )
                self.stop_price = stop_price
                if self.p.use_reentry_exit:
                    self.target_price = None
                else:
                    self.target_price = (
                        fill_price + stop_distance * self.p.rr_ratio
                        if direction == 1
                        else fill_price - stop_distance * self.p.rr_ratio
                    )

                # The excursion is consumed: a fresh breakout is needed next time
                self.count_above = 0
                self.count_below = 0

                self.log(
                    f"{'LONG' if direction == 1 else 'SHORT'} OPEN | {ctx['setup']} | "
                    f"Entry:{fill_price:.5f} Stop:{self.stop_price:.5f} "
                    f"Target:{'band-reentry' if self.target_price is None else f'{self.target_price:.5f}'} "
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
                    setup=meta["setup"],
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

        c = float(self.data.close[0])
        h = float(self.data.high[0])
        l = float(self.data.low[0])
        top = float(self.bb.top[0])
        bot = float(self.bb.bot[0])
        mid = float(self.bb.mid[0])

        # --------------------------------------------------------
        # Track squeeze and breakout persistence on every bar
        # --------------------------------------------------------
        width_pct = 100.0 * (top - bot) / c if c > 0 else 0.0
        if width_pct < self.p.squeeze_pct:
            if self.bars_since_squeeze is None or self.bars_since_squeeze > 0:
                self.log(f"SQUEEZE | band width {width_pct:.3f}% < {self.p.squeeze_pct}%")
            self.bars_since_squeeze = 0
        elif self.bars_since_squeeze is not None:
            self.bars_since_squeeze += 1

        post_squeeze = (
            self.bars_since_squeeze is not None
            and self.bars_since_squeeze <= self.p.squeeze_lookback
        )

        if c > top:
            self.count_above += 1
            self.count_below = 0
        elif c < bot:
            self.count_below += 1
            self.count_above = 0
        else:
            self.count_above = 0
            self.count_below = 0

        if self.order is not None:
            return

        # --------------------------------------------------------
        # Manage the open position
        # --------------------------------------------------------
        if self.position.size != 0 and self.trade_meta is not None:
            d = self.trade_meta["direction"]
            if d == 1:
                if l <= self.stop_price:
                    self.order_ctx = dict(kind="exit", reason="stop")
                elif self.target_price is not None and h >= self.target_price:
                    self.order_ctx = dict(kind="exit", reason="rr-target")
                elif self.target_price is None and c < top:
                    self.order_ctx = dict(kind="exit", reason="band-reentry")
            else:
                if h >= self.stop_price:
                    self.order_ctx = dict(kind="exit", reason="stop")
                elif self.target_price is not None and l <= self.target_price:
                    self.order_ctx = dict(kind="exit", reason="rr-target")
                elif self.target_price is None and c > bot:
                    self.order_ctx = dict(kind="exit", reason="band-reentry")
            if self.order_ctx:
                self.order = self.close()
            return

        # --------------------------------------------------------
        # Entries: confirmed breakout + volume (+ optional squeeze gate)
        # --------------------------------------------------------
        atr_value = float(self.atr[0])
        if atr_value <= 0:
            return
        if self.p.require_squeeze and not post_squeeze:
            return

        setup = "bb-breakout-squeeze" if post_squeeze else "bb-breakout"

        if (
            self.p.allow_long
            and self.count_above >= self.p.confirm_closes
            and self._volume_ok()
        ):
            stop_price = mid if self.p.stop_at_mean else c - self.p.stop_atr_mult * atr_value
            stop_distance = c - stop_price
            if stop_distance <= 0:
                return
            size = self._calc_size(stop_distance)
            if size <= 0:
                return
            self.order_ctx = dict(kind="entry", direction=1, stop_price=stop_price, setup=setup)
            self.log(
                f"LONG BREAKOUT | {setup} | {self.count_above} closes above "
                f"{top:.5f} | Size:{size:.4f}")
            self.order = self.buy(size=size)

        elif (
            self.p.allow_short
            and self.count_below >= self.p.confirm_closes
            and self._volume_ok()
        ):
            stop_price = mid if self.p.stop_at_mean else c + self.p.stop_atr_mult * atr_value
            stop_distance = stop_price - c
            if stop_distance <= 0:
                return
            size = self._calc_size(stop_distance)
            if size <= 0:
                return
            self.order_ctx = dict(kind="entry", direction=-1, stop_price=stop_price, setup=setup)
            self.log(
                f"SHORT BREAKOUT | {setup} | {self.count_below} closes below "
                f"{bot:.5f} | Size:{size:.4f}")
            self.order = self.sell(size=size)

    # ====================================================================================
    # params_metadata is optional -- it is used to configure the "Run Backtest/Bot" modal.
    params_metadata = {
        "bb_period": {
            "label": "Bollinger period",
            "helper_text": "Period of the moving average and bands",
            "value_type": "int",
        },
        "bb_dev": {
            "label": "Bollinger std dev",
            "helper_text": "Standard deviation multiplier of the bands",
            "value_type": "float",
        },
        "vol_mult": {
            "label": "Volume multiplier",
            "helper_text": "Breakout volume must exceed this x its rolling average",
            "value_type": "float",
        },
        "confirm_closes": {
            "label": "Confirming closes",
            "helper_text": "Consecutive closes outside the band before entering "
                           "(false-breakout filter)",
            "value_type": "int",
        },
        "squeeze_pct": {
            "label": "Squeeze width %",
            "helper_text": "Band width below this % of price counts as a squeeze",
            "value_type": "float",
        },
        "squeeze_lookback": {
            "label": "Squeeze lookback",
            "helper_text": "Breakouts within N bars of a squeeze are tagged post-squeeze",
            "value_type": "int",
        },
        "require_squeeze": {
            "label": "Post-squeeze only",
            "helper_text": "Trade only breakouts that follow a band squeeze",
            "value_type": "bool",
        },
        "stop_at_mean": {
            "label": "Stop at mean",
            "helper_text": "Stop at the middle band; off = ATR-based stop",
            "value_type": "bool",
        },
        "stop_atr_mult": {
            "label": "Stop (ATRs)",
            "helper_text": "ATR stop distance used when 'Stop at mean' is off",
            "value_type": "float",
        },
        "use_reentry_exit": {
            "label": "Exit on band re-entry",
            "helper_text": "Trail out when price closes back inside the bands; "
                           "off = fixed R:R target",
            "value_type": "bool",
        },
        "rr_ratio": {
            "label": "Reward:risk",
            "helper_text": "Fixed R:R target used when the re-entry exit is off",
            "value_type": "float",
        },
        "risk_per_trade_pct": {
            "label": "Risk per trade %",
            "helper_text": "Percent of account equity risked on each trade",
            "value_type": "float",
        },
    }
