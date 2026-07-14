# Bots on TradeLocker are implemented in Python using the Backtrader framework.
# You can find more examples and the complete documentation at https://www.backtrader.com/

# Intraday VWAP mean reversion:
#   - Session VWAP and volume-weighted standard deviation bands, both
#     resetting at the start of every trading day (never rolling across
#     days).
#   - LONG when price has stretched below VWAP by `band_mult_1` standard
#     deviations and prints a bullish reversal candle closing back toward
#     VWAP, on above-average volume. SHORT is the mirror above VWAP.
#   - Optional tier-2 add-on if price stretches further to `band_mult_2`
#     deviations with another confirmation.
#   - Stop beyond the deviation bands (`stop_band_mult`), frozen at entry.
#   - Take profit at VWAP (full reversion). Optionally take off half at
#     VWAP and trail the rest with an ATR stop (`partial_exit_at_vwap`).
#   - Time-of-day filters: no signals in the first `skip_open_minutes`
#     after the open (VWAP unreliable), no new entries after
#     `last_entry_min`, and everything is flat by `flat_min` -- no
#     overnight or gap risk.
#   - Session times are minutes from midnight in the FEED's timezone.

import math

import backtrader as bt


class VwapStdBands(bt.Indicator):
    """Session VWAP plus the volume-weighted standard deviation of the
    typical price around it. Resets at the start of each trading day.
    Falls back to equal weighting (TWAP) when the feed has no volume."""

    lines = ("vwap", "dev")
    plotinfo = dict(subplot=False)

    def __init__(self):
        self._cur_date = None
        self._cum_pv = 0.0
        self._cum_pv2 = 0.0
        self._cum_vol = 0.0

    def next(self):
        d = self.data.datetime.date(0)
        if d != self._cur_date:
            self._cur_date = d
            self._cum_pv = 0.0
            self._cum_pv2 = 0.0
            self._cum_vol = 0.0

        typical = (self.data.high[0] + self.data.low[0] + self.data.close[0]) / 3.0
        vol = float(self.data.volume[0])
        if vol <= 0:
            vol = 1.0

        self._cum_pv += typical * vol
        self._cum_pv2 += typical * typical * vol
        self._cum_vol += vol

        vwap = self._cum_pv / self._cum_vol
        variance = max(self._cum_pv2 / self._cum_vol - vwap * vwap, 0.0)
        self.lines.vwap[0] = vwap
        self.lines.dev[0] = math.sqrt(variance)


class VwapReversionIntraday(bt.Strategy):
    params = {
        # Deviation bands
        "band_mult_1": 1.5,          # first entry tier (std devs from VWAP)
        "band_mult_2": 2.5,          # optional add-on tier
        "enable_tier2": True,
        "stop_band_mult": 3.0,       # stop beyond this band, frozen at entry
        # Confirmation
        "confirm_body_min": 0.50,    # reversal candle body vs range
        "vol_period": 20,            # rolling average volume lookback
        "use_volume_filter": True,
        # Exits
        "partial_exit_at_vwap": False,  # True: half off at VWAP, trail the rest
        "atr_period": 14,
        "trail_atr_mult": 1.5,       # trail distance after the partial exit
        # Time-of-day filters (minutes from midnight, feed timezone)
        "session_open_min": 9 * 60 + 30,
        "skip_open_minutes": 30,     # no signals right after the open
        "last_entry_min": 15 * 60,   # no new entries after this
        "flat_min": 15 * 60 + 55,    # flat for the last minutes of the session
        # Risk
        "risk_per_trade_pct": 1.0,
        "allow_long": True,
        "allow_short": True,
        "min_size": 0.01,
        "verbose": False,
    }

    def __init__(self) -> None:
        self.bands = VwapStdBands(self.data)
        self.vol_sma = bt.indicators.SimpleMovingAverage(
            self.data.volume, period=self.p.vol_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)

        self.order = None
        self.order_ctx = None

        # Aggregate open-position state (tier 1 + optional tier 2)
        self.pos_meta = None         # direction, avg_entry, total_size, stop, risk, tiers
        self.phase = "normal"        # "normal" | "trail" (after the partial exit)
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

    def _round_down_to_step(self, value, step):
        if step <= 0:
            return float(value)
        return math.floor(float(value) / step) * step

    def _calc_size(self, stop_distance):
        equity = float(self.broker.getvalue())
        risk_cash = equity * (self.p.risk_per_trade_pct / 100.0)
        unit_risk = max(stop_distance, self._tick_size()) * self._contract_size()
        if unit_risk <= 0:
            return 0.0
        size = self._round_down_to_step(risk_cash / unit_risk, self._lot_step())
        return size if size >= self.p.min_size else 0.0

    def _minutes_now(self):
        t = self.data.datetime.time(0)
        return t.hour * 60 + t.minute

    def _body_ratio(self, o, h, l, c):
        rng = h - l
        if rng <= 0:
            return 0.0
        return abs(c - o) / rng

    def _volume_ok(self):
        if not self.p.use_volume_filter:
            return True
        avg = float(self.vol_sma[0])
        if avg <= 0:
            return True  # feed has no volume data
        return float(self.data.volume[0]) > avg

    def _record_chunk(self, fill_price, chunk_size, risk_share, reason):
        meta = self.pos_meta
        pnl = (
            (fill_price - meta["avg_entry"])
            * meta["direction"] * chunk_size * self._contract_size()
        )
        self.trade_records.append(dict(
            setup=f"vwap-reversion-t{meta['tiers']}",
            day_type="-",
            direction="long" if meta["direction"] == 1 else "short",
            entry_dt=meta["entry_dt"],
            exit_dt=self.data.datetime.datetime(0),
            pnl=pnl,
            pnlcomm=pnl,
            risk_cash=risk_share,
            r_multiple=pnl / risk_share if risk_share > 0 else 0.0,
            reason=reason,
        ))
        self.log(f"CLOSED | {reason} | Size:{chunk_size:.4f} PnL:{pnl:.2f}")

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
            fill_size = abs(float(order.executed.size))

            if ctx and ctx["kind"] == "entry":
                direction = ctx["direction"]
                stop_price = ctx["stop_price"]
                risk_add = abs(fill_price - stop_price) * fill_size * self._contract_size()

                if self.pos_meta is None:
                    self.pos_meta = dict(
                        direction=direction,
                        avg_entry=fill_price,
                        total_size=fill_size,
                        stop_price=stop_price,
                        risk_cash=risk_add,
                        entry_dt=self.data.datetime.datetime(0),
                        tiers=1,
                    )
                else:
                    meta = self.pos_meta
                    new_total = meta["total_size"] + fill_size
                    meta["avg_entry"] = (
                        meta["avg_entry"] * meta["total_size"] + fill_price * fill_size
                    ) / new_total
                    meta["total_size"] = new_total
                    # Never loosen the original stop; risk measured to it
                    meta["risk_cash"] += (
                        abs(fill_price - meta["stop_price"])
                        * fill_size * self._contract_size()
                    )
                    meta["tiers"] = 2

                self.phase = "normal"
                self.log(
                    f"{'LONG' if direction == 1 else 'SHORT'} "
                    f"TIER {self.pos_meta['tiers']} OPEN | Entry:{fill_price:.5f} "
                    f"Stop:{self.pos_meta['stop_price']:.5f} "
                    f"TotalSize:{self.pos_meta['total_size']:.4f}"
                )

            elif ctx and ctx["kind"] == "partial" and self.pos_meta is not None:
                meta = self.pos_meta
                risk_share = meta["risk_cash"] * (fill_size / meta["total_size"])
                self._record_chunk(fill_price, fill_size, risk_share, "vwap-partial")
                meta["total_size"] -= fill_size
                meta["risk_cash"] -= risk_share
                self.phase = "trail"
                self.extreme = fill_price
                self.trail_stop = (
                    fill_price - self.p.trail_atr_mult * float(self.atr[0])
                    if meta["direction"] == 1
                    else fill_price + self.p.trail_atr_mult * float(self.atr[0])
                )

            elif ctx and ctx["kind"] == "exit" and self.pos_meta is not None:
                meta = self.pos_meta
                self._record_chunk(
                    fill_price, meta["total_size"], meta["risk_cash"],
                    ctx.get("reason", "exit"))
                self.pos_meta = None
                self.phase = "normal"
                self.trail_stop = None
                self.extreme = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"ORDER FAILED | Status: {order.getstatusname()}")

        self.order = None
        self.order_ctx = None

    # ------------------------------------------------------------
    # Trading logic, executed on each candle
    # ------------------------------------------------------------
    def next(self) -> None:
        if self.order is not None:
            return
        if len(self.data) <= max(self.p.vol_period, self.p.atr_period) + 1:
            return

        mins = self._minutes_now()
        o = float(self.data.open[0])
        h = float(self.data.high[0])
        l = float(self.data.low[0])
        c = float(self.data.close[0])
        vwap = float(self.bands.vwap[0])
        dev = float(self.bands.dev[0])

        # Flat rule for the end of the session: no overnight / gap risk
        if mins >= self.p.flat_min:
            if self.position.size != 0 and self.pos_meta is not None:
                self.order_ctx = dict(kind="exit", reason="session-close")
                self.order = self.close()
            return

        # ------------------------------------------------------------
        # Manage the open position
        # ------------------------------------------------------------
        if self.position.size != 0 and self.pos_meta is not None:
            meta = self.pos_meta
            d = meta["direction"]

            if self.phase == "normal":
                stop_hit = l <= meta["stop_price"] if d == 1 else h >= meta["stop_price"]
                vwap_hit = h >= vwap if d == 1 else l <= vwap

                if stop_hit:
                    self.order_ctx = dict(kind="exit", reason="band-stop")
                    self.order = self.close()
                elif vwap_hit:
                    half = self._round_down_to_step(
                        meta["total_size"] / 2.0, self._lot_step())
                    if self.p.partial_exit_at_vwap and half >= self.p.min_size:
                        self.order_ctx = dict(kind="partial")
                        self.order = (
                            self.sell(size=half) if d == 1 else self.buy(size=half))
                    else:
                        self.order_ctx = dict(kind="exit", reason="vwap-target")
                        self.order = self.close()
                if self.order is not None:
                    return
            else:  # trail phase: half is banked, trail the remainder
                atr_value = float(self.atr[0])
                if d == 1:
                    self.extreme = max(self.extreme, h)
                    self.trail_stop = max(
                        self.trail_stop,
                        self.extreme - self.p.trail_atr_mult * atr_value)
                    if l <= self.trail_stop:
                        self.order_ctx = dict(kind="exit", reason="trail-stop")
                        self.order = self.close()
                        return
                else:
                    self.extreme = min(self.extreme, l)
                    self.trail_stop = min(
                        self.trail_stop,
                        self.extreme + self.p.trail_atr_mult * atr_value)
                    if h >= self.trail_stop:
                        self.order_ctx = dict(kind="exit", reason="trail-stop")
                        self.order = self.close()
                        return

        # ------------------------------------------------------------
        # Entries and tier-2 add-ons
        # ------------------------------------------------------------
        if mins < self.p.session_open_min + self.p.skip_open_minutes:
            return
        if mins > self.p.last_entry_min:
            return
        if dev <= 0:
            return

        body = self._body_ratio(o, h, l, c)
        prev_low = float(self.data.low[-1])
        prev_high = float(self.data.high[-1])

        # Tier-2 add-on: deeper stretch against an existing position
        if self.pos_meta is not None:
            meta = self.pos_meta
            if (
                self.p.enable_tier2
                and self.phase == "normal"
                and meta["tiers"] == 1
            ):
                if meta["direction"] == 1:
                    band2 = vwap - self.p.band_mult_2 * dev
                    stretched = min(l, prev_low) <= band2
                    confirmed = c > o and body >= self.p.confirm_body_min and c > band2
                else:
                    band2 = vwap + self.p.band_mult_2 * dev
                    stretched = max(h, prev_high) >= band2
                    confirmed = c < o and body >= self.p.confirm_body_min and c < band2

                if stretched and confirmed and self._volume_ok():
                    stop_distance = abs(c - meta["stop_price"])
                    size = self._calc_size(stop_distance)
                    if size > 0:
                        self.order_ctx = dict(
                            kind="entry", direction=meta["direction"],
                            stop_price=meta["stop_price"])
                        self.log(f"TIER 2 ADD | C:{c:.5f} Band2:{band2:.5f} Size:{size:.4f}")
                        self.order = (
                            self.buy(size=size) if meta["direction"] == 1
                            else self.sell(size=size))
            return

        # Fresh tier-1 entries
        if self.p.allow_long:
            band1 = vwap - self.p.band_mult_1 * dev
            stretched = min(l, prev_low) <= band1
            confirmed = c > o and body >= self.p.confirm_body_min and c > band1
            if stretched and confirmed and self._volume_ok():
                stop_price = vwap - self.p.stop_band_mult * dev
                stop_distance = c - stop_price
                size = self._calc_size(stop_distance)
                if size > 0:
                    self.order_ctx = dict(kind="entry", direction=1, stop_price=stop_price)
                    self.log(
                        f"LONG REVERSION | C:{c:.5f} VWAP:{vwap:.5f} "
                        f"Band1:{band1:.5f} Size:{size:.4f}")
                    self.order = self.buy(size=size)
                    return

        if self.p.allow_short:
            band1 = vwap + self.p.band_mult_1 * dev
            stretched = max(h, prev_high) >= band1
            confirmed = c < o and body >= self.p.confirm_body_min and c < band1
            if stretched and confirmed and self._volume_ok():
                stop_price = vwap + self.p.stop_band_mult * dev
                stop_distance = stop_price - c
                size = self._calc_size(stop_distance)
                if size > 0:
                    self.order_ctx = dict(kind="entry", direction=-1, stop_price=stop_price)
                    self.log(
                        f"SHORT REVERSION | C:{c:.5f} VWAP:{vwap:.5f} "
                        f"Band1:{band1:.5f} Size:{size:.4f}")
                    self.order = self.sell(size=size)

    # ====================================================================================
    # params_metadata is optional -- it is used to configure the "Run Backtest/Bot" modal.
    params_metadata = {
        "band_mult_1": {
            "label": "Entry band (std devs)",
            "helper_text": "First entry tier: deviation from VWAP in standard deviations",
            "value_type": "float",
        },
        "band_mult_2": {
            "label": "Add-on band (std devs)",
            "helper_text": "Tier-2 add-on deviation from VWAP",
            "value_type": "float",
        },
        "enable_tier2": {
            "label": "Enable tier-2 add-on",
            "helper_text": "Add a second unit if price stretches to the add-on band",
            "value_type": "bool",
        },
        "stop_band_mult": {
            "label": "Stop band (std devs)",
            "helper_text": "Stop loss beyond this deviation band, frozen at entry",
            "value_type": "float",
        },
        "confirm_body_min": {
            "label": "Confirmation body",
            "helper_text": "Minimum body/range ratio of the reversal candle",
            "value_type": "float",
        },
        "use_volume_filter": {
            "label": "Volume filter",
            "helper_text": "Require the reversal candle's volume above its rolling average",
            "value_type": "bool",
        },
        "partial_exit_at_vwap": {
            "label": "Partial exit at VWAP",
            "helper_text": "Take half off at VWAP and trail the rest with an ATR stop",
            "value_type": "bool",
        },
        "trail_atr_mult": {
            "label": "Trail (ATRs)",
            "helper_text": "Trailing stop distance after the partial exit",
            "value_type": "float",
        },
        "skip_open_minutes": {
            "label": "Skip after open (min)",
            "helper_text": "No signals for this many minutes after the session open",
            "value_type": "int",
        },
        "last_entry_min": {
            "label": "Last entry (min)",
            "helper_text": "No new entries after this time (minutes from midnight)",
            "value_type": "int",
        },
        "flat_min": {
            "label": "Flat by (min)",
            "helper_text": "Close everything after this time -- avoids overnight gap risk",
            "value_type": "int",
        },
        "risk_per_trade_pct": {
            "label": "Risk per trade %",
            "helper_text": "Percent of account equity risked per entry tier",
            "value_type": "float",
        },
    }
