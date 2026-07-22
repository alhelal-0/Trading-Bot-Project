# Bots on TradeLocker are implemented in Python using the Backtrader framework.
# You can find more examples and the complete documentation at https://www.backtrader.com/

# Trend Following V2 -- institutional-grade trend system.
#
# Philosophy: never predict, only confirm; protect capital first; let
# winners run, cut losers fast; never average down; every rule objective.
#
# ENTRY (ALL filters must agree -- quality over quantity):
#   1.  Higher-timeframe trend proxy: price on the right side of the
#       200 EMA (the chart-timeframe stand-in for HTF confirmation)
#   2.  EMA alignment: fast(9) vs slow(21) stacked with the trend
#   3.  EMA slope: BOTH fast and slow EMAs rising (falling for shorts),
#       slope normalized by ATR so it is instrument-independent
#   4.  Trend quality: ADX above threshold (no ranging markets)
#   5.  Market structure: last two pivot highs AND lows ascending for
#       longs (HH/HL), descending for shorts (LH/LL)
#   6.  Momentum: RSI on the trend side of 50 and rising/falling with it
#   7.  Volume: entry bar volume above its rolling average (auto-passes
#       on feeds without volume)
#   8.  Volatility floor: ATR above a fraction of its own average --
#       dead markets produce false trends
#   9.  Spike/news guard: no entries while any recent bar's range
#       exceeds a spike multiple of ATR (proxy for a news candle)
#   10. Session filter (optional): entries only inside chosen hours
#   11. Trigger: pullback into the EMA zone + strong confirmation candle
#       (big body, closing near its extreme, NOT an inside bar)
#   12. Overextension guards: skip if price is too far from the slow EMA
#       or the structural stop would exceed a max ATR multiple (late,
#       chasing entries are declined)
#
# EXITS (in priority order):
#   - Initial stop beyond the signal bar / swing, clamped to a minimum
#     ATR distance (never tighter than noise)
#   - Break-even at +`be_trigger_r` R
#   - Partial profit: take a configurable fraction off at +1R
#   - ATR trailing stop on the remainder once +`trail_trigger_r` R
#   - Trend-reversal exit: fast/slow EMA cross against the position
#   - Momentum-loss exit: RSI collapses through its exit level
#
# RISK:
#   - 0.5% risk per trade (ATR/structure-based position sizing)
#   - Size halved when volatility is elevated (vol-adjusted sizing)
#   - Daily kill switch at -4% equity, weekly at -8%
#   - Trading blocked for the day after `max_consec_losses` losses
#   - One position at a time; never averages down
#
# Session times are minutes from midnight in the FEED's timezone.

import math

import backtrader as bt


class TrendFollowingV2(bt.Strategy):
    params = {
        # EMAs / higher-timeframe proxy
        "fast_period": 9,
        "slow_period": 21,
        "trend_period": 200,
        # Slope
        "slope_lookback": 6,
        "slope_min": 0.05,           # per-bar EMA slope normalized by ATR
        # Trend quality
        "adx_period": 14,
        "adx_min": 20.0,
        # Structure
        "structure_lookback": 40,    # bars scanned for HH/HL pivots
        # Momentum
        "rsi_period": 14,
        "momentum_exit_level": 40.0, # long exits if RSI drops through this
        "use_momentum_exit": True,
        # Volume
        "vol_period": 20,
        "use_volume_filter": True,
        # Volatility
        "atr_period": 14,
        "atr_baseline": 100,         # ATR average for the volatility floor
        "vol_floor_ratio": 0.6,      # ATR must exceed this x its average
        "high_vol_ratio": 1.5,       # above this x average = elevated vol
        "high_vol_size_factor": 0.5, # halve size in elevated vol
        # Spike / news guard
        "news_spike_atr": 3.0,
        "news_guard_bars": 3,
        # Entry trigger quality
        "ema_touch_atr": 0.25,
        "signal_body_min": 0.50,
        "signal_close_pct": 0.30,
        "max_ext_atr": 3.0,          # max distance from slow EMA (no chasing)
        # Stops / exits
        "swing_lookback": 3,
        "stop_pad_ticks": 2,
        "min_stop_atr": 1.0,         # never tighter than noise
        "max_stop_atr": 3.0,         # wider than this = late entry, skip
        "be_trigger_r": 0.8,
        "use_partial": True,
        "partial_target_r": 1.0,
        "partial_pct": 50.0,         # % of the position taken off at the partial
        "trail_trigger_r": 1.0,
        "trail_atr_mult": 2.0,
        # Risk limits
        "risk_per_trade_pct": 0.5,
        "daily_loss_limit_pct": 4.0,
        "weekly_loss_limit_pct": 8.0,
        "max_consec_losses": 3,
        # Session filter (optional)
        "use_session_filter": False,
        "session_open_min": 9 * 60 + 30,
        "last_entry_min": 15 * 60 + 30,
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
        self.ema_trend = bt.indicators.ExponentialMovingAverage(
            self.data.close, period=self.p.trend_period)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(
            self.data, period=self.p.adx_period)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.vol_sma = bt.indicators.SimpleMovingAverage(
            self.data.volume, period=self.p.vol_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.atr_avg = bt.indicators.SimpleMovingAverage(
            self.atr, period=self.p.atr_baseline)

        self.order = None
        self.order_ctx = None

        # One aggregate position (partials shrink it, never grows: no averaging down)
        self.pos_meta = None
        self.trail_active = False
        self.extreme = None

        # Risk-limit state
        self.cur_date = None
        self.cur_week = None
        self.day_start_equity = None
        self.week_start_equity = None
        self.blocked_today = False
        self.blocked_week = False
        self.consec_losses = 0

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

    def _calc_size(self, stop_distance, risk_pct):
        equity = float(self.broker.getvalue())
        risk_cash = equity * (risk_pct / 100.0)
        unit_risk = max(stop_distance, self._tick_size()) * self._contract_size()
        if unit_risk <= 0:
            return 0.0
        size = self._round_down_to_step(risk_cash / unit_risk, self._lot_step())
        return size if size >= self.p.min_size else 0.0

    def _warmed_up(self):
        needed = max(self.p.trend_period, self.p.atr_baseline,
                     self.p.structure_lookback + 2, self.p.adx_period * 2) + 1
        return len(self.data) > needed

    def _minutes_now(self):
        t = self.data.datetime.time(0)
        return t.hour * 60 + t.minute

    def _slope_per_atr(self, line, atr_value):
        lb = int(self.p.slope_lookback)
        if atr_value <= 0:
            return 0.0
        return (float(line[0]) - float(line[-lb])) / (atr_value * lb)

    def _swing_low(self):
        return min(float(self.data.low[-i]) for i in range(0, self.p.swing_lookback + 1))

    def _swing_high(self):
        return max(float(self.data.high[-i]) for i in range(0, self.p.swing_lookback + 1))

    def _pivots(self, get, is_extreme):
        # 1-bar pivots over the structure lookback, nearest first
        out = []
        for i in range(2, self.p.structure_lookback):
            v = get(i)
            if is_extreme(v, get(i - 1)) and is_extreme(v, get(i + 1)):
                out.append(v)
                if len(out) == 2:
                    break
        return out

    def _structure(self):
        """Returns +1 for HH/HL, -1 for LH/LL, 0 for unclear."""
        highs = self._pivots(lambda i: float(self.data.high[-i]), lambda a, b: a >= b)
        lows = self._pivots(lambda i: float(self.data.low[-i]), lambda a, b: a <= b)
        if len(highs) < 2 or len(lows) < 2:
            return 0
        if highs[0] > highs[1] and lows[0] > lows[1]:
            return 1
        if highs[0] < highs[1] and lows[0] < lows[1]:
            return -1
        return 0

    def _volume_ok(self):
        if not self.p.use_volume_filter:
            return True
        avg = float(self.vol_sma[0])
        if avg <= 0:
            return True
        return float(self.data.volume[0]) > avg

    def _record_chunk(self, fill_price, chunk_size, risk_share, reason):
        meta = self.pos_meta
        pnl = (
            (fill_price - meta["entry_price"])
            * meta["direction"] * chunk_size * self._contract_size()
        )
        meta["realized_pnl"] += pnl
        self.trade_records.append(dict(
            setup="trend-v2",
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

    def _finish_position(self):
        # Consecutive-loss tracking on the completed position
        if self.pos_meta["realized_pnl"] < 0:
            self.consec_losses += 1
            if self.consec_losses >= self.p.max_consec_losses:
                self.blocked_today = True
                self.log(
                    f"RISK BLOCK | {self.consec_losses} consecutive losses "
                    f"-- no more entries today")
        else:
            self.consec_losses = 0
        self.pos_meta = None
        self.trail_active = False
        self.extreme = None

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
                stop_distance = ctx["stop_distance"]
                self.pos_meta = dict(
                    direction=ctx["direction"],
                    entry_dt=self.data.datetime.datetime(0),
                    entry_price=fill_price,
                    total_size=fill_size,
                    stop_distance=stop_distance,
                    stop_price=(
                        fill_price - stop_distance if ctx["direction"] == 1
                        else fill_price + stop_distance),
                    risk_cash=stop_distance * fill_size * self._contract_size(),
                    partial_done=False,
                    at_breakeven=False,
                    realized_pnl=0.0,
                )
                self.extreme = fill_price
                self.trail_active = False
                self.log(
                    f"{'LONG' if ctx['direction'] == 1 else 'SHORT'} OPEN | "
                    f"Entry:{fill_price:.5f} Stop:{self.pos_meta['stop_price']:.5f} "
                    f"Size:{fill_size:.4f}")

            elif ctx and ctx["kind"] == "partial" and self.pos_meta is not None:
                meta = self.pos_meta
                risk_share = meta["risk_cash"] * (fill_size / meta["total_size"])
                self._record_chunk(fill_price, fill_size, risk_share, "partial-1r")
                meta["total_size"] -= fill_size
                meta["risk_cash"] -= risk_share
                meta["partial_done"] = True

            elif ctx and ctx["kind"] == "exit" and self.pos_meta is not None:
                meta = self.pos_meta
                self._record_chunk(
                    fill_price, meta["total_size"], meta["risk_cash"],
                    ctx.get("reason", "exit"))
                self._finish_position()

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"ORDER FAILED | Status: {order.getstatusname()}")

        self.order = None
        self.order_ctx = None

    # ------------------------------------------------------------
    # Trading logic, executed on each candle
    # ------------------------------------------------------------
    def next(self) -> None:
        if not self._warmed_up():
            return

        # ------- Daily / weekly risk-limit bookkeeping -------
        d = self.data.datetime.date(0)
        if d != self.cur_date:
            self.cur_date = d
            self.day_start_equity = float(self.broker.getvalue())
            self.blocked_today = False
            self.consec_losses = 0
        week = d.isocalendar()[:2]
        if week != self.cur_week:
            self.cur_week = week
            self.week_start_equity = float(self.broker.getvalue())
            self.blocked_week = False

        if self.order is not None:
            return

        equity = float(self.broker.getvalue())
        if (
            not self.blocked_today
            and self.day_start_equity
            and equity <= self.day_start_equity * (1 - self.p.daily_loss_limit_pct / 100.0)
        ):
            self.blocked_today = True
            self.log(
                f"KILL SWITCH | daily loss {self.p.daily_loss_limit_pct}% hit "
                f"-- flat, done today")
        if (
            not self.blocked_week
            and self.week_start_equity
            and equity <= self.week_start_equity * (1 - self.p.weekly_loss_limit_pct / 100.0)
        ):
            self.blocked_week = True
            self.log(
                f"KILL SWITCH | weekly loss {self.p.weekly_loss_limit_pct}% hit "
                f"-- flat, done this week")

        if (self.blocked_today or self.blocked_week) and self.position.size != 0:
            self.order_ctx = dict(kind="exit", reason="kill-switch")
            self.order = self.close()
            return

        o = float(self.data.open[0])
        h = float(self.data.high[0])
        l = float(self.data.low[0])
        c = float(self.data.close[0])
        atr_value = float(self.atr[0])

        # ------- Manage the open position -------
        if self.position.size != 0 and self.pos_meta is not None:
            meta = self.pos_meta
            dd = meta["direction"]
            sd = meta["stop_distance"]

            if dd == 1:
                self.extreme = max(self.extreme, h)
                favorable = self.extreme - meta["entry_price"]
            else:
                self.extreme = min(self.extreme, l)
                favorable = meta["entry_price"] - self.extreme

            # Break-even
            if not meta["at_breakeven"] and favorable >= sd * self.p.be_trigger_r:
                meta["stop_price"] = (
                    max(meta["stop_price"], meta["entry_price"]) if dd == 1
                    else min(meta["stop_price"], meta["entry_price"]))
                meta["at_breakeven"] = True
                self.log("BREAK-EVEN | stop moved to entry")

            # Trailing
            if not self.trail_active and favorable >= sd * self.p.trail_trigger_r:
                self.trail_active = True
                self.log("TRAIL ACTIVE")
            if self.trail_active:
                if dd == 1:
                    meta["stop_price"] = max(
                        meta["stop_price"],
                        self.extreme - self.p.trail_atr_mult * atr_value)
                else:
                    meta["stop_price"] = min(
                        meta["stop_price"],
                        self.extreme + self.p.trail_atr_mult * atr_value)

            # Exit checks (stop first, then reversal signals)
            stop_hit = l <= meta["stop_price"] if dd == 1 else h >= meta["stop_price"]
            ema_cross_against = (
                float(self.ema_fast[0]) < float(self.ema_slow[0]) if dd == 1
                else float(self.ema_fast[0]) > float(self.ema_slow[0]))
            momo_lost = self.p.use_momentum_exit and (
                float(self.rsi[0]) < self.p.momentum_exit_level if dd == 1
                else float(self.rsi[0]) > 100.0 - self.p.momentum_exit_level)

            if stop_hit:
                reason = "trail-stop" if self.trail_active else "stop"
                self.order_ctx = dict(kind="exit", reason=reason)
                self.order = self.close()
                return
            if ema_cross_against:
                self.order_ctx = dict(kind="exit", reason="trend-reversal")
                self.order = self.close()
                return
            if momo_lost:
                self.order_ctx = dict(kind="exit", reason="momentum-loss")
                self.order = self.close()
                return

            # Partial profit at target R
            if (
                self.p.use_partial
                and not meta["partial_done"]
                and favorable >= sd * self.p.partial_target_r
            ):
                chunk = self._round_down_to_step(
                    meta["total_size"] * self.p.partial_pct / 100.0, self._lot_step())
                if chunk >= self.p.min_size and chunk < meta["total_size"]:
                    self.order_ctx = dict(kind="partial")
                    self.order = (
                        self.sell(size=chunk) if dd == 1 else self.buy(size=chunk))
                    return
            return

        # ------- Entries -------
        if self.blocked_today or self.blocked_week:
            return
        if self.p.use_session_filter:
            mins = self._minutes_now()
            if mins < self.p.session_open_min or mins > self.p.last_entry_min:
                return
        if atr_value <= 0:
            return

        # Volatility floor: dead markets produce false trends
        atr_mean = float(self.atr_avg[0])
        if atr_mean > 0 and atr_value < self.p.vol_floor_ratio * atr_mean:
            return

        # Spike / news guard: stand aside around abnormal candles
        for i in range(0, self.p.news_guard_bars):
            rng_i = float(self.data.high[-i]) - float(self.data.low[-i])
            if rng_i > self.p.news_spike_atr * atr_value:
                return

        ema_fast = float(self.ema_fast[0])
        ema_slow = float(self.ema_slow[0])
        ema_trend = float(self.ema_trend[0])
        slope_fast = self._slope_per_atr(self.ema_fast, atr_value)
        slope_slow = self._slope_per_atr(self.ema_slow, atr_value)
        adx_ok = float(self.adx.adx[0]) >= self.p.adx_min
        structure = self._structure()
        rsi_now = float(self.rsi[0])
        rsi_prev = float(self.rsi[-self.p.slope_lookback])

        rng = h - l
        body_ratio = abs(c - o) / rng if rng > 0 else 0.0
        closes_high = rng > 0 and (h - c) <= self.p.signal_close_pct * rng
        closes_low = rng > 0 and (c - l) <= self.p.signal_close_pct * rng
        inside_bar = h <= float(self.data.high[-1]) and l >= float(self.data.low[-1])

        touch = self.p.ema_touch_atr * atr_value
        pad = self._tick_size() * self.p.stop_pad_ticks

        def try_enter(direction):
            # All filters must agree; the first failure declines the trade
            if direction == 1:
                if not (c > ema_trend and ema_fast > ema_slow):
                    return
                if not (slope_fast >= self.p.slope_min and slope_slow >= self.p.slope_min):
                    return
                if not adx_ok or structure != 1:
                    return
                if not (rsi_now > 50.0 and rsi_now >= rsi_prev):
                    return
                if not self._volume_ok():
                    return
                pulled = min(l, float(self.data.low[-1])) <= max(ema_fast, ema_slow) + touch
                signal = c > o and body_ratio >= self.p.signal_body_min and closes_high
                if not (pulled and signal) or inside_bar:
                    return
                if (c - ema_slow) > self.p.max_ext_atr * atr_value:
                    self.log("DECLINED | overextended above the slow EMA")
                    return
                stop_price = min(l, self._swing_low()) - pad
                stop_distance = c - stop_price
            else:
                if not (c < ema_trend and ema_fast < ema_slow):
                    return
                if not (slope_fast <= -self.p.slope_min and slope_slow <= -self.p.slope_min):
                    return
                if not adx_ok or structure != -1:
                    return
                if not (rsi_now < 50.0 and rsi_now <= rsi_prev):
                    return
                if not self._volume_ok():
                    return
                pulled = max(h, float(self.data.high[-1])) >= min(ema_fast, ema_slow) - touch
                signal = c < o and body_ratio >= self.p.signal_body_min and closes_low
                if not (pulled and signal) or inside_bar:
                    return
                if (ema_slow - c) > self.p.max_ext_atr * atr_value:
                    self.log("DECLINED | overextended below the slow EMA")
                    return
                stop_price = max(h, self._swing_high()) + pad
                stop_distance = stop_price - c

            # Stop sanity: never tighter than noise, never a chasing entry
            if stop_distance < self.p.min_stop_atr * atr_value:
                stop_distance = self.p.min_stop_atr * atr_value
            if stop_distance > self.p.max_stop_atr * atr_value:
                self.log("DECLINED | structural stop too wide (late entry)")
                return

            # Volatility-adjusted sizing: reduce risk in elevated volatility
            risk_pct = self.p.risk_per_trade_pct
            if atr_mean > 0 and atr_value > self.p.high_vol_ratio * atr_mean:
                risk_pct *= self.p.high_vol_size_factor

            size = self._calc_size(stop_distance, risk_pct)
            if size <= 0:
                return

            self.order_ctx = dict(kind="entry", direction=direction,
                                  stop_distance=stop_distance)
            self.log(
                f"{'LONG' if direction == 1 else 'SHORT'} ENTRY | all filters "
                f"agree | ADX:{float(self.adx.adx[0]):.1f} "
                f"Slope:{slope_slow:.3f} Risk:{risk_pct:.2f}% Size:{size:.4f}")
            self.order = self.buy(size=size) if direction == 1 else self.sell(size=size)

        if self.p.allow_long:
            try_enter(1)
        if self.order is None and self.p.allow_short:
            try_enter(-1)

    # ====================================================================================
    # params_metadata is optional -- it is used to configure the "Run Backtest/Bot" modal.
    params_metadata = {
        "fast_period": {
            "label": "Fast EMA",
            "helper_text": "Fast EMA of the entry timeframe",
            "value_type": "int",
        },
        "slow_period": {
            "label": "Slow EMA",
            "helper_text": "Slow EMA of the entry timeframe",
            "value_type": "int",
        },
        "trend_period": {
            "label": "Trend EMA (HTF proxy)",
            "helper_text": "Long EMA acting as the higher-timeframe trend filter",
            "value_type": "int",
        },
        "slope_min": {
            "label": "Min EMA slope",
            "helper_text": "Both EMAs must rise/fall at least this fast (ATR-normalized)",
            "value_type": "float",
        },
        "adx_min": {
            "label": "Min ADX",
            "helper_text": "Trend-quality floor; below this the market is ranging",
            "value_type": "float",
        },
        "vol_floor_ratio": {
            "label": "Volatility floor",
            "helper_text": "ATR must exceed this fraction of its average",
            "value_type": "float",
        },
        "max_ext_atr": {
            "label": "Max extension (ATRs)",
            "helper_text": "Skip entries further than this from the slow EMA",
            "value_type": "float",
        },
        "max_stop_atr": {
            "label": "Max stop (ATRs)",
            "helper_text": "Skip entries whose structural stop is wider than this",
            "value_type": "float",
        },
        "be_trigger_r": {
            "label": "Break-even (R)",
            "helper_text": "Move the stop to entry after this much profit",
            "value_type": "float",
        },
        "partial_pct": {
            "label": "Partial size %",
            "helper_text": "Fraction of the position taken off at +1R",
            "value_type": "float",
        },
        "trail_atr_mult": {
            "label": "Trail (ATRs)",
            "helper_text": "Trailing stop distance once trailing activates",
            "value_type": "float",
        },
        "risk_per_trade_pct": {
            "label": "Risk per trade %",
            "helper_text": "Percent of equity risked per trade (0.5-1 recommended)",
            "value_type": "float",
        },
        "daily_loss_limit_pct": {
            "label": "Daily loss limit %",
            "helper_text": "Flat and no trading for the rest of the day beyond this",
            "value_type": "float",
        },
        "weekly_loss_limit_pct": {
            "label": "Weekly loss limit %",
            "helper_text": "Flat and no trading for the rest of the week beyond this",
            "value_type": "float",
        },
        "max_consec_losses": {
            "label": "Max consecutive losses",
            "helper_text": "Stop entering for the day after this many losses in a row",
            "value_type": "int",
        },
        "use_session_filter": {
            "label": "Session filter",
            "helper_text": "Restrict entries to the configured session hours",
            "value_type": "bool",
        },
    }
