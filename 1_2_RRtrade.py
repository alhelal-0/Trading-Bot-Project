import math
import backtrader as bt


class OneToTwoRRBot(bt.Strategy):
    """
    NAS100 / USTEC / MNQ 5-minute price action bot — fixed 1:2 RR variant.

    Same day-type/price-action engine as nas100_price_action_bot.py, with
    a strict risk contract:
      - EVERY setup takes profit at exactly 1:2 (risk one, make two)
      - Risk exactly 1% of the account on each trade
      - Equity down 4% from the day's start -> flat, trading shut off
        until the next day (kill switch)

    1. Classify the day at the open before trading:
       - First 3 candles all big red  -> bear trend day (only sells)
       - First 3 candles all big bull -> bull trend day (only buys)
       - Surprise bar at the open     -> bias in its direction (60% trend day)
       - Big up + big down, confusion -> trading range day (buy low, sell high)
       - Unclear -> wait, let price break out of the opening range box
    2. Keep the bias all day. Only flip after 3 big opposite candles
       closing at their extremes (confirmed reversal); a flip closes
       every open leg.
    3. Entries only with the bias, only on a CLOSED strong signal candle:
       - Pullback to the 9/15 EMA zone rejected by a candle closing
         at its high (buy) / low (sell)
       - Surprise bar: DISABLED for entries in this variant (worst
         performer in backtests; still sets the day's bias)
       - Double bottom / double top: two lows (highs) at the same level,
         then a strong rejection candle
       - Range day: bull signal in the lower third of the day's range,
         bear signal in the upper third
       - Choppy / sideways stretch ("avoiding zone": EMAs intertwined,
         price weaving across them) -> NO entries of any setup until
         the market resolves
       All of them target 1:2, and every entry must agree with the
       intraday VWAP: buys only when price is above it, sells only
       when price is below it.
    4. Risk 1% per trade, SL beyond the signal bar / recent swing,
       no adding to losers, stop to break-even after ~1R. Once every
       open leg is risk-free at break-even, the bot may pyramid another
       leg in the same direction on the next signal, up to
       max_concurrent_legs — each add-on also risks 1% and targets 1:2.
    5. Trades all hours by default; the day is classified from the first
       candles after each new day starts. Set trade_all_hours=False to
       restrict to the NY session and force flat before the close.
    6. Kill switch: if equity drops 4% from the day's starting value,
       everything is closed and no trades are taken until the next day.
       Separately, no single trade may lose more than 1% of portfolio
       value — a leg is force-closed the moment its open loss hits that
       cap, even if its stop hasn't been touched (gap/slippage guard).
    """

    params = dict(
        # Core indicators
        fast_period=9,
        slow_period=15,
        atr_period=14,

        # Session control
        trade_all_hours=True,            # True: trade around the clock, never force flat
        # Used only when trade_all_hours=False (minutes from midnight, feed time)
        session_open_min=9 * 60 + 30,    # 09:30 NY open
        last_entry_min=15 * 60 + 30,     # no new trades after 15:30
        flat_min=15 * 60 + 55,           # force flat by 15:55
        bar_minutes=5,

        # Day classification
        classify_bars=3,                 # first 15 minutes decide the day
        big_candle_atr=1.3,              # "big" candle range vs ATR
        surprise_atr=2.0,                # surprise bar range vs ATR
        surprise_body_min=0.70,

        # Setup toggles
        enable_surprise_bar=False,       # disabled: worst performer in backtests
                                         # (still used for day classification)

        # VWAP filter (intraday, resets each day)
        use_vwap_filter=True,            # buys only above VWAP, sells only below it

        # Signal candle quality
        signal_body_min=0.50,            # body must dominate the candle
        signal_close_pct=0.30,           # close within 30% of its extreme
        spike_range_atr=3.0,             # news-type spike: skip if huge range
        ema_touch_atr=0.25,              # how close a pullback must come to the 9 EMA

        # Trend strength (9/15 EMA angle proxy)
        slope_lookback=6,
        slope_min=0.08,                  # EMA15 slope per bar, normalized by ATR

        # Chop filter ("avoiding zone")
        chop_lookback=10,                # bars to inspect for weaving
        chop_max_flips=3,                # close flipping across the 9 EMA this often = chop
        chop_ema_dist_atr=0.15,          # EMAs closer than this (in ATRs) = intertwined
        chop_blocks_all=True,            # choppy/sideways: take NO trades of any setup

        # Range day
        min_range_atr=2.0,               # day range must be worth trading
        range_third=0.3333,

        # Bias flip
        reversal_flip_candles=3,

        # Double bottom / double top
        dbdt_lookback=20,                # bars to scan for the two lows/highs
        dbdt_min_separation=3,           # bars between the two touches
        dbdt_tolerance_atr=0.30,         # how equal the two levels must be
        dbdt_entry_dist_atr=1.0,         # signal candle must react this close to the level

        # Pyramiding ("keep buying until the trend reverses")
        max_concurrent_legs=3,
        pyramid_risk_factor=1.0,         # add-on legs risk this fraction of base risk

        # Kill switch
        daily_loss_limit_pct=4.0,        # equity down this much from day start -> flat, done for the day
        trade_loss_limit_pct=1.0,        # one leg's open loss may never exceed this much of equity

        # Risk management
        risk_per_trade_pct=1.0,          # exactly 1% of the account per trade
        rr_trend=2.0,                    # fixed 1:2 on every setup in this variant
        rr_counter=2.0,                  # (counter/range setups too — no 1:1 here)
        breakeven_trigger_r=1.0,
        stop_pad_ticks=2,
        swing_lookback=3,

        # Position sizing
        min_size=0.01,
        size_step=0.01,

        # Logging
        verbose=True,
    )

    def __init__(self):
        self.ema_fast = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.p.fast_period)
        self.ema_slow = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.p.slow_period)
        self.atr = bt.indicators.AverageTrueRange(self.data, period=self.p.atr_period)

        # Order / leg state (multiple same-direction legs = pyramiding)
        self.order = None
        self.order_ctx = None            # what the in-flight order is doing
        self.legs = []                   # open entries, each with its own stop/TP
        self.flatten_reason = None       # close everything on the next bar

        # Measurement: one record per closed leg, read by backtest_report.py
        self.trade_records = []

        # Kill switch state
        self.day_start_equity = None
        self.kill_switch = False         # tripped: no trading until the next day

        # Day state
        self.cur_date = None
        self.day_type = None             # 'bull' | 'bear' | 'range' | None
        self.open_candles = []           # first candles of the session
        self.or_high = None              # opening range box
        self.or_low = None
        self.day_high = None
        self.day_low = None
        self.opp_streak = 0              # big opposite candles in a row

        # Intraday VWAP (typical price x volume, resets each day;
        # falls back to TWAP when the feed has no volume)
        self.vwap_cum_pv = 0.0
        self.vwap_cum_vol = 0.0
        self.vwap = None

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def log(self, txt):
        if self.p.verbose:
            dt = self.data.datetime.datetime(0)
            print(f"{dt} | {txt}")

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
        return float(self.p.size_step)

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

    def _calc_size(self, entry_price, stop_price, risk_pct=None):
        equity = float(self.broker.getvalue())
        if risk_pct is None:
            risk_pct = self.p.risk_per_trade_pct
        risk_cash = equity * (risk_pct / 100.0)

        stop_distance = abs(float(entry_price) - float(stop_price))
        stop_distance = max(stop_distance, self._tick_size())

        cash_risk_per_1_lot = stop_distance * self._contract_size()
        if cash_risk_per_1_lot <= 0:
            return float(self.p.min_size)

        raw_size = risk_cash / cash_risk_per_1_lot
        rounded = self._round_down_to_step(raw_size, self._lot_step())

        if rounded < self.p.min_size:
            return float(self.p.min_size)
        return float(rounded)

    def _bar_minutes_now(self):
        t = self.data.datetime.time(0)
        return t.hour * 60 + t.minute

    def _candle(self, ago=0):
        return dict(
            o=float(self.data.open[ago]),
            h=float(self.data.high[ago]),
            l=float(self.data.low[ago]),
            c=float(self.data.close[ago]),
        )

    def _body_ratio(self, k):
        rng = k["h"] - k["l"]
        if rng <= 0:
            return 0.0
        return abs(k["c"] - k["o"]) / rng

    def _closes_at_high(self, k):
        rng = k["h"] - k["l"]
        if rng <= 0:
            return False
        return (k["h"] - k["c"]) <= self.p.signal_close_pct * rng

    def _closes_at_low(self, k):
        rng = k["h"] - k["l"]
        if rng <= 0:
            return False
        return (k["c"] - k["l"]) <= self.p.signal_close_pct * rng

    def _is_big(self, k, atr_value):
        return (k["h"] - k["l"]) >= self.p.big_candle_atr * atr_value

    def _is_bull_signal(self, k, atr_value):
        # Strong bull candle closing at its high, not a news spike
        if k["c"] <= k["o"]:
            return False
        if self._body_ratio(k) < self.p.signal_body_min:
            return False
        if not self._closes_at_high(k):
            return False
        if (k["h"] - k["l"]) > self.p.spike_range_atr * atr_value:
            return False
        return True

    def _is_bear_signal(self, k, atr_value):
        if k["c"] >= k["o"]:
            return False
        if self._body_ratio(k) < self.p.signal_body_min:
            return False
        if not self._closes_at_low(k):
            return False
        if (k["h"] - k["l"]) > self.p.spike_range_atr * atr_value:
            return False
        return True

    def _is_surprise_bull(self, k, atr_value):
        return (
            k["c"] > k["o"]
            and (k["h"] - k["l"]) >= self.p.surprise_atr * atr_value
            and self._body_ratio(k) >= self.p.surprise_body_min
            and self._closes_at_high(k)
        )

    def _is_surprise_bear(self, k, atr_value):
        return (
            k["c"] < k["o"]
            and (k["h"] - k["l"]) >= self.p.surprise_atr * atr_value
            and self._body_ratio(k) >= self.p.surprise_body_min
            and self._closes_at_low(k)
        )

    def _is_choppy(self, atr_value):
        # "Avoiding zone": price weaving back and forth across the 9 EMA,
        # or the two EMAs flat and intertwined
        lb = int(self.p.chop_lookback)
        if len(self.data) <= lb or atr_value <= 0:
            return True

        flips = 0
        prev_side = None
        for i in range(lb - 1, -1, -1):
            side = float(self.data.close[-i]) > float(self.ema_fast[-i])
            if prev_side is not None and side != prev_side:
                flips += 1
            prev_side = side
        if flips >= self.p.chop_max_flips:
            return True

        ema_dist = abs(float(self.ema_fast[0]) - float(self.ema_slow[0]))
        if ema_dist < self.p.chop_ema_dist_atr * atr_value:
            return True

        return False

    def _ema_slope_per_atr(self, atr_value):
        lb = int(self.p.slope_lookback)
        if len(self.data) <= lb or atr_value <= 0:
            return 0.0
        return (float(self.ema_slow[0]) - float(self.ema_slow[-lb])) / (atr_value * lb)

    def _recent_swing_low(self):
        lows = [float(self.data.low[-i]) for i in range(0, self.p.swing_lookback + 1)]
        return min(lows)

    def _recent_swing_high(self):
        highs = [float(self.data.high[-i]) for i in range(0, self.p.swing_lookback + 1)]
        return max(highs)

    def _record_leg_exit(self, leg, exit_price, reason):
        d = leg["direction"]
        pnl = (float(exit_price) - leg["entry_price"]) * leg["size"] * self._contract_size() * d
        risk = leg["risk_cash"]
        self.trade_records.append(dict(
            setup=leg["setup"],
            day_type=leg["day_type"],
            direction="long" if d == 1 else "short",
            entry_dt=leg["entry_dt"],
            exit_dt=self.data.datetime.datetime(0),
            pnl=pnl,
            pnlcomm=pnl,
            risk_cash=risk,
            r_multiple=pnl / risk if risk > 0 else 0.0,
        ))
        self.log(f"LEG CLOSED | {reason} | Setup:{leg['setup']} | PnL:{pnl:.2f}")

    def _pivot_lows(self, lookback):
        # Simple 1-bar pivots over closed bars, nearest first
        pivots = []
        for i in range(2, lookback):
            lo = float(self.data.low[-i])
            if lo <= float(self.data.low[-(i - 1)]) and lo <= float(self.data.low[-(i + 1)]):
                pivots.append((i, lo))
        return pivots

    def _pivot_highs(self, lookback):
        pivots = []
        for i in range(2, lookback):
            hi = float(self.data.high[-i])
            if hi >= float(self.data.high[-(i - 1)]) and hi >= float(self.data.high[-(i + 1)]):
                pivots.append((i, hi))
        return pivots

    def _double_bottom_signal(self, k, atr_value):
        # Two lows at the same level, then the caller's strong bull
        # rejection candle reacting at that level and holding above it
        if atr_value <= 0 or len(self.data) <= self.p.dbdt_lookback + 1:
            return None
        pivots = self._pivot_lows(int(self.p.dbdt_lookback))
        tol = self.p.dbdt_tolerance_atr * atr_value
        for j in range(len(pivots) - 1):
            i2, l2 = pivots[j]
            for i1, l1 in pivots[j + 1:]:
                if (i1 - i2) < self.p.dbdt_min_separation:
                    continue
                if abs(l1 - l2) > tol:
                    continue
                level = min(l1, l2)
                near = k["l"] <= level + self.p.dbdt_entry_dist_atr * atr_value
                held = k["c"] > level
                if near and held:
                    return dict(direction=1, rr=self.p.rr_counter, setup="double-bottom")
        return None

    def _double_top_signal(self, k, atr_value):
        if atr_value <= 0 or len(self.data) <= self.p.dbdt_lookback + 1:
            return None
        pivots = self._pivot_highs(int(self.p.dbdt_lookback))
        tol = self.p.dbdt_tolerance_atr * atr_value
        for j in range(len(pivots) - 1):
            i2, h2 = pivots[j]
            for i1, h1 in pivots[j + 1:]:
                if (i1 - i2) < self.p.dbdt_min_separation:
                    continue
                if abs(h1 - h2) > tol:
                    continue
                level = max(h1, h2)
                near = k["h"] >= level - self.p.dbdt_entry_dist_atr * atr_value
                held = k["c"] < level
                if near and held:
                    return dict(direction=-1, rr=self.p.rr_counter, setup="double-top")
        return None

    def _reset_day_state(self):
        self.day_type = None
        self.open_candles = []
        self.or_high = None
        self.or_low = None
        self.day_high = None
        self.day_low = None
        self.opp_streak = 0
        self.vwap_cum_pv = 0.0
        self.vwap_cum_vol = 0.0
        self.vwap = None

    # ------------------------------------------------------------
    # Day classification
    # ------------------------------------------------------------
    def _classify_day(self, atr_value):
        candles = self.open_candles
        bulls = [k for k in candles if k["c"] > k["o"]]
        bears = [k for k in candles if k["c"] < k["o"]]
        big_bulls = [k for k in bulls if self._is_big(k, atr_value)]
        big_bears = [k for k in bears if self._is_big(k, atr_value)]

        # 3 big red candles at the open -> bear day, only sell
        if len(bears) == len(candles) and len(big_bears) >= 2:
            return "bear"
        if len(bulls) == len(candles) and len(big_bulls) >= 2:
            return "bull"

        # Surprise bar at the open -> bias its direction
        for k in candles:
            if self._is_surprise_bull(k, atr_value):
                return "bull"
            if self._is_surprise_bear(k, atr_value):
                return "bear"

        # Big up AND big down -> confusion, trading range day
        if big_bulls and big_bears:
            return "range"

        return None

    def _maybe_resolve_unclear_day(self, k, atr_value):
        # Let price get out of the opening range box, with the EMAs agreeing
        if self.or_high is None or self.or_low is None:
            return
        slope = self._ema_slope_per_atr(atr_value)

        if (
            k["c"] > self.or_high
            and float(self.ema_fast[0]) > float(self.ema_slow[0])
            and self._is_bull_signal(k, atr_value)
        ):
            self.day_type = "bull"
            self.log(f"DAY RESOLVED | Opening range breakout up | Slope:{slope:.3f}")
        elif (
            k["c"] < self.or_low
            and float(self.ema_fast[0]) < float(self.ema_slow[0])
            and self._is_bear_signal(k, atr_value)
        ):
            self.day_type = "bear"
            self.log(f"DAY RESOLVED | Opening range breakdown | Slope:{slope:.3f}")
        elif slope >= self.p.slope_min and k["c"] > float(self.ema_fast[0]):
            self.day_type = "bull"
            self.log(f"DAY RESOLVED | Strong EMA slope up:{slope:.3f}")
        elif slope <= -self.p.slope_min and k["c"] < float(self.ema_fast[0]):
            self.day_type = "bear"
            self.log(f"DAY RESOLVED | Strong EMA slope down:{slope:.3f}")

    def _update_bias_flip(self, k, atr_value):
        # Don't even think about the other side until 3-4 big candles
        # close at their extremes against the bias
        if self.day_type == "bull":
            against = k["c"] < k["o"] and self._is_big(k, atr_value) and self._closes_at_low(k)
        elif self.day_type == "bear":
            against = k["c"] > k["o"] and self._is_big(k, atr_value) and self._closes_at_high(k)
        else:
            return

        self.opp_streak = self.opp_streak + 1 if against else 0

        if self.opp_streak >= self.p.reversal_flip_candles:
            old = self.day_type
            self.day_type = "bear" if old == "bull" else "bull"
            self.opp_streak = 0
            self.log(f"BIAS FLIP | {old} -> {self.day_type} | reversal confirmed")
            if self.legs:
                self.flatten_reason = "bias flip"

    # ------------------------------------------------------------
    # Backtrader callbacks
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
                sig = ctx["signal"]
                direction = sig["direction"]
                stop_distance = float(sig["stop_distance"])
                rr = float(sig["rr"])
                size = abs(float(order.executed.size))

                leg = dict(
                    direction=direction,
                    setup=sig["setup"],
                    day_type=self.day_type,
                    entry_dt=self.data.datetime.datetime(0),
                    entry_price=fill_price,
                    size=size,
                    stop_distance=stop_distance,
                    stop_price=fill_price - stop_distance if direction == 1 else fill_price + stop_distance,
                    tp_price=fill_price + stop_distance * rr if direction == 1 else fill_price - stop_distance * rr,
                    at_breakeven=False,
                    risk_cash=stop_distance * size * self._contract_size(),
                )
                self.legs.append(leg)

                self.log(
                    f"{'LONG' if direction == 1 else 'SHORT'} OPEN | "
                    f"Leg {len(self.legs)}/{self.p.max_concurrent_legs} | "
                    f"Entry:{fill_price:.2f} Stop:{leg['stop_price']:.2f} "
                    f"TP:{leg['tp_price']:.2f} RR:{rr:.1f} Setup:{sig['setup']}"
                )

            elif ctx and ctx["kind"] == "close":
                leg = ctx["leg"]
                self._record_leg_exit(leg, fill_price, ctx.get("reason", "close"))
                if leg in self.legs:
                    self.legs.remove(leg)

            elif ctx and ctx["kind"] == "flat":
                for leg in list(self.legs):
                    self._record_leg_exit(leg, fill_price, ctx.get("reason", "flat"))
                self.legs = []

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"ORDER FAILED | Status: {order.getstatusname()}")

        self.order = None
        self.order_ctx = None

    # ------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------
    def next(self):
        needed = max(self.p.slow_period, self.p.atr_period, self.p.slope_lookback + 1)
        if len(self.data) <= needed:
            return

        cur_date = self.data.datetime.date(0)
        if cur_date != self.cur_date:
            self.cur_date = cur_date
            self._reset_day_state()
            self.day_start_equity = float(self.broker.getvalue())
            self.kill_switch = False

        mins = self._bar_minutes_now()
        atr_value = float(self.atr[0])
        k = self._candle(0)

        if not self.p.trade_all_hours:
            # Outside the session: do nothing (pre/post market data)
            if mins < self.p.session_open_min:
                return

            # Force flat into the close
            if mins >= self.p.flat_min:
                if self.legs and self.order is None:
                    self.log("SESSION FLAT | Closing before end of day")
                    self.order_ctx = dict(kind="flat", reason="session close")
                    self.order = self.close()
                return

        # Track the day's range and the opening range box
        self.day_high = k["h"] if self.day_high is None else max(self.day_high, k["h"])
        self.day_low = k["l"] if self.day_low is None else min(self.day_low, k["l"])

        # Intraday VWAP
        typical = (k["h"] + k["l"] + k["c"]) / 3.0
        vol = float(self.data.volume[0])
        if vol <= 0:
            vol = 1.0  # no volume on this feed -> equal weights (TWAP)
        self.vwap_cum_pv += typical * vol
        self.vwap_cum_vol += vol
        self.vwap = self.vwap_cum_pv / self.vwap_cum_vol

        collecting = len(self.open_candles) < self.p.classify_bars
        if collecting:
            self.open_candles.append(k)
            self.or_high = k["h"] if self.or_high is None else max(self.or_high, k["h"])
            self.or_low = k["l"] if self.or_low is None else min(self.or_low, k["l"])
            if len(self.open_candles) == self.p.classify_bars:
                self.day_type = self._classify_day(atr_value)
                self.log(f"DAY TYPE | {self.day_type or 'unclear, waiting'}")
        elif self.day_type is None:
            self._maybe_resolve_unclear_day(k, atr_value)
        else:
            self._update_bias_flip(k, atr_value)

        if self.order is not None:
            return

        # ------------------------------------------------------------
        # Kill switch: down 4% from the day's start -> flat, done today
        # ------------------------------------------------------------
        equity = float(self.broker.getvalue())
        if (
            not self.kill_switch
            and self.day_start_equity
            and equity <= self.day_start_equity * (1.0 - self.p.daily_loss_limit_pct / 100.0)
        ):
            self.kill_switch = True
            drop = (self.day_start_equity - equity) / self.day_start_equity * 100.0
            self.log(
                f"KILL SWITCH | Daily loss {drop:.2f}% >= {self.p.daily_loss_limit_pct:.1f}% "
                f"| flat and done until tomorrow"
            )
            self.flatten_reason = "kill switch: daily loss limit"

        # A confirmed reversal closes everything before anything else
        if self.flatten_reason:
            reason, self.flatten_reason = self.flatten_reason, None
            if self.legs:
                self.log(f"FLAT | {reason}")
                self.order_ctx = dict(kind="flat", reason=reason)
                self.order = self.close()
                return

        # ------------------------------------------------------------
        # Manage open legs: break-even first, then stops and targets
        # ------------------------------------------------------------
        for leg in self.legs:
            if leg["direction"] == 1:
                favorable = k["h"] - leg["entry_price"]
            else:
                favorable = leg["entry_price"] - k["l"]

            if (
                not leg["at_breakeven"]
                and favorable >= leg["stop_distance"] * self.p.breakeven_trigger_r
            ):
                if leg["direction"] == 1:
                    leg["stop_price"] = max(leg["stop_price"], leg["entry_price"])
                else:
                    leg["stop_price"] = min(leg["stop_price"], leg["entry_price"])
                leg["at_breakeven"] = True
                self.log(f"BREAK-EVEN | Leg stop moved to entry {leg['entry_price']:.2f}")

        trade_loss_cap = equity * (self.p.trade_loss_limit_pct / 100.0)
        for leg in self.legs:
            if leg["direction"] == 1:
                hit_stop = k["l"] <= leg["stop_price"]
                hit_tp = k["h"] >= leg["tp_price"]
            else:
                hit_stop = k["h"] >= leg["stop_price"]
                hit_tp = k["l"] <= leg["tp_price"]

            # Hard cap: one leg may never lose more than 1% of equity,
            # even if a gap or slippage carried price past its stop
            open_loss = (
                (leg["entry_price"] - k["c"]) * leg["direction"]
                * leg["size"] * self._contract_size()
            )
            hit_cap = open_loss >= trade_loss_cap

            if hit_stop or hit_tp or hit_cap:
                if hit_stop:
                    reason = "stop"
                elif hit_tp:
                    reason = "take-profit"
                else:
                    reason = f"trade loss cap {self.p.trade_loss_limit_pct:.1f}%"
                self.order_ctx = dict(kind="close", leg=leg, reason=reason)
                self.order = (
                    self.sell(size=leg["size"]) if leg["direction"] == 1
                    else self.buy(size=leg["size"])
                )
                return

        # ------------------------------------------------------------
        # Entries
        # ------------------------------------------------------------
        if self.kill_switch:
            return  # done for the day
        if collecting:
            return  # never trade the first candles, let the open settle
        if self.day_type is None:
            return
        if not self.p.trade_all_hours and mins > self.p.last_entry_min:
            return

        # Pyramiding gate: add only in the same direction, only once
        # every earlier leg is risk-free at break-even, up to the cap
        required_direction = None
        if self.legs:
            if len(self.legs) >= self.p.max_concurrent_legs:
                return
            if not all(leg["at_breakeven"] for leg in self.legs):
                return
            required_direction = self.legs[0]["direction"]

        ema_fast = float(self.ema_fast[0])
        ema_slow = float(self.ema_slow[0])
        touch = self.p.ema_touch_atr * atr_value
        pulled_into_emas_from_above = (
            min(k["l"], float(self.data.low[-1])) <= max(ema_fast, ema_slow) + touch
        )
        pulled_into_emas_from_below = (
            max(k["h"], float(self.data.high[-1])) >= min(ema_fast, ema_slow) - touch
        )

        signal = None
        choppy = self._is_choppy(atr_value)

        # Sideways / choppy market: stand aside completely
        if choppy and self.p.chop_blocks_all:
            return

        if self.day_type == "bull":
            if self.p.enable_surprise_bar and self._is_surprise_bull(k, atr_value):
                signal = dict(direction=1, rr=self.p.rr_counter, setup="surprise-bar")
            elif (
                ema_fast > ema_slow
                and pulled_into_emas_from_above
                and self._is_bull_signal(k, atr_value)
            ):
                if choppy:
                    self.log("SIGNAL BLOCKED | choppy market (avoiding zone)")
                else:
                    signal = dict(direction=1, rr=self.p.rr_trend, setup="ema-pullback")
            elif self._is_bull_signal(k, atr_value):
                signal = self._double_bottom_signal(k, atr_value)

        elif self.day_type == "bear":
            if self.p.enable_surprise_bar and self._is_surprise_bear(k, atr_value):
                signal = dict(direction=-1, rr=self.p.rr_counter, setup="surprise-bar")
            elif (
                ema_fast < ema_slow
                and pulled_into_emas_from_below
                and self._is_bear_signal(k, atr_value)
            ):
                if choppy:
                    self.log("SIGNAL BLOCKED | choppy market (avoiding zone)")
                else:
                    signal = dict(direction=-1, rr=self.p.rr_trend, setup="ema-pullback")
            elif self._is_bear_signal(k, atr_value):
                signal = self._double_top_signal(k, atr_value)

        elif self.day_type == "range":
            day_range = (self.day_high or 0) - (self.day_low or 0)
            if day_range >= self.p.min_range_atr * atr_value:
                lower_third = self.day_low + day_range * self.p.range_third
                upper_third = self.day_high - day_range * self.p.range_third

                if k["c"] <= lower_third and self._is_bull_signal(k, atr_value):
                    signal = dict(direction=1, rr=self.p.rr_counter, setup="range-buy-low")
                elif k["c"] >= upper_third and self._is_bear_signal(k, atr_value):
                    signal = dict(direction=-1, rr=self.p.rr_counter, setup="range-sell-high")

            # A tested double bottom/top works anywhere inside the range
            if signal is None:
                if self._is_bull_signal(k, atr_value):
                    signal = self._double_bottom_signal(k, atr_value)
                elif self._is_bear_signal(k, atr_value):
                    signal = self._double_top_signal(k, atr_value)

        if signal is None:
            return
        if required_direction is not None and signal["direction"] != required_direction:
            return

        # VWAP filter: buy only above VWAP, sell only below it
        if self.p.use_vwap_filter and self.vwap is not None:
            if signal["direction"] == 1 and k["c"] <= self.vwap:
                self.log(
                    f"SIGNAL BLOCKED | VWAP filter | {signal['setup']} buy "
                    f"C:{k['c']:.2f} <= VWAP:{self.vwap:.2f}"
                )
                return
            if signal["direction"] == -1 and k["c"] >= self.vwap:
                self.log(
                    f"SIGNAL BLOCKED | VWAP filter | {signal['setup']} sell "
                    f"C:{k['c']:.2f} >= VWAP:{self.vwap:.2f}"
                )
                return

        pad = self._tick_size() * self.p.stop_pad_ticks
        if signal["direction"] == 1:
            stop_price = min(k["l"], self._recent_swing_low()) - pad
            stop_distance = k["c"] - stop_price
        else:
            stop_price = max(k["h"], self._recent_swing_high()) + pad
            stop_distance = stop_price - k["c"]

        if stop_distance <= 0:
            return

        risk_pct = self.p.risk_per_trade_pct
        if self.legs:
            risk_pct *= self.p.pyramid_risk_factor

        size = self._calc_size(k["c"], stop_price, risk_pct)
        if size <= 0:
            self.log("SIGNAL BLOCKED | Size too small")
            return

        signal["stop_distance"] = stop_distance
        self.order_ctx = dict(kind="entry", signal=signal)

        self.log(
            f"{'LONG' if signal['direction'] == 1 else 'SHORT'} SIGNAL | {signal['setup']} | "
            f"Day:{self.day_type} | Leg:{len(self.legs) + 1} | C:{k['c']:.2f} Stop:{stop_price:.2f} "
            f"ATR:{atr_value:.2f} Size:{size:.4f}"
        )
        self.order = self.buy(size=size) if signal["direction"] == 1 else self.sell(size=size)
