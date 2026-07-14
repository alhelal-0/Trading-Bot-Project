import math
import backtrader as bt


class OneMinuteTrendScalper(bt.Strategy):
    params = dict(
        # Core indicators
        fast_period=9,
        mid_period=21,
        trend_period=200,
        rsi_period=14,
        atr_period=14,

        # Entry filters
        rsi_long_level=50.0,
        rsi_short_level=50.0,
        body_ratio_min=0.45,   # candle body must be a decent part of the range

        # Risk management
        risk_per_trade_pct=0.5,
        rr_ratio=1.5,
        atr_stop_buffer=0.8,   # minimum stop distance based on ATR
        atr_trail_mult=1.0,
        breakeven_trigger_mult=0.8,

        # Position sizing
        min_size=0.01,
        size_step=0.01,

        # Trade management
        swing_lookback=3,
        max_hold_bars=80,

        # Logging
        verbose=True,
    )

    def __init__(self):
        # Indicators
        self.ema_fast = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.p.fast_period)
        self.ema_mid = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.p.mid_period)
        self.ema_trend = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.p.trend_period)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.atr = bt.indicators.AverageTrueRange(self.data, period=self.p.atr_period)

        # Optional crossover helper
        self.cross_fast_mid = bt.indicators.CrossOver(self.ema_fast, self.ema_mid)

        # Order / trade state
        self.order = None
        self.pending_signal = None

        self.entry_price = None
        self.entry_stop_distance = None
        self.stop_price = None
        self.take_profit_price = None
        self.entry_direction = 0
        self.entry_bar = None

        self.highest_price = None
        self.lowest_price = None

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

    def _body_ratio(self):
        candle_range = float(self.data.high[0] - self.data.low[0])
        if candle_range <= 0:
            return 0.0
        body = abs(float(self.data.close[0] - self.data.open[0]))
        return body / candle_range

    def _recent_swing_low(self):
        lookback = int(self.p.swing_lookback)
        lows = [float(self.data.low[-i]) for i in range(1, lookback + 1)]
        return min(lows)

    def _recent_swing_high(self):
        lookback = int(self.p.swing_lookback)
        highs = [float(self.data.high[-i]) for i in range(1, lookback + 1)]
        return max(highs)

    def _calc_size(self, entry_price, stop_price):
        equity = float(self.broker.getvalue())
        risk_cash = equity * (self.p.risk_per_trade_pct / 100.0)

        stop_distance = abs(float(entry_price) - float(stop_price))
        stop_distance = max(stop_distance, self._tick_size())

        contract_size = self._contract_size()
        cash_risk_per_1_lot = stop_distance * contract_size

        if cash_risk_per_1_lot <= 0:
            return float(self.p.min_size)

        raw_size = risk_cash / cash_risk_per_1_lot
        rounded = self._round_down_to_step(raw_size, self._lot_step())

        if rounded < self.p.min_size:
            return float(self.p.min_size)

        return float(rounded)

    def _reset_trade_state(self):
        self.pending_signal = None
        self.entry_price = None
        self.entry_stop_distance = None
        self.stop_price = None
        self.take_profit_price = None
        self.entry_direction = 0
        self.entry_bar = None
        self.highest_price = None
        self.lowest_price = None

    def _enough_bars(self):
        needed = max(
            self.p.fast_period,
            self.p.mid_period,
            self.p.trend_period,
            self.p.rsi_period,
            self.p.atr_period,
            self.p.swing_lookback + 2,
        )
        return len(self.data) > needed

    # ------------------------------------------------------------
    # Backtrader callbacks
    # ------------------------------------------------------------
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if self.order is not None and order.ref == self.order.ref:
            if order.status == order.Completed:
                fill_price = float(order.executed.price)
                fill_size = float(order.executed.size)
                direction = self.pending_signal["direction"] if self.pending_signal else 0

                self.log(
                    f"ORDER FILLED | Price: {fill_price:.5f} | Size: {fill_size:.5f} | "
                    f"Direction: {'LONG' if direction == 1 else 'SHORT' if direction == -1 else 'N/A'}"
                )

                if direction == 1:
                    self.entry_direction = 1
                    self.entry_price = fill_price
                    self.entry_stop_distance = float(self.pending_signal["stop_distance"])

                    self.stop_price = self.entry_price - self.entry_stop_distance
                    self.take_profit_price = self.entry_price + (
                        self.entry_stop_distance * self.p.rr_ratio
                    )
                    self.highest_price = self.entry_price
                    self.entry_bar = len(self.data)

                    self.log(
                        f"LONG OPEN | Entry: {self.entry_price:.5f} | Stop: {self.stop_price:.5f} | "
                        f"TP: {self.take_profit_price:.5f} | RiskDist: {self.entry_stop_distance:.5f}"
                    )

                elif direction == -1:
                    self.entry_direction = -1
                    self.entry_price = fill_price
                    self.entry_stop_distance = float(self.pending_signal["stop_distance"])

                    self.stop_price = self.entry_price + self.entry_stop_distance
                    self.take_profit_price = self.entry_price - (
                        self.entry_stop_distance * self.p.rr_ratio
                    )
                    self.lowest_price = self.entry_price
                    self.entry_bar = len(self.data)

                    self.log(
                        f"SHORT OPEN | Entry: {self.entry_price:.5f} | Stop: {self.stop_price:.5f} | "
                        f"TP: {self.take_profit_price:.5f} | RiskDist: {self.entry_stop_distance:.5f}"
                    )

                self.pending_signal = None

            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                self.log(f"ORDER FAILED | Status: {order.getstatusname()}")
                self.pending_signal = None

            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(
                f"TRADE CLOSED | Gross PnL: {trade.pnl:.2f} | Net PnL: {trade.pnlcomm:.2f}"
            )
            self._reset_trade_state()

    def next(self):
        if not self._enough_bars():
            return

        if self.order is not None:
            return

        # Current candle values
        o = float(self.data.open[0])
        h = float(self.data.high[0])
        l = float(self.data.low[0])
        c = float(self.data.close[0])

        ema_fast = float(self.ema_fast[0])
        ema_mid = float(self.ema_mid[0])
        ema_trend = float(self.ema_trend[0])
        rsi_value = float(self.rsi[0])
        atr_value = float(self.atr[0])
        body_ratio = self._body_ratio()

        bullish_candle = c > o and body_ratio >= self.p.body_ratio_min
        bearish_candle = c < o and body_ratio >= self.p.body_ratio_min

        # Trend filter
        trend_up = c > ema_trend and ema_fast > ema_mid
        trend_down = c < ema_trend and ema_fast < ema_mid

        # Simple continuation trigger:
        # - price must cross/reclaim the fast EMA
        # - trend must agree
        # - candle must show momentum
        # - RSI gives a light confirmation
        long_trigger = (
            trend_up
            and bullish_candle
            and self.data.close[-1] <= self.ema_fast[-1]
            and c > ema_fast
            and rsi_value >= self.p.rsi_long_level
        )

        short_trigger = (
            trend_down
            and bearish_candle
            and self.data.close[-1] >= self.ema_fast[-1]
            and c < ema_fast
            and rsi_value <= self.p.rsi_short_level
        )

        # ------------------------------------------------------------
        # Manage open long
        # ------------------------------------------------------------
        if self.position.size > 0 and self.entry_direction == 1:
            self.highest_price = max(self.highest_price, h)

            # Structural stop from recent swing low
            swing_stop = self._recent_swing_low() - self._tick_size() * 2
            atr_stop = self.entry_price - (atr_value * self.p.atr_stop_buffer)

            # Use the tighter of swing or ATR, but never loosen an existing stop
            candidate_stop = max(self.stop_price, min(swing_stop, atr_stop))

            # Trail once price moves in our favor
            trail_stop = self.highest_price - (atr_value * self.p.atr_trail_mult)
            candidate_stop = max(candidate_stop, trail_stop)

            # Break-even after enough profit
            breakeven_trigger = self.entry_price + (
                self.entry_stop_distance * self.p.breakeven_trigger_mult
            )
            if self.highest_price >= breakeven_trigger:
                candidate_stop = max(candidate_stop, self.entry_price)

            if candidate_stop > self.stop_price:
                self.stop_price = candidate_stop
                self.log(f"LONG STOP UPDATE | New Stop: {self.stop_price:.5f}")

            # Exit checks
            if l <= self.stop_price:
                self.log(
                    f"LONG STOP HIT | Low: {l:.5f} <= Stop: {self.stop_price:.5f}"
                )
                self.order = self.close()
                return

            if h >= self.take_profit_price:
                self.log(
                    f"LONG TP HIT | High: {h:.5f} >= TP: {self.take_profit_price:.5f}"
                )
                self.order = self.close()
                return

            if self.entry_bar is not None and (len(self.data) - self.entry_bar) >= self.p.max_hold_bars:
                self.log("LONG TIME EXIT | Max hold bars reached")
                self.order = self.close()
                return

            return

        # ------------------------------------------------------------
        # Manage open short
        # ------------------------------------------------------------
        if self.position.size < 0 and self.entry_direction == -1:
            self.lowest_price = min(self.lowest_price, l)

            swing_stop = self._recent_swing_high() + self._tick_size() * 2
            atr_stop = self.entry_price + (atr_value * self.p.atr_stop_buffer)

            candidate_stop = min(self.stop_price, max(swing_stop, atr_stop))

            trail_stop = self.lowest_price + (atr_value * self.p.atr_trail_mult)
            candidate_stop = min(candidate_stop, trail_stop)

            breakeven_trigger = self.entry_price - (
                self.entry_stop_distance * self.p.breakeven_trigger_mult
            )
            if self.lowest_price <= breakeven_trigger:
                candidate_stop = min(candidate_stop, self.entry_price)

            if candidate_stop < self.stop_price:
                self.stop_price = candidate_stop
                self.log(f"SHORT STOP UPDATE | New Stop: {self.stop_price:.5f}")

            if h >= self.stop_price:
                self.log(
                    f"SHORT STOP HIT | High: {h:.5f} >= Stop: {self.stop_price:.5f}"
                )
                self.order = self.close()
                return

            if l <= self.take_profit_price:
                self.log(
                    f"SHORT TP HIT | Low: {l:.5f} <= TP: {self.take_profit_price:.5f}"
                )
                self.order = self.close()
                return

            if self.entry_bar is not None and (len(self.data) - self.entry_bar) >= self.p.max_hold_bars:
                self.log("SHORT TIME EXIT | Max hold bars reached")
                self.order = self.close()
                return

            return

        # ------------------------------------------------------------
        # Entry logic
        # ------------------------------------------------------------
        if self.position.size != 0:
            return

        if long_trigger:
            # Structure-based stop below recent swing low, with ATR fallback if needed
            structure_stop = self._recent_swing_low() - self._tick_size() * 2
            atr_stop = c - (atr_value * self.p.atr_stop_buffer)
            stop_price = min(structure_stop, atr_stop)

            # Avoid ultra-tight stops
            min_stop_distance = atr_value * 0.6
            if (c - stop_price) < min_stop_distance:
                stop_price = c - min_stop_distance

            size = self._calc_size(c, stop_price)

            if size <= 0:
                self.log("LONG SIGNAL BLOCKED | Size too small")
                return

            self.pending_signal = dict(
                direction=1,
                stop_distance=abs(c - stop_price),
            )

            self.log(
                f"LONG SIGNAL | O:{o:.5f} H:{h:.5f} L:{l:.5f} C:{c:.5f} | "
                f"EMA9:{ema_fast:.5f} EMA21:{ema_mid:.5f} EMA200:{ema_trend:.5f} | "
                f"RSI:{rsi_value:.2f} ATR:{atr_value:.5f} BodyRatio:{body_ratio:.2f} | "
                f"Stop:{stop_price:.5f} Size:{size:.5f}"
            )
            self.order = self.buy(size=size)
            return

        if short_trigger:
            structure_stop = self._recent_swing_high() + self._tick_size() * 2
            atr_stop = c + (atr_value * self.p.atr_stop_buffer)
            stop_price = max(structure_stop, atr_stop)

            min_stop_distance = atr_value * 0.6
            if (stop_price - c) < min_stop_distance:
                stop_price = c + min_stop_distance

            size = self._calc_size(c, stop_price)

            if size <= 0:
                self.log("SHORT SIGNAL BLOCKED | Size too small")
                return

            self.pending_signal = dict(
                direction=-1,
                stop_distance=abs(stop_price - c),
            )

            self.log(
                f"SHORT SIGNAL | O:{o:.5f} H:{h:.5f} L:{l:.5f} C:{c:.5f} | "
                f"EMA9:{ema_fast:.5f} EMA21:{ema_mid:.5f} EMA200:{ema_trend:.5f} | "
                f"RSI:{rsi_value:.2f} ATR:{atr_value:.5f} BodyRatio:{body_ratio:.2f} | "
                f"Stop:{stop_price:.5f} Size:{size:.5f}"
            )
            self.order = self.sell(size=size)
            return

        # Helpful debug when market is quiet
        if self.p.verbose and (self.cross_fast_mid[0] != 0):
            self.log(
                f"SIGNAL BLOCKED | TrendUp:{trend_up} TrendDown:{trend_down} | "
                f"RSI:{rsi_value:.2f} | BodyRatio:{body_ratio:.2f}"
            )
