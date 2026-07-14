# Bots on TradeLocker are implemented in Python using the Backtrader framework.
# You can find more examples and the complete documentation at https://www.backtrader.com/

# VWAP pullback strategy:
#   - Trades only during regular US market hours (session params are in
#     minutes from midnight, in the FEED's timezone -- adjust if your
#     feed is not in exchange time); flat before the close.
#   - LONG only when price is above the intraday VWAP, pulled back to
#     within `pullback_pct` of VWAP in the last few bars, and the current
#     bar is a bullish confirmation candle. SHORT is the mirror below VWAP.
#   - Risk `risk_per_trade_pct` of equity against a stop beyond the
#     pullback extreme.
#   - Exit at `rr_ratio`:1 reward-to-risk, on the stop, or the moment
#     price closes across VWAP against the position.

import math

import backtrader as bt


class IntradayVWAP(bt.Indicator):
    """Volume-weighted average price, resetting at the start of each day.

    Falls back to equal weighting (TWAP) when the feed has no volume."""

    lines = ("vwap",)
    plotinfo = dict(subplot=False)

    def __init__(self):
        self._cur_date = None
        self._cum_pv = 0.0
        self._cum_vol = 0.0

    def next(self):
        d = self.data.datetime.date(0)
        if d != self._cur_date:
            self._cur_date = d
            self._cum_pv = 0.0
            self._cum_vol = 0.0

        typical = (self.data.high[0] + self.data.low[0] + self.data.close[0]) / 3.0
        vol = float(self.data.volume[0])
        if vol <= 0:
            vol = 1.0

        self._cum_pv += typical * vol
        self._cum_vol += vol
        self.lines.vwap[0] = self._cum_pv / self._cum_vol


class VwapPullbackStrategy(bt.Strategy):
    params = {
        "pullback_pct": 0.10,        # pullback must come within this % of VWAP
        "pullback_lookback": 3,      # bars in which the touch may have happened
        "confirm_body_min": 0.50,    # confirmation candle body vs range
        "rr_ratio": 2.0,             # take profit at this multiple of risk
        "min_stop_pct": 0.05,        # stop never tighter than this % of price
        "stop_pad_pct": 0.02,        # pad beyond the pullback extreme
        "risk_per_trade_pct": 1.0,
        "allow_long": True,
        "allow_short": True,
        "session_open_min": 9 * 60 + 30,   # 09:30
        "last_entry_min": 15 * 60 + 30,    # 15:30
        "flat_min": 15 * 60 + 55,          # 15:55
        "vwap_warmup_bars": 3,
        "min_size": 0.01,
        "verbose": False,
    }

    def __init__(self) -> None:
        self.vwap = IntradayVWAP(self.data)

        self.order = None
        self.order_ctx = None
        self.trade_meta = None
        self.stop_price = None
        self.target_price = None

        self.cur_date = None
        self.bars_today = 0

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

    def _minutes_now(self):
        t = self.data.datetime.time(0)
        return t.hour * 60 + t.minute

    def _body_ratio(self, o, h, l, c):
        rng = h - l
        if rng <= 0:
            return 0.0
        return abs(c - o) / rng

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
                    setup="vwap-pullback",
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
        d = self.data.datetime.date(0)
        if d != self.cur_date:
            self.cur_date = d
            self.bars_today = 0
        self.bars_today += 1

        if self.order is not None:
            return

        mins = self._minutes_now()
        o = float(self.data.open[0])
        h = float(self.data.high[0])
        l = float(self.data.low[0])
        c = float(self.data.close[0])
        vwap = float(self.vwap[0])

        # Regular hours only: flatten into the close
        if mins >= self.p.flat_min:
            if self.position.size != 0:
                self.order_ctx = dict(kind="exit", reason="session-close")
                self.order = self.close()
            return

        # Manage the open position
        if self.position.size != 0 and self.trade_meta is not None:
            if self.position.size > 0:
                if l <= self.stop_price:
                    self.order_ctx = dict(kind="exit", reason="stop")
                elif h >= self.target_price:
                    self.order_ctx = dict(kind="exit", reason="rr-target")
                elif c < vwap:
                    self.order_ctx = dict(kind="exit", reason="vwap-cross")
            else:
                if h >= self.stop_price:
                    self.order_ctx = dict(kind="exit", reason="stop")
                elif l <= self.target_price:
                    self.order_ctx = dict(kind="exit", reason="rr-target")
                elif c > vwap:
                    self.order_ctx = dict(kind="exit", reason="vwap-cross")
            if self.order_ctx:
                self.order = self.close()
            return

        # Entries: RTH window, warmed-up VWAP only
        if mins < self.p.session_open_min or mins > self.p.last_entry_min:
            return
        if self.bars_today < self.p.vwap_warmup_bars or vwap <= 0:
            return

        lookback = min(self.p.pullback_lookback, self.bars_today)
        body = self._body_ratio(o, h, l, c)

        direction = 0
        stop_distance = 0.0

        if self.p.allow_long and c > vwap:
            touch = vwap * (1.0 + self.p.pullback_pct / 100.0)
            pulled = any(float(self.data.low[-i]) <= touch for i in range(lookback))
            bullish = c > o and body >= self.p.confirm_body_min
            if pulled and bullish:
                direction = 1
                pull_low = min(float(self.data.low[-i]) for i in range(lookback))
                stop_price = pull_low * (1.0 - self.p.stop_pad_pct / 100.0)
                stop_distance = max(c - stop_price, c * self.p.min_stop_pct / 100.0)

        if direction == 0 and self.p.allow_short and c < vwap:
            touch = vwap * (1.0 - self.p.pullback_pct / 100.0)
            pulled = any(float(self.data.high[-i]) >= touch for i in range(lookback))
            bearish = c < o and body >= self.p.confirm_body_min
            if pulled and bearish:
                direction = -1
                pull_high = max(float(self.data.high[-i]) for i in range(lookback))
                stop_price = pull_high * (1.0 + self.p.stop_pad_pct / 100.0)
                stop_distance = max(stop_price - c, c * self.p.min_stop_pct / 100.0)

        if direction == 0:
            return

        size = self._calc_size(stop_distance)
        if size <= 0:
            return

        self.order_ctx = dict(kind="entry", direction=direction, stop_distance=stop_distance)
        self.log(
            f"{'LONG' if direction == 1 else 'SHORT'} SIGNAL | "
            f"C:{c:.5f} VWAP:{vwap:.5f} Body:{body:.2f} Size:{size:.4f}"
        )
        self.order = self.buy(size=size) if direction == 1 else self.sell(size=size)

    # ====================================================================================
    # params_metadata is optional -- it is used to configure the "Run Backtest/Bot" modal.
    params_metadata = {
        "pullback_pct": {
            "label": "Pullback distance %",
            "helper_text": "Pullback must come within this percent of VWAP",
            "value_type": "float",
        },
        "pullback_lookback": {
            "label": "Pullback lookback",
            "helper_text": "Bars in which the VWAP touch may have happened",
            "value_type": "int",
        },
        "confirm_body_min": {
            "label": "Confirmation body",
            "helper_text": "Minimum body/range ratio of the confirmation candle",
            "value_type": "float",
        },
        "rr_ratio": {
            "label": "Reward:risk",
            "helper_text": "Take profit at this multiple of the risk",
            "value_type": "float",
        },
        "risk_per_trade_pct": {
            "label": "Risk per trade %",
            "helper_text": "Percent of account equity risked on each trade",
            "value_type": "float",
        },
        "session_open_min": {
            "label": "Session open (min)",
            "helper_text": "Session start, minutes from midnight in the feed's timezone",
            "value_type": "int",
        },
        "flat_min": {
            "label": "Flat by (min)",
            "helper_text": "Close everything after this many minutes from midnight",
            "value_type": "int",
        },
    }
