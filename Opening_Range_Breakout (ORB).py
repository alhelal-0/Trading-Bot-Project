# Bots on TradeLocker are implemented in Python using the Backtrader framework.
# You can find more examples and the complete documentation at https://www.backtrader.com/

# Opening Range Breakout (ORB):
#   - Build the opening range from the first `or_minutes` (15) after the
#     US market open (`session_open_min`, in the FEED's timezone --
#     adjust if your feed is not in exchange time).
#   - Enter LONG when a bar CLOSES above the opening range high with
#     above-average volume; SHORT below the range low with above-average
#     volume. One attempt per direction per day.
#   - Stop loss: the opposite side of the range.
#   - Take profit: `rr_ratio` (2) x the initial risk.
#   - Position size risks `risk_per_trade_pct` (1%) of account equity.
#   - Everything is flattened before the session close.

import math

import backtrader as bt


class OpeningRangeBreakoutStrategy(bt.Strategy):
    params = {
        "or_minutes": 15,                  # opening range length in minutes
        "session_open_min": 9 * 60 + 30,   # 09:30 in feed time
        "last_entry_min": 15 * 60,         # no new breakout entries after 15:00
        "flat_min": 15 * 60 + 55,          # close everything by 15:55
        "vol_period": 20,                  # average volume lookback
        "use_volume_filter": True,
        "rr_ratio": 2.0,                   # take profit at this multiple of risk
        "risk_per_trade_pct": 1.0,
        "allow_long": True,
        "allow_short": True,
        "min_size": 0.01,
        "verbose": False,
    }

    def __init__(self) -> None:
        self.vol_sma = bt.indicators.SimpleMovingAverage(
            self.data.volume, period=self.p.vol_period)

        self.order = None
        self.order_ctx = None
        self.trade_meta = None
        self.stop_price = None
        self.target_price = None

        # Day state
        self.cur_date = None
        self.or_high = None
        self.or_low = None
        self.or_done = False
        self.tried_long = False
        self.tried_short = False

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

    def _volume_ok(self):
        if not self.p.use_volume_filter:
            return True
        avg = float(self.vol_sma[0])
        if avg <= 0:
            return True  # feed has no volume data
        return float(self.data.volume[0]) > avg

    def _reset_day(self):
        self.or_high = None
        self.or_low = None
        self.or_done = False
        self.tried_long = False
        self.tried_short = False

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
                self.target_price = (
                    fill_price + stop_distance * self.p.rr_ratio if direction == 1
                    else fill_price - stop_distance * self.p.rr_ratio
                )

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
                    setup="orb-breakout",
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
            self._reset_day()

        if self.order is not None:
            return

        mins = self._minutes_now()
        h = float(self.data.high[0])
        l = float(self.data.low[0])
        c = float(self.data.close[0])

        # Flatten into the close
        if mins >= self.p.flat_min:
            if self.position.size != 0:
                self.order_ctx = dict(kind="exit", reason="session-close")
                self.order = self.close()
            return

        # ------------------------------------------------------------
        # Build the opening range from the first `or_minutes` of the session
        # ------------------------------------------------------------
        or_end = self.p.session_open_min + self.p.or_minutes
        if self.p.session_open_min <= mins < or_end:
            self.or_high = h if self.or_high is None else max(self.or_high, h)
            self.or_low = l if self.or_low is None else min(self.or_low, l)
            return  # never trade inside the opening range window
        if mins >= or_end and self.or_high is not None:
            self.or_done = True

        # ------------------------------------------------------------
        # Manage the open position
        # ------------------------------------------------------------
        if self.position.size != 0 and self.trade_meta is not None:
            if self.position.size > 0:
                if l <= self.stop_price:
                    self.order_ctx = dict(kind="exit", reason="range-stop")
                elif h >= self.target_price:
                    self.order_ctx = dict(kind="exit", reason="rr-target")
            else:
                if h >= self.stop_price:
                    self.order_ctx = dict(kind="exit", reason="range-stop")
                elif l <= self.target_price:
                    self.order_ctx = dict(kind="exit", reason="rr-target")
            if self.order_ctx:
                self.order = self.close()
            return

        # ------------------------------------------------------------
        # Entries: close beyond the range with above-average volume
        # ------------------------------------------------------------
        if not self.or_done:
            return
        if mins > self.p.last_entry_min:
            return

        if (
            self.p.allow_long
            and not self.tried_long
            and c > self.or_high
            and self._volume_ok()
        ):
            self.tried_long = True
            stop_distance = c - self.or_low
            size = self._calc_size(stop_distance)
            if size <= 0:
                return
            self.order_ctx = dict(kind="entry", direction=1, stop_price=self.or_low)
            self.log(
                f"LONG BREAKOUT | C:{c:.5f} > ORH:{self.or_high:.5f} "
                f"Vol:{float(self.data.volume[0]):.0f} Size:{size:.4f}"
            )
            self.order = self.buy(size=size)

        elif (
            self.p.allow_short
            and not self.tried_short
            and c < self.or_low
            and self._volume_ok()
        ):
            self.tried_short = True
            stop_distance = self.or_high - c
            size = self._calc_size(stop_distance)
            if size <= 0:
                return
            self.order_ctx = dict(kind="entry", direction=-1, stop_price=self.or_high)
            self.log(
                f"SHORT BREAKOUT | C:{c:.5f} < ORL:{self.or_low:.5f} "
                f"Vol:{float(self.data.volume[0]):.0f} Size:{size:.4f}"
            )
            self.order = self.sell(size=size)

    # ====================================================================================
    # params_metadata is optional -- it is used to configure the "Run Backtest/Bot" modal.
    params_metadata = {
        "or_minutes": {
            "label": "Opening range (min)",
            "helper_text": "Minutes after the open used to build the range",
            "value_type": "int",
        },
        "session_open_min": {
            "label": "Session open (min)",
            "helper_text": "Market open, minutes from midnight in the feed's timezone",
            "value_type": "int",
        },
        "last_entry_min": {
            "label": "Last entry (min)",
            "helper_text": "No new breakout entries after this time",
            "value_type": "int",
        },
        "flat_min": {
            "label": "Flat by (min)",
            "helper_text": "Close everything after this many minutes from midnight",
            "value_type": "int",
        },
        "rr_ratio": {
            "label": "Reward:risk",
            "helper_text": "Take profit at this multiple of the initial risk",
            "value_type": "float",
        },
        "risk_per_trade_pct": {
            "label": "Risk per trade %",
            "helper_text": "Percent of account equity risked on each trade",
            "value_type": "float",
        },
        "vol_period": {
            "label": "Volume average period",
            "helper_text": "Lookback for the average-volume filter",
            "value_type": "int",
        },
        "use_volume_filter": {
            "label": "Volume filter",
            "helper_text": "Require above-average volume on the breakout candle",
            "value_type": "bool",
        },
    }
