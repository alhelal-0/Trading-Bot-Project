# Bots on TradeLocker are implemented in Python using the Backtrader framework.
# You can find more examples and the complete documentation at https://www.backtrader.com/

# Bollinger Band mean reversion (long-only):
#   - BUY when the close is below the lower Bollinger Band AND RSI is
#     below the oversold threshold.
#   - EXIT on the FIRST of: price reaching the middle Bollinger Band,
#     a profit target of `target_atr_mult` x ATR, or a stop loss
#     `stop_atr_mult` x ATR below entry.
#   - Position size risks `risk_per_trade_pct` of account equity against
#     the ATR stop.

import math

import backtrader as bt


class BollingerMeanReversionStrategy(bt.Strategy):
    params = {
        "bb_period": 20,
        "bb_dev": 2.0,
        "rsi_period": 14,
        "rsi_oversold": 30.0,
        "atr_period": 14,
        "stop_atr_mult": 1.0,
        "target_atr_mult": 2.0,
        "risk_per_trade_pct": 1.0,
        "min_size": 0.01,
        "verbose": False,
    }

    def __init__(self) -> None:
        self.bb = bt.indicators.BollingerBands(
            self.data.close, period=self.p.bb_period, devfactor=self.p.bb_dev)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)

        self.order = None
        self.order_ctx = None
        self.trade_meta = None
        self.stop_price = None
        self.target_price = None

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
        needed = max(self.p.bb_period, self.p.rsi_period, self.p.atr_period) + 1
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
                stop_distance = ctx["stop_distance"]
                target_distance = ctx["target_distance"]
                size = abs(float(order.executed.size))

                self.trade_meta = dict(
                    entry_dt=self.data.datetime.datetime(0),
                    entry_price=fill_price,
                    size=size,
                    risk_cash=stop_distance * size * self._contract_size(),
                )
                self.stop_price = fill_price - stop_distance
                self.target_price = fill_price + target_distance

                self.log(
                    f"LONG OPEN | Entry:{fill_price:.5f} Stop:{self.stop_price:.5f} "
                    f"Target:{self.target_price:.5f} Size:{size:.4f}"
                )

            elif ctx and ctx["kind"] == "exit" and self.trade_meta is not None:
                meta = self.trade_meta
                pnl = (fill_price - meta["entry_price"]) * meta["size"] * self._contract_size()
                risk = meta["risk_cash"]
                self.trade_records.append(dict(
                    setup="bb-mean-reversion",
                    day_type="-",
                    direction="long",
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
        if self.order is not None:
            return

        c = float(self.data.close[0])
        h = float(self.data.high[0])
        l = float(self.data.low[0])
        atr_value = float(self.atr[0])

        # Manage the open position: stop first, then targets
        if self.position.size > 0 and self.trade_meta is not None:
            if l <= self.stop_price:
                self.order_ctx = dict(kind="exit", reason="stop")
                self.order = self.close()
            elif h >= self.target_price:
                self.order_ctx = dict(kind="exit", reason="atr-target")
                self.order = self.close()
            elif h >= float(self.bb.mid[0]):
                self.order_ctx = dict(kind="exit", reason="middle-band")
                self.order = self.close()
            return

        # Entry: close below lower band + oversold RSI
        if atr_value <= 0:
            return

        if c < float(self.bb.bot[0]) and float(self.rsi[0]) < self.p.rsi_oversold:
            stop_distance = self.p.stop_atr_mult * atr_value
            target_distance = self.p.target_atr_mult * atr_value

            size = self._calc_size(stop_distance)
            if size <= 0:
                return

            self.order_ctx = dict(
                kind="entry",
                stop_distance=stop_distance,
                target_distance=target_distance,
            )
            self.log(
                f"LONG SIGNAL | C:{c:.5f} < LowerBB:{float(self.bb.bot[0]):.5f} "
                f"RSI:{float(self.rsi[0]):.1f} Size:{size:.4f}"
            )
            self.order = self.buy(size=size)

    # ====================================================================================
    # params_metadata is optional -- it is used to configure the "Run Backtest/Bot" modal.
    params_metadata = {
        "bb_period": {
            "label": "Bollinger period",
            "helper_text": "Period of the Bollinger Bands",
            "value_type": "int",
        },
        "bb_dev": {
            "label": "Bollinger std dev",
            "helper_text": "Standard deviation multiplier of the bands",
            "value_type": "float",
        },
        "rsi_period": {
            "label": "RSI period",
            "helper_text": "Period of the RSI filter",
            "value_type": "int",
        },
        "rsi_oversold": {
            "label": "RSI oversold level",
            "helper_text": "Buy only when RSI is below this level",
            "value_type": "float",
        },
        "stop_atr_mult": {
            "label": "Stop (ATRs)",
            "helper_text": "Stop loss distance below entry, in ATRs",
            "value_type": "float",
        },
        "target_atr_mult": {
            "label": "Target (ATRs)",
            "helper_text": "Profit target above entry, in ATRs",
            "value_type": "float",
        },
        "risk_per_trade_pct": {
            "label": "Risk per trade %",
            "helper_text": "Percent of account equity risked on each trade",
            "value_type": "float",
        },
    }
