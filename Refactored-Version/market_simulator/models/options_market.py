import time
from market_simulator.models.options import bs_price

STRIKES = list(range(1020, 1080, 5))   # $10 spacing: 1000, 1010, ..., 1100
EXPIRY_DURATIONS = [120, 300, 600]           # seconds: 2-min and 5-min, 10-min
CONTRACT_SIZE = 100                     # shares per contract
MM_SPREAD_PCT = 0.05                    # MM quotes ±2.5% around BS mid
SIGMA = 0.20                            # fixed 20% annualized vol
RISK_FREE = 0.05                        # 5% risk-free rate
# Simulation time scale: 2 real hours = 1 simulation year.
# This gives meaningful time value even near expiry (a 2-min expiry
# behaves like ~6 calendar days, so OTM options are worth cents to dollars).
SECS_PER_YEAR = 315360 # in 10x speed of reality


def _spy_price():
    """Lazily import to avoid circular imports at module load time."""
    from market_simulator.utils.market_utils import markets
    return float(markets["SPY"].last_price)


class OptionsMarket:
    def __init__(self):
        self._next_id = 0
        self.expiries = {}        # id -> {duration, start, label}
        self.user_positions = {}  # (strike, expiry_id, option_type) -> signed qty
        self.mm_net_delta = 0.0

        for dur in EXPIRY_DURATIONS:
            self._add_expiry(dur)

    # ── internal helpers ──────────────────────────────────────────────────

    def _add_expiry(self, duration_secs):
        eid = self._next_id
        self._next_id += 1
        label = f"{duration_secs // 60}-min"
        self.expiries[eid] = {
            "duration": duration_secs,
            "start": time.time(),
            "label": label,
        }
        return eid

    def _time_remaining(self, expiry):
        elapsed = time.time() - expiry["start"]
        return max(0.0, expiry["duration"] - elapsed)

    def _get_chain_row(self, strike, expiry):
        S = _spy_price()
        T = self._time_remaining(expiry) / SECS_PER_YEAR
        c_mid, c_delta, c_gamma, c_theta, c_vega = bs_price(S, strike, T, RISK_FREE, SIGMA, 'call')
        p_mid, p_delta, p_gamma, p_theta, p_vega = bs_price(S, strike, T, RISK_FREE, SIGMA, 'put')

        half = MM_SPREAD_PCT / 2
        return {
            "strike": strike,
            "atm": abs(S - strike) < (STRIKES[1] - STRIKES[0]) / 2,
            "call": {
                "bid":   round(c_mid * (1 - half), 4),
                "ask":   round(c_mid * (1 + half), 4),
                "mid":   round(c_mid, 4),
                "delta": round(c_delta, 4),
                "gamma": round(c_gamma, 6),
                "theta": round(c_theta, 6),
                "vega":  round(c_vega, 4),
            },
            "put": {
                "bid":   round(p_mid * (1 - half), 4),
                "ask":   round(p_mid * (1 + half), 4),
                "mid":   round(p_mid, 4),
                "delta": round(p_delta, 4),
                "gamma": round(p_gamma, 6),
                "theta": round(p_theta, 6),
                "vega":  round(p_vega, 4),
            },
        }

    def _recalculate_mm_delta(self):
        S = _spy_price()
        total = 0.0
        for (strike, eid, opt_type), qty in self.user_positions.items():
            if eid not in self.expiries or qty == 0:
                continue
            T = self._time_remaining(self.expiries[eid]) / SECS_PER_YEAR
            _, delta, _, _, _ = bs_price(S, strike, T, RISK_FREE, SIGMA, opt_type)
            total += delta * qty * CONTRACT_SIZE
        self.mm_net_delta = total

    # ── public API ────────────────────────────────────────────────────────

    def get_chain(self):
        result = []
        for eid, expiry in self.expiries.items():
            rows = [self._get_chain_row(k, expiry) for k in STRIKES]
            result.append({
                "expiry_id":      eid,
                "label":          expiry["label"],
                "time_remaining": round(self._time_remaining(expiry), 1),
                "rows":           rows,
            })
        return result

    def place_option_order(self, strike, expiry_id, option_type, qty, direction):
        """
        Returns (premium, error).
        premium = fill_price * qty * CONTRACT_SIZE (always positive).
        direction: 'buy' or 'sell'.
        """
        if strike not in STRIKES:
            return 0, "Invalid strike"
        expiry_id = int(expiry_id)
        if expiry_id not in self.expiries:
            return 0, "Expiry not found"
        if option_type not in ('call', 'put'):
            return 0, "option_type must be 'call' or 'put'"
        if qty <= 0:
            return 0, "qty must be positive"

        expiry = self.expiries[expiry_id]
        T = self._time_remaining(expiry) / SECS_PER_YEAR
        S = _spy_price()
        mid, delta, _, _, _ = bs_price(S, strike, T, RISK_FREE, SIGMA, option_type)

        half = MM_SPREAD_PCT / 2
        if direction == 'buy':
            fill_price = mid * (1 + half)   # user pays the ask
            signed_qty = qty
        else:
            fill_price = mid * (1 - half)   # user receives the bid
            signed_qty = -qty

        fill_price = max(fill_price, 0.0001)
        premium = round(fill_price * qty * CONTRACT_SIZE, 4)

        key = (strike, expiry_id, option_type)
        self.user_positions[key] = self.user_positions.get(key, 0) + signed_qty

        # Update MM delta incrementally
        self.mm_net_delta += delta * signed_qty * CONTRACT_SIZE

        return premium, None

    def settle_expired(self, cash_callback):
        """
        Check all expiries. For expired ones: compute intrinsic payout,
        call cash_callback(net_cash, label, spy_price), remove positions,
        create a new rolling expiry of the same duration.
        """
        S = _spy_price()
        expired_ids = [eid for eid, exp in self.expiries.items()
                       if self._time_remaining(exp) <= 0]

        for eid in expired_ids:
            expiry = self.expiries[eid]
            label = expiry["label"]
            net_cash = 0.0

            # Settle user positions in this expiry
            keys_to_remove = [k for k in self.user_positions if k[1] == eid]
            for key in keys_to_remove:
                strike, _, opt_type = key
                qty = self.user_positions.pop(key)
                if opt_type == 'call':
                    intrinsic = max(S - strike, 0)
                else:
                    intrinsic = max(strike - S, 0)
                net_cash += intrinsic * qty * CONTRACT_SIZE

            cash_callback(net_cash, label, S)

            # Create rolling replacement expiry
            self._add_expiry(expiry["duration"])
            del self.expiries[eid]

        if expired_ids:
            self._recalculate_mm_delta()

    def get_user_positions(self):
        """Return serializable list with current BS value and delta for each open position."""
        S = _spy_price()
        result = []
        for (strike, eid, opt_type), qty in self.user_positions.items():
            if qty == 0 or eid not in self.expiries:
                continue
            T = self._time_remaining(self.expiries[eid]) / SECS_PER_YEAR
            price, delta, gamma, theta, vega = bs_price(S, strike, T, RISK_FREE, SIGMA, opt_type)
            half = MM_SPREAD_PCT / 2
            mid_value = price * abs(qty) * CONTRACT_SIZE
            result.append({
                "strike":     strike,
                "expiry_id":  eid,
                "expiry_label": self.expiries[eid]["label"],
                "type":       opt_type,
                "qty":        qty,
                "bs_value":   round(mid_value, 2),
                "delta":      round(delta * qty * CONTRACT_SIZE, 2),
                "gamma":      round(gamma, 6),
                "theta":      round(theta, 6),
            })
        return result


options_market = OptionsMarket()
