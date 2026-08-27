"""
Feature Router
==============
Receives the unified 4-input battery reading and derives all extended
feature vectors required by each specialist sub-model.

Minimum required inputs:
    voltage      (V)
    current      (A)  — positive = charging, negative = discharging
    temperature  (°C)
    cycle_number (int)

All other features are estimated/derived from these 4 signals.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Physical constants and estimation parameters
# ─────────────────────────────────────────────────────────────────────────────

NOMINAL_CAPACITY_AH = 2.0          # Standard Li-ion 18650 nominal (Ah)
NOMINAL_SOH_AT_CYCLE_0 = 1.00
SOH_FADE_RATE_PER_CYCLE = 0.00020  # ~20% fade over 1000 cycles

V_NOMINAL = 3.7                    # Nominal Li-ion voltage (V)
V_FULL = 4.2                       # Full charge voltage
V_CUTOFF = 2.5                     # Discharge cutoff

CURRENT_SCALE_TO_MA = 1000.0       # A → mA for Oxford model

# SOH → Capacity estimation:  Cap = SOH × Nominal
# SOH from cycle: simple linear fade model (only used for feature engineering)


class FeatureRouter:
    """
    Derives and routes feature vectors for each specialist sub-model.

    All derivations are physics-based estimates — they are used only to allow
    the specialist models (which expect richer feature sets) to run when
    only the 4 basic signals are available.
    """

    def __init__(self, nominal_capacity_ah: float = NOMINAL_CAPACITY_AH):
        self.nominal_capacity_ah = nominal_capacity_ah

    # ─────────────────────────────────────────────────────────────────────────
    # Internal estimators
    # ─────────────────────────────────────────────────────────────────────────

    def _estimate_soh_ratio(self, cycle_number: int) -> float:
        """
        Simple physics-based SOH estimate from cycle count.
        Used only for feature derivation — NOT the final SOH output.
        Implements a bi-linear fade model: rapid early fade + slow long fade.
        """
        if cycle_number <= 0:
            return 1.0
        # Bi-linear degradation model
        if cycle_number <= 500:
            soh = 1.0 - 0.00020 * cycle_number
        else:
            soh = 0.90 - 0.00015 * (cycle_number - 500)
        return float(np.clip(soh, 0.60, 1.0))

    def _estimate_capacity(self, cycle_number: int) -> float:
        """Estimated discharge capacity (Ah) from cycle number."""
        soh_est = self._estimate_soh_ratio(cycle_number)
        return soh_est * self.nominal_capacity_ah

    def _estimate_voltage_stats(self, voltage: float, cycle_number: int) -> dict:
        """
        Estimate cycle-level voltage statistics from instantaneous voltage.
        Uses physics-based offsets around the measured voltage.
        """
        # As battery ages, voltage range compresses slightly
        soh_est = self._estimate_soh_ratio(cycle_number)
        compression = (1.0 - soh_est) * 0.05  # up to 50mV compression at EOL

        mean_v = voltage
        min_v = max(V_CUTOFF, voltage - 0.35 - compression)
        max_v = min(V_FULL, voltage + 0.25 + compression)
        voltage_std = max(0.02, 0.08 - compression * 0.5)
        return {
            "mean_voltage": float(mean_v),
            "min_voltage": float(min_v),
            "max_voltage": float(max_v),
            "voltage_std": float(voltage_std),
        }

    def _estimate_capacity_fade(self, cycle_number: int) -> dict:
        """
        Estimate capacity fade statistics used by NASA RUL model.
        """
        cap = self._estimate_capacity(cycle_number)
        cap_rolling_mean = cap
        cap_rolling_var = max(0.0, 0.001 * cycle_number / 1000.0)
        cap_trend = -SOH_FADE_RATE_PER_CYCLE * self.nominal_capacity_ah
        return {
            "cap": float(cap),
            "cap_rolling_mean": float(cap_rolling_mean),
            "cap_rolling_var": float(cap_rolling_var),
            "cap_trend": float(cap_trend),
            "nominal_cap": float(self.nominal_capacity_ah),
        }

    def _estimate_time_s(self, current: float, capacity: float) -> float:
        """
        Estimate time into current charge/discharge cycle.
        t = C / I (hours) → seconds. Capped at 4 hours.
        """
        if abs(current) < 1e-6:
            return 0.0
        hours = min(4.0, capacity / max(abs(current), 0.01))
        return float(hours * 3600.0)

    def _estimate_charging_stats(self, temperature: float, cycle_number: int) -> dict:
        """
        Estimate Oxford SOH model charging statistics.
        """
        temp_rise = max(0.5, 1.0 + 0.005 * cycle_number)  # rises with aging
        max_ch_temp = temperature + temp_rise
        # Time to reach 4V decreases as battery ages (charges faster when degraded)
        time_to_4v = max(600.0, 2500.0 - 1.5 * cycle_number)
        return {
            "avg_ch_temp": float(temperature),
            "max_ch_temp": float(max_ch_temp),
            "temp_rise": float(temp_rise),
            "time_to_4v": float(time_to_4v),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Public API: build feature dicts for each specialist model
    # ─────────────────────────────────────────────────────────────────────────

    def for_calce_soc(self, voltage: float, current: float,
                      temperature: float, cycle_number: int) -> dict:
        """
        CALCE SOC Model features:
            ['voltage_V', 'current_A', 'temperature_C', 'time_s', 'cycle_number']
        """
        cap = self._estimate_capacity(cycle_number)
        time_s = self._estimate_time_s(current, cap)
        return {
            "voltage_V": float(voltage),
            "current_A": float(current),
            "temperature_C": float(temperature),
            "time_s": float(time_s),
            "cycle_number": float(cycle_number),
        }

    def for_calce_soh(self, voltage: float, current: float,
                      temperature: float, cycle_number: int) -> dict:
        """
        CALCE SOH Model features:
            ['cycle_number', 'mean_voltage', 'min_voltage', 'max_voltage',
             'voltage_std', 'mean_current', 'mean_temperature']
        """
        v_stats = self._estimate_voltage_stats(voltage, cycle_number)
        return {
            "cycle_number": float(cycle_number),
            "mean_voltage": v_stats["mean_voltage"],
            "min_voltage": v_stats["min_voltage"],
            "max_voltage": v_stats["max_voltage"],
            "voltage_std": v_stats["voltage_std"],
            "mean_current": float(current),
            "mean_temperature": float(temperature),
        }

    def for_calce_anomaly(self, voltage: float, current: float,
                          temperature: float, cycle_number: int) -> dict:
        """
        CALCE Anomaly Detector features:
            ['voltage_V', 'current_A', 'temperature_C', 'cycle_number']
        """
        return {
            "voltage_V": float(voltage),
            "current_A": float(current),
            "temperature_C": float(temperature),
            "cycle_number": float(cycle_number),
        }

    def for_oxford_soc(self, voltage: float, current: float,
                       temperature: float, cycle_number: int) -> dict:
        """
        Oxford SOC Model features:
            ['voltage_v', 'current_ma', 'temp_c', 'power_mw',
             'dv_dt', 'di_dt', 'dt_dt', 'v_roll_mean',
             'v_roll_std', 'i_roll_mean', 't_roll_mean']
        Oxford model uses milliamps (mA) and milliwatts (mW).
        """
        current_ma = current * CURRENT_SCALE_TO_MA
        power_mw = voltage * current_ma
        return {
            "voltage_v": float(voltage),
            "current_ma": float(current_ma),
            "temp_c": float(temperature),
            "power_mw": float(power_mw),
            "dv_dt": 0.0,   # Not available from single reading
            "di_dt": 0.0,
            "dt_dt": 0.0,
            "v_roll_mean": float(voltage),
            "v_roll_std": 0.0,
            "i_roll_mean": float(current_ma),
            "t_roll_mean": float(temperature),
        }

    def for_oxford_soh(self, voltage: float, current: float,
                       temperature: float, cycle_number: int) -> dict:
        """
        Oxford SOH Model features:
            ['cycle_num', 'avg_ch_temp', 'max_ch_temp', 'temp_rise', 'time_to_4v']
        """
        ch_stats = self._estimate_charging_stats(temperature, cycle_number)
        return {
            "cycle_num": float(cycle_number),
            **ch_stats,
        }

    def for_oxford_validator(self, voltage: float, current: float,
                             temperature: float) -> dict:
        """
        Oxford Sensor Validator inputs (scalar values, not dict).
        Returns kwargs for validator.validate_sample().
        """
        current_ma = current * CURRENT_SCALE_TO_MA
        return {
            "voltage_v": float(voltage),
            "current_ma": float(current_ma),
            "temp_c": float(temperature),
            "dv_dt": 0.0,
            "dt_dt": 0.0,
        }

    def for_nasa_soh(self, voltage: float, current: float,
                     temperature: float, cycle_number: int) -> list:
        """
        NASA SOH Model feature vector:
            ['cycle_idx', 'cap', 'cap_rolling_mean', 'cap_rolling_var',
             'cap_trend', 'nominal_cap']
        Returns numpy array (shape 1×6).
        """
        fade = self._estimate_capacity_fade(cycle_number)
        feature_names = ['cycle_idx', 'cap', 'cap_rolling_mean',
                         'cap_rolling_var', 'cap_trend', 'nominal_cap']
        values = [
            float(cycle_number),
            fade["cap"],
            fade["cap_rolling_mean"],
            fade["cap_rolling_var"],
            fade["cap_trend"],
            fade["nominal_cap"],
        ]
        return np.array([values], dtype=np.float32)

    def for_nasa_rul(self, voltage: float, current: float,
                     temperature: float, cycle_number: int) -> np.ndarray:
        """
        NASA RUL Model uses cell-level features (22 features).
        Since we only have current reading (not full cell history),
        we construct a feature vector using physics-based estimates.
        Feature order from train_nasa.py extract_cell_features():
            cap_initial, cap_cycle5, cap_cycle10, cap_max, cap_variance,
            total_cycles, cap_fade_slope, cap_fade_intercept, cap_fade_r2,
            cap_drop_early, dqdv_min, dqdv_var, dqdv_skew, dqdv_kurtosis,
            dqdv_mean, volt_mean_early, volt_min_early, volt_var_early,
            temp_mean, temp_max, temp_var, nominal_capacity
        """
        soh = self._estimate_soh_ratio(cycle_number)
        cap_now = soh * self.nominal_capacity_ah
        cap_initial = self.nominal_capacity_ah
        cap_cycle5 = self.nominal_capacity_ah * self._estimate_soh_ratio(5)
        cap_cycle10 = self.nominal_capacity_ah * self._estimate_soh_ratio(10)
        cap_max = cap_initial
        cap_variance = max(0.0, 0.001 * cycle_number / 1000.0)
        fade_slope = -SOH_FADE_RATE_PER_CYCLE * self.nominal_capacity_ah
        fade_intercept = cap_initial
        fade_r2 = 0.95
        cap_drop_early = cap_initial - cap_cycle10

        # dQ/dV features: zero/default when no full cycle data available
        dqdv_min, dqdv_var, dqdv_skew, dqdv_kurtosis, dqdv_mean = (
            -8.0, -10.0, -0.1, 0.5, -0.002
        )

        v_stats = self._estimate_voltage_stats(voltage, cycle_number)
        volt_mean_early = v_stats["mean_voltage"]
        volt_min_early = v_stats["min_voltage"]
        volt_var_early = v_stats["voltage_std"] ** 2

        temp_mean = float(temperature)
        temp_max = float(temperature + 3.0)
        temp_var = 2.0

        values = [
            cap_initial, cap_cycle5, cap_cycle10, cap_max, cap_variance,
            float(cycle_number), fade_slope, fade_intercept, fade_r2,
            cap_drop_early, dqdv_min, dqdv_var, dqdv_skew,
            dqdv_kurtosis, dqdv_mean,
            volt_mean_early, volt_min_early, volt_var_early,
            temp_mean, temp_max, temp_var,
            self.nominal_capacity_ah,
        ]
        return np.array([values], dtype=np.float32)
