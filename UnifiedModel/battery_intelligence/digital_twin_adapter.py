"""
Digital Twin Adapter
=====================
Provides a clean interface between a Battery Digital Twin engine and
the BatteryIntelligenceModel.

The Digital Twin generates realistic battery behaviour data at each
simulation timestep. This adapter:
    1. Receives raw digital twin outputs
    2. Adapts the format to BatteryIntelligenceModel.predict()
    3. Returns structured intelligence results
    4. Optionally maintains a sliding window for rolling statistics

Digital Twin → BatteryDigitalTwinAdapter → BatteryIntelligenceModel → Predictions

Usage:
    from battery_intelligence.digital_twin_adapter import BatteryDigitalTwinAdapter

    # Create once with your loaded intelligence model
    adapter = BatteryDigitalTwinAdapter(intelligence_model)

    # Call at each simulation timestep
    for timestep in simulation:
        dt_output = digital_twin.step()  # Digital twin output dict
        result = adapter.process(dt_output)
        print(result)
"""

import numpy as np
from typing import Optional, List
from collections import deque


class BatteryDigitalTwinAdapter:
    """
    Adapter between a Battery Digital Twin and the BatteryIntelligenceModel.

    Maintains a rolling window of recent readings for improved predictions
    using derivative features and rolling statistics.

    Args:
        intelligence_model: A loaded BatteryIntelligenceModel instance
        window_size: Number of recent timesteps to buffer for statistics
    """

    # Expected Digital Twin output keys and their alternatives
    DT_KEY_MAP = {
        "voltage": ["voltage", "voltage_v", "V", "terminal_voltage", "v_terminal"],
        "current": ["current", "current_a", "I", "i_out", "cell_current"],
        "temperature": ["temperature", "temp", "temperature_c", "T", "cell_temp"],
        "cycle_number": ["cycle_number", "cycle", "n_cycle", "cycle_count", "step"],
    }

    def __init__(self, intelligence_model, window_size: int = 20):
        self.model = intelligence_model
        self.window_size = window_size

        # Rolling history buffers
        self._v_history: deque = deque(maxlen=window_size)
        self._i_history: deque = deque(maxlen=window_size)
        self._t_history: deque = deque(maxlen=window_size)
        self._results_history: list = []

        self._timestep = 0
        self._current_cycle = 0

    def process(self, dt_output: dict) -> dict:
        """
        Process one digital twin timestep output.

        Args:
            dt_output: Dictionary from digital twin containing at least
                       voltage, current, and temperature.

        Returns:
            Dictionary with full intelligence model output plus metadata:
                - soc, soh, rul, risk_score, risk_level, anomaly, faults
                - confidence, specialist_predictions
                - timestep, cycle_number
                - rolling_stats (moving averages)
        """
        # ── Parse Digital Twin Output ───────────────────────────────────────
        voltage = self._extract(dt_output, "voltage")
        current = self._extract(dt_output, "current")
        temperature = self._extract(dt_output, "temperature")
        cycle_number = self._extract_int(dt_output, "cycle_number",
                                         default=self._current_cycle)

        self._current_cycle = cycle_number
        self._timestep += 1

        # ── Update rolling window ───────────────────────────────────────────
        self._v_history.append(voltage)
        self._i_history.append(current)
        self._t_history.append(temperature)

        # ── Run Intelligence Model ──────────────────────────────────────────
        result = self.model.predict(
            voltage=voltage,
            current=current,
            temperature=temperature,
            cycle_number=cycle_number,
        )

        # ── Augment with rolling stats ──────────────────────────────────────
        result["rolling_stats"] = self._compute_rolling_stats()
        result["timestep"] = self._timestep
        result["digital_twin_input"] = {
            "voltage": voltage,
            "current": current,
            "temperature": temperature,
            "cycle_number": cycle_number,
        }

        # Store result
        self._results_history.append({
            "timestep": self._timestep,
            "soc": result["soc"],
            "soh": result["soh"],
            "rul": result["rul"],
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
        })

        return result

    def process_batch(self, dt_outputs: List[dict]) -> List[dict]:
        """
        Process a batch of digital twin timestep outputs.
        Useful for post-processing simulation replay data.
        """
        return [self.process(dt_out) for dt_out in dt_outputs]

    def get_trend_summary(self) -> dict:
        """
        Get a summary of battery trends over the recent history.
        Useful for dashboards and maintenance reports.
        """
        if len(self._results_history) < 2:
            return {"status": "insufficient_data", "n_samples": len(self._results_history)}

        recent = self._results_history[-min(50, len(self._results_history)):]
        soc_vals = [r["soc"] for r in recent]
        soh_vals = [r["soh"] for r in recent]
        rul_vals = [r["rul"] for r in recent]
        risk_vals = [r["risk_score"] for r in recent]

        return {
            "n_timesteps": self._timestep,
            "current_cycle": self._current_cycle,
            "soc_trend": {
                "current": soc_vals[-1],
                "mean": round(float(np.mean(soc_vals)), 2),
                "min": round(float(np.min(soc_vals)), 2),
                "max": round(float(np.max(soc_vals)), 2),
                "trend": "decreasing" if soc_vals[-1] < soc_vals[0] else "stable",
            },
            "soh_trend": {
                "current": soh_vals[-1],
                "mean": round(float(np.mean(soh_vals)), 2),
                "degradation_rate": round(
                    (soh_vals[0] - soh_vals[-1]) / max(1, len(soh_vals)), 4
                ),
            },
            "rul_trend": {
                "current": rul_vals[-1],
                "mean": round(float(np.mean(rul_vals)), 2),
            },
            "risk_summary": {
                "current": risk_vals[-1],
                "max_observed": round(float(np.max(risk_vals)), 2),
                "critical_events": sum(1 for r in recent if r["risk_level"] == "CRITICAL"),
                "warning_events": sum(1 for r in recent if r["risk_level"] == "WARNING"),
            },
        }

    def reset(self):
        """Reset adapter state (call when starting a new simulation run)."""
        self._v_history.clear()
        self._i_history.clear()
        self._t_history.clear()
        self._results_history.clear()
        self._timestep = 0
        self._current_cycle = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _extract(self, dt_output: dict, key: str) -> float:
        """Extract a float value from digital twin output, trying all aliases."""
        aliases = self.DT_KEY_MAP.get(key, [key])
        for alias in aliases:
            if alias in dt_output:
                return float(dt_output[alias])
        raise KeyError(
            f"Digital twin output missing required field '{key}'. "
            f"Expected one of: {aliases}. Got keys: {list(dt_output.keys())}"
        )

    def _extract_int(
        self, dt_output: dict, key: str, default: int = 0
    ) -> int:
        """Extract an int value, returning default if not found."""
        aliases = self.DT_KEY_MAP.get(key, [key])
        for alias in aliases:
            if alias in dt_output:
                return int(float(dt_output[alias]))
        return default

    def _compute_rolling_stats(self) -> dict:
        """Compute rolling statistics from the current history window."""
        v_arr = np.array(self._v_history)
        i_arr = np.array(self._i_history)
        t_arr = np.array(self._t_history)

        stats = {}
        for name, arr in [("voltage", v_arr), ("current", i_arr), ("temperature", t_arr)]:
            if len(arr) > 0:
                stats[name] = {
                    "mean": round(float(np.mean(arr)), 4),
                    "std": round(float(np.std(arr)), 4),
                    "min": round(float(np.min(arr)), 4),
                    "max": round(float(np.max(arr)), 4),
                }
                if len(arr) >= 2:
                    stats[name]["derivative"] = round(
                        float(arr[-1] - arr[-2]), 6
                    )
        return stats
