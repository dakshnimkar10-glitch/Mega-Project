"""
Meta-Learner Fusion Layer
==========================
Collects predictions from all specialist sub-models and fuses them
into calibrated final predictions using Ridge Regression meta-learners.

Architecture:
    [CALCE_SOC, Oxford_SOC]                → SOC Meta-Learner → soc_final
    [CALCE_SOH, Oxford_SOH, NASA_SOH]     → SOH Meta-Learner → soh_final
    [CALCE_RUL, NASA_RUL]                 → RUL Meta-Learner → rul_final
    [CALCE_Anomaly, Oxford_Validator, ...]→ Risk Engine       → risk_final

The meta-learners are trained on synthetic calibration samples drawn
from the physical operating envelope of Li-ion batteries. No raw
battery dataset is required for this training step.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# Calibration data generator for meta-learner training
# ─────────────────────────────────────────────────────────────────────────────

class CalibrationDataGenerator:
    """
    Generates synthetic (features, targets) pairs to train meta-learners.
    Uses physics-based SOC/SOH/RUL profiles from electrochemistry literature.
    """

    def __init__(self, n_samples: int = 5000, seed: int = 42):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def generate_soc_calibration(self) -> tuple:
        """
        Generate (specialist_predictions, true_soc) pairs.
        True SOC follows a voltage-based open-circuit curve.
        """
        # Sample voltages uniformly in operating range
        voltages = self.rng.uniform(2.7, 4.2, self.n_samples)
        cycles = self.rng.integers(0, 800, self.n_samples).astype(float)
        noise = self.rng.normal(0, 1.5, self.n_samples)

        # Physics-based SOC from voltage (simplified OCV curve for Li-ion NMC)
        # OCV ≈ (V - 2.7) / (4.2 - 2.7) × 100
        true_soc = np.clip((voltages - 2.7) / 1.5 * 100.0 + noise, 0.0, 100.0)

        # Simulate specialist model prediction errors
        # CALCE SOC (RandomForest) — good at medium SOC, noisy at extremes
        calce_soc = true_soc + self.rng.normal(0, 2.5, self.n_samples)
        calce_soc = np.clip(calce_soc, 0.0, 100.0)

        # Oxford SOC (XGBoost) — more accurate with dynamic features
        oxford_soc = true_soc + self.rng.normal(0, 1.8, self.n_samples)
        oxford_soc = np.clip(oxford_soc, 0.0, 100.0)

        X = np.column_stack([calce_soc, oxford_soc])
        y = true_soc
        return X, y

    def generate_soh_calibration(self) -> tuple:
        """
        Generate (specialist_predictions, true_soh) pairs.
        True SOH follows a bi-linear degradation curve.
        """
        cycles = self.rng.uniform(0, 1000, self.n_samples)

        # Bi-linear SOH degradation model
        true_soh = np.where(
            cycles <= 500,
            100.0 - 0.020 * cycles,
            90.0 - 0.015 * (cycles - 500)
        )
        true_soh = np.clip(true_soh, 60.0, 100.0)
        noise = self.rng.normal(0, 1.0, self.n_samples)
        true_soh += noise

        # Simulate specialist model predictions
        calce_soh = true_soh + self.rng.normal(0, 2.0, self.n_samples)
        oxford_soh = true_soh + self.rng.normal(0, 1.5, self.n_samples)
        nasa_soh = true_soh / 100.0  # NASA outputs ratio (0–1)
        nasa_soh_pct = nasa_soh * 100.0 + self.rng.normal(0, 1.8, self.n_samples)

        X = np.column_stack([
            np.clip(calce_soh, 0, 100),
            np.clip(oxford_soh, 0, 100),
            np.clip(nasa_soh_pct, 0, 100),
        ])
        y = np.clip(true_soh, 0.0, 100.0)
        return X, y

    def generate_rul_calibration(self) -> tuple:
        """
        Generate (specialist_predictions, true_rul) pairs.
        True RUL is cycles until SOH hits 80% (EOL threshold).
        """
        cycles = self.rng.uniform(0, 800, self.n_samples)
        eol_cycle = self.rng.uniform(800, 1200, self.n_samples)
        true_rul = np.maximum(0.0, eol_cycle - cycles)

        # Simulate specialist predictions (XGBoost models have some variance)
        calce_rul = true_rul + self.rng.normal(0, 15.0, self.n_samples)
        nasa_rul = true_rul + self.rng.normal(0, 12.0, self.n_samples)

        X = np.column_stack([
            np.maximum(0.0, calce_rul),
            np.maximum(0.0, nasa_rul),
        ])
        y = np.maximum(0.0, true_rul)
        return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Meta-Learner
# ─────────────────────────────────────────────────────────────────────────────

class MetaLearnerLayer:
    """
    Ridge regression meta-learners for SOC, SOH, and RUL fusion.

    Each meta-learner takes the predictions of all specialist models
    for a given output head and produces a single calibrated prediction.
    """

    # Relative dataset weights for each specialist model
    # Higher = more trusted. Based on dataset size, algorithm quality.
    SOC_WEIGHTS = {
        "calce": 0.42,   # CALCE: RF, 5 features
        "oxford": 0.58,  # Oxford: XGBoost, richer features
    }
    SOH_WEIGHTS = {
        "calce": 0.30,   # CALCE: GradientBoosting
        "oxford": 0.35,  # Oxford: XGBoost
        "nasa": 0.35,    # NASA: XGBoost, per-cycle + capacity features
    }
    RUL_WEIGHTS = {
        "calce": 0.45,   # CALCE: XGBoost
        "nasa": 0.55,    # NASA: XGBoost, richer cell-level features
    }

    def __init__(self):
        self.soc_scaler = StandardScaler()
        self.soh_scaler = StandardScaler()
        self.rul_scaler = StandardScaler()

        self.soc_meta = Ridge(alpha=0.1)
        self.soh_meta = Ridge(alpha=0.1)
        self.rul_meta = Ridge(alpha=0.5)

        self._is_trained = False

    def train(self, n_samples: int = 8000, seed: int = 42):
        """
        Train all meta-learners using synthetic calibration data.
        No raw battery data needed — uses physics-based simulation.
        """
        gen = CalibrationDataGenerator(n_samples=n_samples, seed=seed)

        # ── SOC meta-learner ──────────────────────────────────────────
        X_soc, y_soc = gen.generate_soc_calibration()
        X_soc_scaled = self.soc_scaler.fit_transform(X_soc)
        self.soc_meta.fit(X_soc_scaled, y_soc)

        # ── SOH meta-learner ──────────────────────────────────────────
        X_soh, y_soh = gen.generate_soh_calibration()
        X_soh_scaled = self.soh_scaler.fit_transform(X_soh)
        self.soh_meta.fit(X_soh_scaled, y_soh)

        # ── RUL meta-learner ──────────────────────────────────────────
        X_rul, y_rul = gen.generate_rul_calibration()
        X_rul_scaled = self.rul_scaler.fit_transform(X_rul)
        self.rul_meta.fit(X_rul_scaled, y_rul)

        self._is_trained = True
        return self

    def fuse_soc(self, calce_soc: float, oxford_soc: float) -> float:
        """
        Fuse SOC predictions from CALCE and Oxford specialists.
        Returns calibrated SOC in percent (0–100).
        """
        if not self._is_trained:
            raise RuntimeError("MetaLearnerLayer must be trained before calling fuse_soc()")
        X = np.array([[calce_soc, oxford_soc]])
        X_scaled = self.soc_scaler.transform(X)
        pred = self.soc_meta.predict(X_scaled)[0]
        return float(np.clip(pred, 0.0, 100.0))

    def fuse_soh(self, calce_soh: float, oxford_soh: float, nasa_soh: float) -> float:
        """
        Fuse SOH predictions from CALCE, Oxford, and NASA specialists.
        Returns calibrated SOH in percent (0–100).
        """
        if not self._is_trained:
            raise RuntimeError("MetaLearnerLayer must be trained before calling fuse_soh()")
        # NASA model outputs ratio (0-1), convert to percent
        nasa_soh_pct = nasa_soh * 100.0 if nasa_soh <= 1.0 else nasa_soh
        X = np.array([[calce_soh, oxford_soh, nasa_soh_pct]])
        X_scaled = self.soh_scaler.transform(X)
        pred = self.soh_meta.predict(X_scaled)[0]
        return float(np.clip(pred, 0.0, 100.0))

    def fuse_rul(self, calce_rul: float, nasa_rul: float) -> float:
        """
        Fuse RUL predictions from CALCE and NASA specialists.
        Returns calibrated RUL in cycles (≥ 0).
        """
        if not self._is_trained:
            raise RuntimeError("MetaLearnerLayer must be trained before calling fuse_rul()")
        X = np.array([[calce_rul, nasa_rul]])
        X_scaled = self.rul_scaler.transform(X)
        pred = self.rul_meta.predict(X_scaled)[0]
        return float(max(0.0, pred))

    def weighted_average(self, predictions: dict, weights: dict) -> float:
        """
        Fallback: simple weighted average when meta-learner is unavailable.
        """
        total = 0.0
        total_weight = 0.0
        for key, pred in predictions.items():
            w = weights.get(key, 1.0)
            total += w * pred
            total_weight += w
        return total / total_weight if total_weight > 0 else 0.0

    def compute_confidence(
        self,
        soc_preds: dict,
        soh_preds: dict,
        rul_preds: dict,
    ) -> dict:
        """
        Compute per-output confidence scores based on specialist agreement.
        Higher disagreement → lower confidence.
        """
        def disagreement_to_confidence(values: list, scale: float) -> float:
            """
            Convert standard deviation of specialist predictions to confidence %.
            scale: the expected full range of the output (e.g., 100 for SOC/SOH)
            """
            std = float(np.std(values))
            # Confidence = 1 - normalized_std, clipped to [0, 1]
            confidence = 1.0 - min(1.0, std / (scale * 0.15))
            return round(confidence * 100.0, 1)

        soc_values = list(soc_preds.values())
        soh_values = list(soh_preds.values())
        rul_values = list(rul_preds.values())

        return {
            "soc_confidence": disagreement_to_confidence(soc_values, 100.0),
            "soh_confidence": disagreement_to_confidence(soh_values, 100.0),
            "rul_confidence": disagreement_to_confidence(rul_values, 500.0),
        }
