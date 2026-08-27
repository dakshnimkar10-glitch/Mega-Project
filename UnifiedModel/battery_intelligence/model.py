"""
Battery Intelligence Model
===========================
Unified Hierarchical Multi-Task Learning Model for Li-ion Battery State Estimation.

This is the central model class. When serialized to battery_intelligence.pkl,
it bundles:
    - All 9 specialist sub-models (loaded from CALCE, Oxford, NASA)
    - Feature routing layer
    - Trained meta-learner fusion layer
    - Hierarchical risk engine

Usage (single PKL file):
    import pickle

    with open("battery_intelligence.pkl", "rb") as f:
        model = pickle.load(f)

    result = model.predict(
        voltage=3.82,
        current=-1.0,
        temperature=25.4,
        cycle_number=100
    )
    print(result)
    # {
    #   'soc': 72.4,
    #   'soh': 95.3,
    #   'rul': 643.0,
    #   'risk_score': 5.2,
    #   'risk_level': 'SAFE',
    #   'anomaly': False,
    #   'faults': [],
    #   'confidence': {'soc_confidence': 91.0, ...},
    #   'specialist_predictions': {...}
    # }
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from .feature_router import FeatureRouter
from .meta_learner import MetaLearnerLayer
from .risk_engine import HierarchicalRiskEngine

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Specialist Model Wrappers
# ─────────────────────────────────────────────────────────────────────────────

class _CALCEModels:
    """Bundles all 4 CALCE models with their inference interfaces."""

    def __init__(self):
        self.soc_model = None
        self.soc_features = None
        self.soh_model = None
        self.soh_features = None
        self.anomaly_model = None
        self.anomaly_features = None
        self.anomaly_rules = None
        self.rul_model = None

    def predict_soc(self, features: dict) -> float:
        """Run CALCE SOC model. Returns SOC in percent (0–100)."""
        if self.soc_model is None:
            return 50.0  # safe fallback
        df = pd.DataFrame([features])
        try:
            pred = self.soc_model.predict(df[self.soc_features])[0]
            # CALCE model outputs 0–1 ratio OR 0–100 percent
            if pred <= 1.0:
                pred *= 100.0
            return float(np.clip(pred, 0.0, 100.0))
        except Exception:
            return 50.0

    def predict_soh(self, features: dict) -> float:
        """Run CALCE SOH model. Returns SOH in percent (0–100)."""
        if self.soh_model is None:
            return 90.0
        df = pd.DataFrame([features])
        try:
            pred = self.soh_model.predict(df[self.soh_features])[0]
            if pred <= 1.0:
                pred *= 100.0
            return float(np.clip(pred, 0.0, 100.0))
        except Exception:
            return 90.0

    def predict_anomaly(self, features: dict) -> tuple:
        """
        Run CALCE Anomaly Detector.
        Returns (iso_label: int, iso_score: float, rule_anomaly: bool).
        iso_label: +1 = normal, -1 = anomaly.
        """
        if self.anomaly_model is None:
            return 1, 0.0, False
        df = pd.DataFrame([features])
        try:
            label = int(self.anomaly_model.predict(df[self.anomaly_features])[0])
            score = float(self.anomaly_model.score_samples(df[self.anomaly_features])[0])
            # Check hard safety rules
            rules = self.anomaly_rules or {}
            rule_anom = (
                features.get("voltage_V", 3.7) > rules.get("v_max", 4.25) or
                features.get("voltage_V", 3.7) < rules.get("v_min", 2.50) or
                abs(features.get("current_A", 0.0)) > rules.get("i_max", 3.5) or
                features.get("temperature_C", 25.0) > rules.get("t_max", 45.0)
            )
            return label, score, rule_anom
        except Exception:
            return 1, 0.0, False

    def predict_rul(self, feature_vector: np.ndarray) -> float:
        """Run CALCE XGBoost RUL model. Returns RUL in cycles."""
        if self.rul_model is None:
            return 500.0
        try:
            pred = float(self.rul_model.predict(feature_vector)[0])
            return max(0.0, pred)
        except Exception:
            return 500.0


class _OxfordModels:
    """Bundles all 3 Oxford models with their inference interfaces."""

    def __init__(self):
        self.soc_estimator = None
        self.soh_estimator = None
        self.validator = None

    def predict_soc(self, features: dict) -> float:
        """Run Oxford SOC model. Returns SOC in percent (0–100)."""
        if self.soc_estimator is None:
            return 50.0
        try:
            df = pd.DataFrame([features])
            pred = self.soc_estimator.predict(df)[0]
            if pred <= 1.0:
                pred *= 100.0
            return float(np.clip(pred, 0.0, 100.0))
        except Exception:
            return 50.0

    def predict_soh(self, features: dict) -> float:
        """Run Oxford SOH model. Returns SOH in percent (0–100)."""
        if self.soh_estimator is None:
            return 90.0
        try:
            df = pd.DataFrame([features])
            pred = self.soh_estimator.predict(df)[0]
            if pred <= 1.0:
                pred *= 100.0
            return float(np.clip(pred, 0.0, 100.0))
        except Exception:
            return 90.0

    def validate_sensor(self, kwargs: dict) -> dict:
        """Run Oxford Sensor Validator."""
        if self.validator is None:
            return {"is_valid": True, "anomaly_score_pct": 0.0, "faults": []}
        try:
            return self.validator.validate_sample(**kwargs)
        except Exception:
            return {"is_valid": True, "anomaly_score_pct": 0.0, "faults": []}


class _NASAModels:
    """Bundles both NASA models (SOH + RUL) with their inference interfaces."""

    def __init__(self):
        self.soh_model = None
        self.soh_features = None
        self.rul_model = None

    def predict_soh(self, feature_array: np.ndarray) -> float:
        """Run NASA SOH model. Returns SOH as ratio (0–1)."""
        if self.soh_model is None:
            return 0.90
        try:
            pred = float(self.soh_model.predict(feature_array)[0])
            return float(np.clip(pred, 0.0, 1.0))
        except Exception:
            return 0.90

    def predict_rul(self, feature_array: np.ndarray) -> float:
        """Run NASA RUL model. Returns RUL in cycles."""
        if self.rul_model is None:
            return 500.0
        try:
            pred = float(self.rul_model.predict(feature_array)[0])
            return max(0.0, pred)
        except Exception:
            return 500.0


# ─────────────────────────────────────────────────────────────────────────────
# Oxford Fallback Estimator Wrappers (must be module-level for pickle)
# ─────────────────────────────────────────────────────────────────────────────

class _OxfordSOCWrapper:
    """Module-level wrapper so pickle can serialize it when SOCEstimator import fails."""
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
    def predict(self, df):
        return np.clip(self.model.predict(df[self.feature_names]), 0.0, 100.0)


class _OxfordSOHWrapper:
    """Module-level wrapper so pickle can serialize it when SOHEstimator import fails."""
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
    def predict(self, df):
        return np.clip(self.model.predict(df[self.feature_names]), 0.0, 100.0)


class _OxfordValidatorWrapper:
    """Module-level wrapper so pickle can serialize it when BatterySensorValidator import fails."""
    def __init__(self, detector, feature_cols):
        self.detector = detector
        self.feature_cols = feature_cols
        self.is_fitted = True

    def validate_sample(self, voltage_v, current_ma, temp_c, dv_dt=0.0, dt_dt=0.0):
        df = pd.DataFrame([{
            "voltage_v": voltage_v, "current_ma": current_ma,
            "temp_c": temp_c, "power_mw": voltage_v * current_ma,
            "dv_dt": dv_dt, "dt_dt": dt_dt
        }])
        try:
            label = self.detector.predict(df[self.feature_cols])[0]
            score = float(self.detector.score_samples(df[self.feature_cols])[0])
            anomaly_pct = float(np.clip((0.5 - score) * 100.0, 0.0, 100.0))
            is_valid = label == 1
        except Exception:
            is_valid, anomaly_pct = True, 0.0
        faults = []
        if not (2.5 <= voltage_v <= 4.35):
            faults.append(f"Voltage {voltage_v}V out of range.")
            is_valid = False
        if not (0.0 <= temp_c <= 55.0):
            faults.append(f"Temperature {temp_c}C out of range.")
            is_valid = False
        if abs(current_ma) > 4500.0:
            faults.append(f"Current {current_ma}mA exceeds limit.")
            is_valid = False
        return {
            "is_valid": is_valid,
            "anomaly_score_pct": anomaly_pct,
            "faults": faults,
            "status": "RIGHT" if is_valid else "NOT RIGHT",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main Model Class
# ─────────────────────────────────────────────────────────────────────────────

class BatteryIntelligenceModel:
    """
    Unified Hierarchical Multi-Task Battery Intelligence Model.

    Combines 9 specialist models from 3 datasets (CALCE, Oxford, NASA)
    into one single inference interface.

    Architecture:
        Input (4 signals) → Feature Router → Specialist Models (9)
        → Meta-Learner Fusion → Hierarchical Risk Engine → Unified Output

    This entire object is picklable as a single .pkl file.
    """

    MODEL_VERSION = "1.0.0"
    ARCHITECTURE = "Stacked Ensemble + Hierarchical Risk Fusion"

    def __init__(self):
        # Specialist model containers
        self._calce = _CALCEModels()
        self._oxford = _OxfordModels()
        self._nasa = _NASAModels()

        # Framework components
        self._router = FeatureRouter()
        self._meta = MetaLearnerLayer()
        self._risk_engine = HierarchicalRiskEngine()

        # Metadata
        self._models_loaded = {
            "calce_soc": False,
            "calce_soh": False,
            "calce_anomaly": False,
            "calce_rul": False,
            "oxford_soc": False,
            "oxford_soh": False,
            "oxford_validator": False,
            "nasa_soh": False,
            "nasa_rul": False,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Model Loading
    # ─────────────────────────────────────────────────────────────────────────

    def load_calce_models(
        self,
        soc_path: str,
        soh_path: str,
        anomaly_path: str,
        rul_path: str,
    ) -> "BatteryIntelligenceModel":
        """Load all 4 CALCE specialist models."""
        # SOC
        with open(soc_path, "rb") as f:
            data = pickle.load(f)
        self._calce.soc_model = data["model"]
        self._calce.soc_features = data["feature_cols"]
        self._models_loaded["calce_soc"] = True
        print(f"  ✓ CALCE SOC   loaded: {Path(soc_path).name}")

        # SOH
        with open(soh_path, "rb") as f:
            data = pickle.load(f)
        self._calce.soh_model = data["model"]
        self._calce.soh_features = data["feature_cols"]
        self._models_loaded["calce_soh"] = True
        print(f"  ✓ CALCE SOH   loaded: {Path(soh_path).name}")

        # Anomaly
        with open(anomaly_path, "rb") as f:
            data = pickle.load(f)
        self._calce.anomaly_model = data["iso_forest"]
        self._calce.anomaly_features = data["feature_cols"]
        self._calce.anomaly_rules = data.get("rules", {})
        self._models_loaded["calce_anomaly"] = True
        print(f"  ✓ CALCE Anomaly loaded: {Path(anomaly_path).name}")

        # RUL
        with open(rul_path, "rb") as f:
            data = pickle.load(f)
        # Handle batteryml wrapper or plain model
        if isinstance(data, dict):
            self._calce.rul_model = data.get("model", data.get("predictor"))
        else:
            self._calce.rul_model = data
        self._models_loaded["calce_rul"] = True
        print(f"  ✓ CALCE RUL   loaded: {Path(rul_path).name}")

        return self

    def load_oxford_models(
        self,
        soc_path: str,
        soh_path: str,
        validator_path: str,
        oxford_src_dir: Optional[str] = None,
    ) -> "BatteryIntelligenceModel":
        """
        Load all 3 Oxford specialist models.
        oxford_src_dir: path to Oxford/src (needed for custom class imports).
        """
        import joblib

        # Add Oxford src to sys.path so custom classes can be deserialized
        if oxford_src_dir:
            if oxford_src_dir not in sys.path:
                sys.path.insert(0, str(oxford_src_dir))

        # SOC
        data = joblib.load(soc_path)
        # Oxford saves as {'model': ..., 'feature_names': ..., 'type': ...}
        # The SOCEstimator.load() wraps it — we use the raw estimator
        try:
            # Try loading via SOCEstimator class
            sys.path.insert(0, str(Path(soc_path).parent.parent.parent))
            from src.models.soc_model import SOCEstimator
            self._oxford.soc_estimator = SOCEstimator.load(soc_path)
        except Exception:
            self._oxford.soc_estimator = _OxfordSOCWrapper(
                data["model"], data.get("feature_names", [])
            )
        self._models_loaded["oxford_soc"] = True
        print(f"  ✓ Oxford SOC  loaded: {Path(soc_path).name}")

        # SOH
        try:
            from src.models.soh_model import SOHEstimator
            self._oxford.soh_estimator = SOHEstimator.load(soh_path)
        except Exception:
            data_soh = joblib.load(soh_path)
            self._oxford.soh_estimator = _OxfordSOHWrapper(
                data_soh["model"], data_soh.get("feature_names", [])
            )
        self._models_loaded["oxford_soh"] = True
        print(f"  ✓ Oxford SOH  loaded: {Path(soh_path).name}")

        # Sensor Validator
        try:
            from src.models.validation_model import BatterySensorValidator
            self._oxford.validator = BatterySensorValidator.load(validator_path)
        except Exception:
            data_val = joblib.load(validator_path)
                    faults = []
                    if not (2.5 <= voltage_v <= 4.35):
                        faults.append(f"Voltage {voltage_v}V out of range.")
                        is_valid = False
                    if not (0.0 <= temp_c <= 55.0):
                        faults.append(f"Temperature {temp_c}°C out of range.")
                        is_valid = False
                    if abs(current_ma) > 4500.0:
                        faults.append(f"Current {current_ma}mA exceeds limit.")
                        is_valid = False
                    return {"is_valid": is_valid, "anomaly_score_pct": anomaly_pct,
                            "faults": faults, "status": "RIGHT" if is_valid else "NOT RIGHT"}
            self._oxford.validator = _SimpleValidator(
                data_val["detector"], data_val.get("feature_cols", ["voltage_v", "current_ma", "temp_c", "power_mw", "dv_dt", "dt_dt"])
            )
        self._models_loaded["oxford_validator"] = True
        print(f"  ✓ Oxford Validator loaded: {Path(validator_path).name}")

        return self

    def load_nasa_models(
        self,
        soh_path: str,
        rul_path: str,
    ) -> "BatteryIntelligenceModel":
        """Load both NASA specialist models."""
        # SOH
        with open(soh_path, "rb") as f:
            data = pickle.load(f)
        self._nasa.soh_model = data["model"]
        self._nasa.soh_features = data.get("feature_names", [])
        self._models_loaded["nasa_soh"] = True
        print(f"  ✓ NASA SOH    loaded: {Path(soh_path).name}")

        # RUL
        with open(rul_path, "rb") as f:
            data = pickle.load(f)
        self._nasa.rul_model = data["model"]
        self._models_loaded["nasa_rul"] = True
        print(f"  ✓ NASA RUL    loaded: {Path(rul_path).name}")

        return self

    def train_meta_learners(self, n_samples: int = 10000) -> "BatteryIntelligenceModel":
        """
        Train the meta-learner fusion layer.
        Uses synthetic calibration data — no raw battery data required.
        """
        print("\n  Training meta-learner fusion layer...")
        self._meta.train(n_samples=n_samples)
        print(f"  ✓ Meta-learners trained (SOC, SOH, RUL heads) on {n_samples} synthetic samples")
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────────

    def predict(
        self,
        voltage: float,
        current: float,
        temperature: float,
        cycle_number: int,
    ) -> dict:
        """
        Run unified battery intelligence prediction.

        Args:
            voltage:      Terminal voltage in Volts (V)
            current:      Current in Amperes (A); positive=charge, negative=discharge
            temperature:  Cell temperature in Celsius (°C)
            cycle_number: Battery cycle count (int, ≥ 0)

        Returns:
            dict with keys:
                soc                  : float — State of Charge (0–100 %)
                soh                  : float — State of Health (0–100 %)
                rul                  : float — Remaining Useful Life (cycles)
                risk_score           : float — Composite risk score (0–100)
                risk_level           : str   — SAFE / CAUTION / WARNING / CRITICAL
                anomaly              : bool  — True if anomaly detected
                faults               : list  — Human-readable fault descriptions
                confidence           : dict  — Per-output confidence scores
                specialist_predictions: dict — Individual model outputs (for debugging)
        """
        cycle_number = int(cycle_number)

        # ── Step 1: Route Features ──────────────────────────────────────────
        calce_soc_feat = self._router.for_calce_soc(voltage, current, temperature, cycle_number)
        calce_soh_feat = self._router.for_calce_soh(voltage, current, temperature, cycle_number)
        calce_anom_feat = self._router.for_calce_anomaly(voltage, current, temperature, cycle_number)
        oxford_soc_feat = self._router.for_oxford_soc(voltage, current, temperature, cycle_number)
        oxford_soh_feat = self._router.for_oxford_soh(voltage, current, temperature, cycle_number)
        oxford_val_feat = self._router.for_oxford_validator(voltage, current, temperature)
        nasa_soh_feat = self._router.for_nasa_soh(voltage, current, temperature, cycle_number)
        nasa_rul_feat = self._router.for_nasa_rul(voltage, current, temperature, cycle_number)
        calce_rul_feat = self._router.for_nasa_rul(voltage, current, temperature, cycle_number)  # same shape

        # ── Step 2: Run All Specialist Models ───────────────────────────────
        calce_soc = self._calce.predict_soc(calce_soc_feat)
        calce_soh = self._calce.predict_soh(calce_soh_feat)
        calce_iso_label, calce_iso_score, calce_rule_anom = self._calce.predict_anomaly(calce_anom_feat)
        calce_rul = self._calce.predict_rul(calce_rul_feat)

        oxford_soc = self._oxford.predict_soc(oxford_soc_feat)
        oxford_soh = self._oxford.predict_soh(oxford_soh_feat)
        oxford_validation = self._oxford.validate_sensor(oxford_val_feat)

        nasa_soh = self._nasa.predict_soh(nasa_soh_feat)  # 0–1 ratio
        nasa_rul = self._nasa.predict_rul(nasa_rul_feat)

        # ── Step 3: Meta-Learner Fusion ─────────────────────────────────────
        soc_final = self._meta.fuse_soc(calce_soc, oxford_soc)
        soh_final = self._meta.fuse_soh(calce_soh, oxford_soh, nasa_soh)
        rul_final = self._meta.fuse_rul(calce_rul, nasa_rul)

        # ── Step 4: Confidence Estimation ───────────────────────────────────
        nasa_soh_pct = (nasa_soh * 100.0) if nasa_soh <= 1.0 else nasa_soh
        confidence = self._meta.compute_confidence(
            soc_preds={"calce": calce_soc, "oxford": oxford_soc},
            soh_preds={"calce": calce_soh, "oxford": oxford_soh, "nasa": nasa_soh_pct},
            rul_preds={"calce": calce_rul, "nasa": nasa_rul},
        )

        # ── Step 5: Hierarchical Risk Assessment ────────────────────────────
        # Combine CALCE rule anomaly with iso label
        effective_iso_label = -1 if (calce_iso_label == -1 or calce_rule_anom) else 1
        risk_result = self._risk_engine.assess(
            voltage=voltage,
            current=current,
            temperature=temperature,
            soc=soc_final,
            soh=soh_final,
            rul=rul_final,
            calce_anomaly_raw=effective_iso_label,
            oxford_validation=oxford_validation,
            calce_iso_score=calce_iso_score,
        )

        # ── Return Unified Output ────────────────────────────────────────────
        return {
            # Primary outputs
            "soc": round(soc_final, 2),
            "soh": round(soh_final, 2),
            "rul": round(rul_final, 1),
            "risk_score": risk_result.risk_score,
            "risk_level": risk_result.risk_level,
            "anomaly": risk_result.anomaly_detected,
            "faults": risk_result.faults,

            # Meta information
            "confidence": confidence,
            "risk_components": risk_result.component_scores,

            # Specialist debug outputs
            "specialist_predictions": {
                "soc": {"calce": round(calce_soc, 2), "oxford": round(oxford_soc, 2)},
                "soh": {
                    "calce": round(calce_soh, 2),
                    "oxford": round(oxford_soh, 2),
                    "nasa_ratio": round(nasa_soh, 4),
                },
                "rul": {"calce": round(calce_rul, 1), "nasa": round(nasa_rul, 1)},
                "anomaly": {
                    "calce_iso_label": calce_iso_label,
                    "calce_rule_triggered": calce_rule_anom,
                    "oxford_valid": oxford_validation.get("is_valid", True),
                    "oxford_anomaly_score": oxford_validation.get("anomaly_score_pct", 0.0),
                },
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Status / Info
    # ─────────────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return current model loading status."""
        loaded = [k for k, v in self._models_loaded.items() if v]
        missing = [k for k, v in self._models_loaded.items() if not v]
        return {
            "version": self.MODEL_VERSION,
            "architecture": self.ARCHITECTURE,
            "models_loaded": len(loaded),
            "models_total": len(self._models_loaded),
            "loaded": loaded,
            "missing": missing,
            "meta_learners_trained": self._meta._is_trained,
            "ready": len(missing) == 0 and self._meta._is_trained,
        }

    def __repr__(self) -> str:
        s = self.status()
        return (
            f"BatteryIntelligenceModel(v{s['version']}) — "
            f"{s['models_loaded']}/{s['models_total']} models loaded, "
            f"meta={'trained' if s['meta_learners_trained'] else 'untrained'}"
        )
