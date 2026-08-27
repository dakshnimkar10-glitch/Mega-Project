"""
Hierarchical Risk Engine
=========================
Computes an integrated Battery Risk Score by combining signals from:

    1. Physical Safety Rules       (hard limits → immediate fault)
    2. CALCE Isolation Forest      (statistical anomaly detection)
    3. Oxford Sensor Validator     (sensor integrity + physics bounds)
    4. SOC Deviation               (very low SOC → deep discharge risk)
    5. SOH Degradation             (low SOH → capacity risk)
    6. RUL Criticality             (low RUL → imminent EOL risk)

The risk score is a weighted composite (0–100):
    0–25   → SAFE
    25–50  → CAUTION
    50–75  → WARNING
    75–100 → CRITICAL

The hierarchical structure ensures:
    - Any hard physical violation immediately → CRITICAL
    - ML anomaly from Isolation Forest elevates risk by +30 points
    - SOC/SOH/RUL deterioration adds proportional risk
    - All signals are additive with saturation at 100
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Physical Safety Thresholds (Li-ion standard)
# ─────────────────────────────────────────────────────────────────────────────

VOLTAGE_MAX = 4.25   # V — absolute max (overcharge)
VOLTAGE_MIN = 2.50   # V — absolute min (deep discharge)
VOLTAGE_WARN_MAX = 4.20
VOLTAGE_WARN_MIN = 2.80

TEMP_MAX = 55.0      # °C — thermal runaway threshold
TEMP_MIN = 0.0       # °C — lithium plating threshold
TEMP_WARN_MAX = 45.0
TEMP_WARN_MIN = 5.0

CURRENT_MAX_A = 3.5  # A — max absolute current
CURRENT_WARN_A = 3.0

SOC_LOW_WARN = 15.0   # % — low SOC warning
SOC_LOW_CRIT = 5.0    # % — critical SOC (deep discharge risk)
SOH_WARN = 80.0       # % — battery nearing EOL
SOH_CRIT = 70.0       # % — battery at EOL
RUL_WARN = 100        # cycles remaining — replace soon
RUL_CRIT = 30         # cycles remaining — replace immediately


@dataclass
class RiskAssessment:
    """Complete risk assessment output from the hierarchical risk engine."""
    risk_score: float           # 0–100 composite risk score
    risk_level: str             # SAFE / CAUTION / WARNING / CRITICAL
    anomaly_detected: bool      # True if any hard anomaly was found
    faults: List[str]           # Human-readable fault descriptions
    component_scores: dict      # Individual risk contributions

    @property
    def is_safe(self) -> bool:
        return self.risk_level == "SAFE"


class HierarchicalRiskEngine:
    """
    Four-layer hierarchical risk assessment engine.

    Layer 1: Physical hard limits  (voltage, current, temperature bounds)
    Layer 2: ML statistical anomaly (CALCE IsolationForest score)
    Layer 3: Sensor integrity       (Oxford validator result)
    Layer 4: State-based risk       (SOC, SOH, RUL thresholds)
    """

    # Risk contribution weights (sum to ~100 when all critical)
    W_PHYSICAL = 40.0     # Hard physical violations
    W_ML_ANOMALY = 25.0   # ML statistical anomaly
    W_SENSOR = 15.0       # Sensor integrity
    W_SOC = 10.0          # SOC-based risk
    W_SOH = 6.0           # SOH-based risk
    W_RUL = 4.0           # RUL-based risk

    def assess(
        self,
        voltage: float,
        current: float,
        temperature: float,
        soc: float,
        soh: float,
        rul: float,
        calce_anomaly_raw: int,          # IsolationForest: 1=normal, -1=anomaly
        oxford_validation: dict,          # From BatterySensorValidator.validate_sample()
        calce_iso_score: float = 0.0,    # Raw IsolationForest score (optional)
    ) -> RiskAssessment:
        """
        Run full hierarchical risk assessment.

        Args:
            voltage:           Terminal voltage (V)
            current:           Current (A)
            temperature:       Temperature (°C)
            soc:               Fused State of Charge (%)
            soh:               Fused State of Health (%)
            rul:               Fused Remaining Useful Life (cycles)
            calce_anomaly_raw: Raw CALCE IsolationForest output (+1 or -1)
            oxford_validation: Result dict from Oxford BatterySensorValidator
            calce_iso_score:   Raw anomaly score from CALCE IsolationForest

        Returns:
            RiskAssessment with score, level, faults, and component breakdown
        """
        faults = []
        component_scores = {}
        total_risk = 0.0
        is_hard_anomaly = False

        # ── Layer 1: Physical Safety Rules ──────────────────────────────────
        physical_risk, physical_faults, hard_violation = self._check_physical(
            voltage, current, temperature
        )
        component_scores["physical"] = round(physical_risk, 2)
        faults.extend(physical_faults)
        total_risk += physical_risk * self.W_PHYSICAL / 100.0

        if hard_violation:
            is_hard_anomaly = True

        # ── Layer 2: ML Statistical Anomaly (CALCE Isolation Forest) ────────
        ml_risk, ml_faults = self._check_ml_anomaly(
            calce_anomaly_raw, calce_iso_score
        )
        component_scores["ml_anomaly"] = round(ml_risk, 2)
        faults.extend(ml_faults)
        total_risk += ml_risk * self.W_ML_ANOMALY / 100.0

        if calce_anomaly_raw == -1:
            is_hard_anomaly = True

        # ── Layer 3: Sensor Integrity (Oxford Validator) ─────────────────────
        sensor_risk, sensor_faults = self._check_sensor_validity(oxford_validation)
        component_scores["sensor"] = round(sensor_risk, 2)
        faults.extend(sensor_faults)
        total_risk += sensor_risk * self.W_SENSOR / 100.0

        if not oxford_validation.get("is_valid", True):
            is_hard_anomaly = True

        # ── Layer 4: State-Based Risk ─────────────────────────────────────────
        soc_risk, soc_faults = self._check_soc(soc)
        soh_risk, soh_faults = self._check_soh(soh)
        rul_risk, rul_faults = self._check_rul(rul)

        component_scores["soc"] = round(soc_risk, 2)
        component_scores["soh"] = round(soh_risk, 2)
        component_scores["rul"] = round(rul_risk, 2)
        faults.extend(soc_faults)
        faults.extend(soh_faults)
        faults.extend(rul_faults)

        total_risk += soc_risk * self.W_SOC / 100.0
        total_risk += soh_risk * self.W_SOH / 100.0
        total_risk += rul_risk * self.W_RUL / 100.0

        # ── Composite Risk Score ──────────────────────────────────────────────
        risk_score = float(np.clip(total_risk, 0.0, 100.0))
        risk_level = self._classify_risk(risk_score, is_hard_anomaly)

        return RiskAssessment(
            risk_score=round(risk_score, 2),
            risk_level=risk_level,
            anomaly_detected=is_hard_anomaly or risk_score > 50.0,
            faults=faults,
            component_scores=component_scores,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Layer implementations
    # ─────────────────────────────────────────────────────────────────────────

    def _check_physical(
        self, voltage: float, current: float, temperature: float
    ) -> Tuple[float, List[str], bool]:
        """
        Layer 1: Physics-based hard limit checks.
        Returns (risk_0_100, faults, is_hard_violation).
        """
        risk = 0.0
        faults = []
        hard = False

        # Voltage checks
        if voltage > VOLTAGE_MAX:
            risk = 100.0
            hard = True
            faults.append(
                f"🔴 CRITICAL: Over-voltage {voltage:.3f}V > {VOLTAGE_MAX}V — "
                f"Overcharge hazard! Risk of thermal runaway."
            )
        elif voltage < VOLTAGE_MIN:
            risk = 100.0
            hard = True
            faults.append(
                f"🔴 CRITICAL: Under-voltage {voltage:.3f}V < {VOLTAGE_MIN}V — "
                f"Deep discharge detected! Potential cell reversal."
            )
        elif voltage > VOLTAGE_WARN_MAX:
            risk = max(risk, 55.0)
            faults.append(
                f"🟡 WARNING: High voltage {voltage:.3f}V — approaching overcharge limit."
            )
        elif voltage < VOLTAGE_WARN_MIN:
            risk = max(risk, 50.0)
            faults.append(
                f"🟡 WARNING: Low voltage {voltage:.3f}V — approaching deep discharge."
            )

        # Temperature checks
        if temperature > TEMP_MAX:
            risk = 100.0
            hard = True
            faults.append(
                f"🔴 CRITICAL: Over-temperature {temperature:.1f}°C > {TEMP_MAX}°C — "
                f"Thermal runaway imminent!"
            )
        elif temperature < TEMP_MIN:
            risk = 100.0
            hard = True
            faults.append(
                f"🔴 CRITICAL: Sub-zero temperature {temperature:.1f}°C — "
                f"Lithium plating risk!"
            )
        elif temperature > TEMP_WARN_MAX:
            risk = max(risk, 60.0)
            faults.append(
                f"🟡 WARNING: High temperature {temperature:.1f}°C — "
                f"Elevated thermal stress."
            )
        elif temperature < TEMP_WARN_MIN:
            risk = max(risk, 40.0)
            faults.append(
                f"🟡 WARNING: Low temperature {temperature:.1f}°C — "
                f"Reduced ionic conductivity."
            )

        # Current checks
        if abs(current) > CURRENT_MAX_A:
            risk = max(risk, 85.0)
            hard = True
            faults.append(
                f"🔴 CRITICAL: Over-current {current:.2f}A > ±{CURRENT_MAX_A}A — "
                f"Cell damage risk!"
            )
        elif abs(current) > CURRENT_WARN_A:
            risk = max(risk, 45.0)
            faults.append(
                f"🟡 WARNING: High current {current:.2f}A — "
                f"Accelerated degradation."
            )

        return risk, faults, hard

    def _check_ml_anomaly(
        self, calce_anomaly_raw: int, calce_iso_score: float = 0.0
    ) -> Tuple[float, List[str]]:
        """
        Layer 2: ML statistical anomaly check from CALCE IsolationForest.
        IsolationForest: +1 = normal, -1 = anomaly.
        """
        if calce_anomaly_raw == -1:
            # Convert IsolationForest score to anomaly percentage
            # score_samples returns values around 0 (more negative = more anomalous)
            anomaly_pct = float(np.clip((0.5 - calce_iso_score) * 100.0, 30.0, 100.0))
            return (
                anomaly_pct,
                [
                    f"🔴 ML ANOMALY: Battery behavior deviates significantly from "
                    f"normal operating profile (anomaly score: {anomaly_pct:.0f}%)."
                ],
            )
        return 0.0, []

    def _check_sensor_validity(
        self, oxford_validation: dict
    ) -> Tuple[float, List[str]]:
        """
        Layer 3: Oxford sensor validator result.
        """
        if not oxford_validation:
            return 0.0, []

        is_valid = oxford_validation.get("is_valid", True)
        anomaly_score_pct = oxford_validation.get("anomaly_score_pct", 0.0)
        oxford_faults = oxford_validation.get("faults", [])

        if not is_valid:
            risk = max(60.0, anomaly_score_pct)
            faults_out = [f"⚠️ SENSOR: {f}" for f in oxford_faults]
            return risk, faults_out

        if anomaly_score_pct > 30.0:
            return anomaly_score_pct * 0.5, [
                f"🟡 SENSOR: Elevated anomaly score {anomaly_score_pct:.0f}%."
            ]

        return 0.0, []

    def _check_soc(self, soc: float) -> Tuple[float, List[str]]:
        """Layer 4a: SOC-based risk."""
        if soc < SOC_LOW_CRIT:
            return (
                100.0,
                [f"🔴 SOC CRITICAL: {soc:.1f}% — Deep discharge risk! Charge immediately."],
            )
        elif soc < SOC_LOW_WARN:
            ratio = (SOC_LOW_WARN - soc) / (SOC_LOW_WARN - SOC_LOW_CRIT)
            return (
                30.0 + ratio * 70.0,
                [f"🟡 SOC LOW: {soc:.1f}% — Charge soon to prevent deep discharge."],
            )
        return 0.0, []

    def _check_soh(self, soh: float) -> Tuple[float, List[str]]:
        """Layer 4b: SOH-based risk."""
        if soh < SOH_CRIT:
            return (
                100.0,
                [f"🔴 SOH CRITICAL: {soh:.1f}% — Battery at end-of-life. Replace immediately."],
            )
        elif soh < SOH_WARN:
            ratio = (SOH_WARN - soh) / (SOH_WARN - SOH_CRIT)
            return (
                40.0 + ratio * 60.0,
                [f"🟡 SOH DEGRADED: {soh:.1f}% — Battery approaching end-of-life."],
            )
        return 0.0, []

    def _check_rul(self, rul: float) -> Tuple[float, List[str]]:
        """Layer 4c: RUL-based risk."""
        if rul < RUL_CRIT:
            return (
                100.0,
                [f"🔴 RUL CRITICAL: {rul:.0f} cycles remaining — Replace battery now!"],
            )
        elif rul < RUL_WARN:
            ratio = (RUL_WARN - rul) / (RUL_WARN - RUL_CRIT)
            return (
                30.0 + ratio * 70.0,
                [f"🟡 RUL LOW: {rul:.0f} cycles remaining — Plan battery replacement."],
            )
        return 0.0, []

    @staticmethod
    def _classify_risk(risk_score: float, is_hard_anomaly: bool) -> str:
        """Map risk score + hard anomaly flag to risk level string."""
        if is_hard_anomaly or risk_score >= 75.0:
            return "CRITICAL"
        elif risk_score >= 50.0:
            return "WARNING"
        elif risk_score >= 25.0:
            return "CAUTION"
        else:
            return "SAFE"
