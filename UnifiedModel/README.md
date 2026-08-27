# Unified Battery Intelligence Model

> **One model. One file. Four predictions.**

A Hierarchical Multi-Task Learning framework that unifies 9 specialist ML models from 3 battery datasets (CALCE, Oxford, NASA) into a single `battery_intelligence.pkl` file.

---

## Architecture

```
Battery Input (voltage, current, temperature, cycle_number)
                         │
                         ▼
              ┌─ Feature Router ─┐
              │ Physics-based     │
              │ feature derivation│
              └──────┬───────────┘
                     │
      ┌──────────────┼──────────────┐
      │              │              │
      ▼              ▼              ▼
 CALCE Models   Oxford Models  NASA Models
 ─────────────  ─────────────  ────────────
 SOC (RF)       SOC (XGBoost)  SOH (XGBoost)
 SOH (GBM)      SOH (XGBoost)  RUL (XGBoost)
 RUL (XGBoost)  Sensor Valid.
 Anomaly (IF)
      │              │              │
      └──────────────┼──────────────┘
                     │
                     ▼
         ┌─ Meta-Learner Fusion ─┐
         │  Ridge Regression     │
         │  SOC head: 2 inputs   │
         │  SOH head: 3 inputs   │
         │  RUL head: 2 inputs   │
         └──────────┬────────────┘
                    │
                    ▼
       ┌─ Hierarchical Risk Engine ─┐
       │  Layer 1: Physical rules   │
       │  Layer 2: ML anomaly (IF)  │
       │  Layer 3: Sensor validity  │
       │  Layer 4: SOC/SOH/RUL risk │
       └───────────┬────────────────┘
                   │
                   ▼
     BATTERY INTELLIGENCE OUTPUT
     ┌──────────────────────────────┐
     │  soc        : float (%)      │
     │  soh        : float (%)      │
     │  rul        : float (cycles) │
     │  risk_score : float (0-100)  │
     │  risk_level : SAFE/CAUTION/  │
     │               WARNING/CRITICAL│
     │  anomaly    : bool           │
     │  faults     : list[str]      │
     │  confidence : dict           │
     └──────────────────────────────┘
```

---

## Quick Start

### Step 1: Build the unified model PKL

```bash
cd UnifiedModel
python build_unified_model.py
```

This will:
1. Load all 9 specialist models from CALCE/, Oxford/, nasa/
2. Train meta-learner fusion layers (takes ~5 seconds)
3. Run 4 verification tests
4. Save `battery_intelligence.pkl`

### Step 2: Run predictions

```bash
python predict.py
```

### Step 3: Use in your own code

```python
import pickle

# Load once
with open("battery_intelligence.pkl", "rb") as f:
    model = pickle.load(f)

# Predict
result = model.predict(
    voltage=3.82,        # Volts
    current=-1.0,        # Amperes (negative = discharging)
    temperature=25.4,    # Celsius
    cycle_number=200     # Battery age (cycles)
)

print(f"SOC:   {result['soc']:.1f}%")
print(f"SOH:   {result['soh']:.1f}%")
print(f"RUL:   {result['rul']:.0f} cycles remaining")
print(f"Risk:  {result['risk_level']} ({result['risk_score']:.0f}/100)")
print(f"Faults: {result['faults']}")
```

---

## Digital Twin Integration

```python
import pickle
from battery_intelligence.digital_twin_adapter import BatteryDigitalTwinAdapter

# Load model
with open("battery_intelligence.pkl", "rb") as f:
    model = pickle.load(f)

# Create adapter (handles key name variations from any Digital Twin)
adapter = BatteryDigitalTwinAdapter(model)

# At each simulation timestep:
for timestep in simulation:
    dt_output = digital_twin.step()  # Any format

    # Supported key aliases:
    # voltage:      'voltage', 'voltage_v', 'V', 'terminal_voltage'
    # current:      'current', 'current_a', 'I', 'i_out'
    # temperature:  'temperature', 'temp', 'T', 'cell_temp'
    # cycle_number: 'cycle_number', 'cycle', 'n_cycle', 'step'

    result = adapter.process(dt_output)

# Get trend summary
trend = adapter.get_trend_summary()
print(trend)
```

---

## Output Format

| Key | Type | Range | Description |
|-----|------|-------|-------------|
| `soc` | float | 0–100 | State of Charge (%) |
| `soh` | float | 0–100 | State of Health (%) |
| `rul` | float | ≥ 0 | Remaining Useful Life (cycles) |
| `risk_score` | float | 0–100 | Composite risk score |
| `risk_level` | str | SAFE/CAUTION/WARNING/CRITICAL | Risk classification |
| `anomaly` | bool | — | True if anomaly detected |
| `faults` | list[str] | — | Human-readable fault descriptions |
| `confidence` | dict | — | Per-output confidence (%) |
| `specialist_predictions` | dict | — | Raw specialist model outputs |
| `risk_components` | dict | — | Risk score breakdown by layer |

---

## Risk Levels

| Level | Score | Meaning |
|-------|-------|---------|
| 🟢 **SAFE** | 0–24 | Normal operation |
| 🟡 **CAUTION** | 25–49 | Monitor closely |
| 🟠 **WARNING** | 50–74 | Action recommended |
| 🔴 **CRITICAL** | 75–100 | Immediate action required |

---

## Files

```
UnifiedModel/
├── battery_intelligence.pkl         ← The unified model (created by build script)
├── build_unified_model.py           ← Run this first to create the PKL
├── predict.py                       ← Inference demo with 6 scenarios
├── battery_intelligence/            ← Python package (embedded in PKL)
│   ├── __init__.py
│   ├── model.py                     ← BatteryIntelligenceModel class
│   ├── feature_router.py            ← Physics-based feature derivation
│   ├── meta_learner.py              ← Ridge meta-learner fusion
│   ├── risk_engine.py               ← Hierarchical 4-layer risk engine
│   └── digital_twin_adapter.py     ← Digital Twin integration interface
└── README.md
```

---

## Specialist Models Bundled

| Model | Source | Algorithm | Task |
|-------|--------|-----------|------|
| CALCE SOC | `CALCE/calce_soc_model.pkl` | RandomForest | SOC estimation |
| CALCE SOH | `CALCE/calce_soh_model.pkl` | GradientBoosting | SOH estimation |
| CALCE Anomaly | `CALCE/calce_anomaly_detector.pkl` | IsolationForest | Anomaly detection |
| CALCE RUL | `CALCE/xgb_calce.pkl` | XGBoost | RUL prediction |
| Oxford SOC | `Oxford/models/soc_model.joblib` | XGBoost | SOC estimation |
| Oxford SOH | `Oxford/models/soh_model.joblib` | XGBoost | SOH estimation |
| Oxford Validator | `Oxford/models/validation_model.joblib` | IsolationForest | Sensor validation |
| NASA SOH | `nasa/trained_models/xgb_nasa_soh.pkl` | XGBoost | SOH estimation |
| NASA RUL | `nasa/trained_models/xgb_nasa_rul.pkl` | XGBoost | RUL prediction |
