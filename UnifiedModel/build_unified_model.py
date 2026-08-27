"""
Build Unified Battery Intelligence Model
=========================================
This script assembles all 9 specialist models into one unified
BatteryIntelligenceModel and saves it as a single battery_intelligence.pkl file.

Usage:
    python build_unified_model.py

Output:
    battery_intelligence.pkl  (in the same directory as this script)

The PKL file contains the entire model — no other files are needed at inference time.
"""

import sys
import io
import pickle
import time
from pathlib import Path

# Force UTF-8 output on Windows (avoids CP1252 emoji encoding errors)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # Mega-Project/

# Add UnifiedModel package to path
sys.path.insert(0, str(SCRIPT_DIR))

# Add Oxford src dir for class deserialization
OXFORD_SRC = PROJECT_ROOT / "Oxford" / "src"
sys.path.insert(0, str(OXFORD_SRC))

# Add NASA root for batteryml
NASA_ROOT = PROJECT_ROOT / "nasa"
sys.path.insert(0, str(NASA_ROOT))

# ── Model paths ──────────────────────────────────────────────────────────────
CALCE_DIR = PROJECT_ROOT / "CALCE"
OXFORD_DIR = PROJECT_ROOT / "Oxford"
NASA_DIR = PROJECT_ROOT / "nasa"

CALCE_SOC_PATH     = CALCE_DIR / "calce_soc_model.pkl"
CALCE_SOH_PATH     = CALCE_DIR / "calce_soh_model.pkl"
CALCE_ANOMALY_PATH = CALCE_DIR / "calce_anomaly_detector.pkl"
CALCE_RUL_PATH     = CALCE_DIR / "xgb_calce.pkl"

OXFORD_SOC_PATH       = OXFORD_DIR / "models" / "soc_model.joblib"
OXFORD_SOH_PATH       = OXFORD_DIR / "models" / "soh_model.joblib"
OXFORD_VALIDATOR_PATH = OXFORD_DIR / "models" / "validation_model.joblib"

NASA_SOH_PATH = NASA_DIR / "trained_models" / "xgb_nasa_soh.pkl"
NASA_RUL_PATH = NASA_DIR / "trained_models" / "xgb_nasa_rul.pkl"

OUTPUT_PATH = SCRIPT_DIR / "battery_intelligence.pkl"


def check_model_files():
    """Verify all source model files exist before proceeding."""
    required = {
        "CALCE SOC":      CALCE_SOC_PATH,
        "CALCE SOH":      CALCE_SOH_PATH,
        "CALCE Anomaly":  CALCE_ANOMALY_PATH,
        "CALCE RUL":      CALCE_RUL_PATH,
        "Oxford SOC":     OXFORD_SOC_PATH,
        "Oxford SOH":     OXFORD_SOH_PATH,
        "Oxford Validator": OXFORD_VALIDATOR_PATH,
        "NASA SOH":       NASA_SOH_PATH,
        "NASA RUL":       NASA_RUL_PATH,
    }
    missing = []
    print("\n  Checking source model files:")
    for name, path in required.items():
        exists = path.exists()
        status = "[OK]" if exists else "[MISSING]"
        print(f"    {status}  {name}: {path.name}")
        if not exists:
            missing.append(f"  - {name}: {path}")
    return missing


def main():
    t_start = time.time()

    print("=" * 65)
    print("  Battery Intelligence Model Builder v1.0.0")
    print("  Unified Hierarchical Multi-Task Learning Framework")
    print("=" * 65)

    # ── Step 0: Verify source files ──────────────────────────────────────────
    missing = check_model_files()
    if missing:
        print(f"\n  [ERROR] {len(missing)} model file(s) not found:")
        for m in missing:
            print(m)
        print("\n  Please ensure all specialist models are trained and saved.")
        sys.exit(1)

    print("\n  All source model files found.")

    # -- Step 1: Import and instantiate model --------------------------------
    from battery_intelligence.model import BatteryIntelligenceModel

    print("\n" + "-" * 65)
    print("  [1/4] Loading specialist models...")
    print("-" * 65)

    model = BatteryIntelligenceModel()

    # Load CALCE models
    model.load_calce_models(
        soc_path=str(CALCE_SOC_PATH),
        soh_path=str(CALCE_SOH_PATH),
        anomaly_path=str(CALCE_ANOMALY_PATH),
        rul_path=str(CALCE_RUL_PATH),
    )

    # Load Oxford models
    model.load_oxford_models(
        soc_path=str(OXFORD_SOC_PATH),
        soh_path=str(OXFORD_SOH_PATH),
        validator_path=str(OXFORD_VALIDATOR_PATH),
        oxford_src_dir=str(OXFORD_SRC),
    )

    # Load NASA models
    model.load_nasa_models(
        soh_path=str(NASA_SOH_PATH),
        rul_path=str(NASA_RUL_PATH),
    )

    # -- Step 2: Train meta-learners -----------------------------------------
    print("\n" + "-" * 65)
    print("  [2/4] Training meta-learner fusion layer...")
    print("-" * 65)
    model.train_meta_learners(n_samples=10000)

    # -- Step 3: Verify with test predictions --------------------------------
    print("\n" + "-" * 65)
    print("  [3/4] Running verification predictions...")
    print("-" * 65)

    test_cases = [
        {
            "name": "Normal Operation (mid-life battery)",
            "voltage": 3.82,
            "current": -1.0,
            "temperature": 25.4,
            "cycle_number": 200,
        },
        {
            "name": "Degraded Battery (near EOL)",
            "voltage": 3.55,
            "current": -0.8,
            "temperature": 32.0,
            "cycle_number": 800,
        },
        {
            "name": "Fault Scenario (over-temperature)",
            "voltage": 3.90,
            "current": 2.5,
            "temperature": 52.0,
            "cycle_number": 350,
        },
        {
            "name": "Fresh Battery (cycle 1)",
            "voltage": 4.18,
            "current": -0.5,
            "temperature": 22.0,
            "cycle_number": 1,
        },
    ]

    all_ok = True
    for tc in test_cases:
        try:
            result = model.predict(
                voltage=tc["voltage"],
                current=tc["current"],
                temperature=tc["temperature"],
                cycle_number=tc["cycle_number"],
            )
            soc_ok = 0.0 <= result["soc"] <= 100.0
            soh_ok = 0.0 <= result["soh"] <= 100.0
            rul_ok = result["rul"] >= 0.0
            risk_ok = result["risk_level"] in ["SAFE", "CAUTION", "WARNING", "CRITICAL"]
            case_ok = soc_ok and soh_ok and rul_ok and risk_ok

            status = "[PASS]" if case_ok else "[FAIL]"
            if not case_ok:
                all_ok = False
            print(f"\n  {status}  {tc['name']}")
            print(f"         SOC: {result['soc']:.1f}%  |  SOH: {result['soh']:.1f}%  "
                  f"|  RUL: {result['rul']:.0f} cycles")
            print(f"         Risk: {result['risk_level']} ({result['risk_score']:.1f})  "
                  f"|  Anomaly: {result['anomaly']}")
            if result["faults"]:
                for fault in result["faults"]:
                    print(f"         Fault: {fault}")
        except Exception as e:
            print(f"\n  [FAIL]  {tc['name']}: {e}")
            all_ok = False

    if not all_ok:
        print("\n  [ERROR] Some verification tests failed. Review output above.")
        sys.exit(1)

    # -- Step 4: Save to PKL -------------------------------------------------
    print("\n" + "-" * 65)
    print("  [4/4] Saving unified model to battery_intelligence.pkl...")
    print("-" * 65)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t_start

    print(f"\n  [OK] Saved: {OUTPUT_PATH}")
    print(f"  [OK] File size: {size_mb:.2f} MB")

    # ── Final status ─────────────────────────────────────────────────────────
    status = model.status()
    print("\n" + "=" * 65)
    print(f"  BUILD COMPLETE in {elapsed:.1f}s")
    print(f"  Models bundled:  {status['models_loaded']}/{status['models_total']}")
    print(f"  Architecture:    {status['architecture']}")
    print(f"  Version:         {status['version']}")
    print(f"  Output:          {OUTPUT_PATH.name}")
    print("=" * 65)
    print()
    print("  Load with:")
    print("    import pickle")
    print(f"    with open('{OUTPUT_PATH.name}', 'rb') as f:")
    print("        model = pickle.load(f)")
    print("    result = model.predict(voltage=3.82, current=-1.0,")
    print("                           temperature=25.4, cycle_number=200)")
    print()


if __name__ == "__main__":
    main()
