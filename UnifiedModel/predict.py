"""
Unified Battery Intelligence Model — Inference Demo
=====================================================
Loads battery_intelligence.pkl and demonstrates all prediction capabilities.

Usage:
    python predict.py

Or load programmatically from any directory:
    import pickle
    from pathlib import Path

    with open(Path(__file__).parent / "battery_intelligence.pkl", "rb") as f:
        model = pickle.load(f)

    result = model.predict(voltage=3.82, current=-1.0,
                           temperature=25.4, cycle_number=100)
"""

import sys
import io
import pickle
from pathlib import Path

# Force UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

PKL_PATH = SCRIPT_DIR / "battery_intelligence.pkl"


def print_result(result: dict, title: str):
    """Pretty-print a prediction result."""
    w = 62
    print("\n" + "─" * w)
    print(f"  {title}")
    print("─" * w)

    # Primary Outputs
    print(f"  {'STATE OF CHARGE (SOC)':<30}  {result['soc']:.2f} %")
    print(f"  {'STATE OF HEALTH (SOH)':<30}  {result['soh']:.2f} %")
    print(f"  {'REMAINING USEFUL LIFE (RUL)':<30}  {result['rul']:.0f} cycles")

    # Risk
    risk_icons = {"SAFE": "🟢", "CAUTION": "🟡", "WARNING": "🟠", "CRITICAL": "🔴"}
    icon = risk_icons.get(result["risk_level"], "⚪")
    print(f"  {'RISK SCORE':<30}  {result['risk_score']:.1f} / 100")
    print(f"  {'RISK LEVEL':<30}  {icon} {result['risk_level']}")
    print(f"  {'ANOMALY DETECTED':<30}  {'YES ⚠️' if result['anomaly'] else 'No'}")

    # Faults
    if result["faults"]:
        print(f"\n  Detected Faults ({len(result['faults'])}):")
        for fault in result["faults"]:
            print(f"    {fault}")

    # Confidence
    conf = result["confidence"]
    print(f"\n  Confidence:  "
          f"SOC={conf['soc_confidence']:.0f}%  "
          f"SOH={conf['soh_confidence']:.0f}%  "
          f"RUL={conf['rul_confidence']:.0f}%")

    # Specialist breakdown
    sp = result["specialist_predictions"]
    print(f"\n  Specialist SOC:  CALCE={sp['soc']['calce']:.1f}%  Oxford={sp['soc']['oxford']:.1f}%")
    print(f"  Specialist SOH:  CALCE={sp['soh']['calce']:.1f}%  "
          f"Oxford={sp['soh']['oxford']:.1f}%  "
          f"NASA={sp['soh']['nasa_ratio']*100:.1f}%")
    print(f"  Specialist RUL:  CALCE={sp['rul']['calce']:.0f}  NASA={sp['rul']['nasa']:.0f}")


def run_digital_twin_demo(model):
    """Demonstrate Digital Twin adapter integration."""
    from battery_intelligence.digital_twin_adapter import BatteryDigitalTwinAdapter

    print("\n" + "=" * 62)
    print("  DIGITAL TWIN INTEGRATION DEMO")
    print("  Simulating 10 timesteps of digital twin output")
    print("=" * 62)

    adapter = BatteryDigitalTwinAdapter(model, window_size=10)

    # Simulate digital twin sending data at each timestep
    import numpy as np
    rng = np.random.default_rng(42)

    for step in range(10):
        # Simulate realistic battery discharge curve
        cycle = 300
        soc_decay = 1.0 - (step / 10.0) * 0.5
        voltage = 3.2 + soc_decay * 0.9 + rng.normal(0, 0.01)

        # Digital twin output dict (any key format works)
        dt_output = {
            "terminal_voltage": voltage,       # alias for 'voltage'
            "i_out": -0.8 + rng.normal(0, 0.05),  # alias for 'current'
            "cell_temp": 28.0 + step * 0.5,       # alias for 'temperature'
            "n_cycle": cycle,                      # alias for 'cycle_number'
        }

        result = adapter.process(dt_output)
        print(f"  Step {step+1:2d} | "
              f"V={dt_output['terminal_voltage']:.3f}V | "
              f"SOC={result['soc']:.1f}% | "
              f"SOH={result['soh']:.1f}% | "
              f"RUL={result['rul']:.0f} | "
              f"Risk={result['risk_level']}")

    # Show trend summary
    trend = adapter.get_trend_summary()
    print(f"\n  SOC trend:   {trend['soc_trend']['max']:.1f}% → {trend['soc_trend']['min']:.1f}%  ({trend['soc_trend']['trend']})")
    print(f"  Risk events: {trend['risk_summary']['critical_events']} critical, {trend['risk_summary']['warning_events']} warnings")


def main():
    print("=" * 62)
    print("  Battery Intelligence Model — Unified Inference Demo")
    print("  Hierarchical Multi-Task Learning Framework v1.0.0")
    print("=" * 62)

    # ── Load Model ───────────────────────────────────────────────────────────
    if not PKL_PATH.exists():
        print(f"\n  [ERROR] battery_intelligence.pkl not found at:")
        print(f"          {PKL_PATH}")
        print("\n  Run build_unified_model.py first to create the PKL file.")
        sys.exit(1)

    print(f"\n  Loading battery_intelligence.pkl...")
    with open(PKL_PATH, "rb") as f:
        model = pickle.load(f)

    print(f"  {model}")

    # ── Test Scenarios ───────────────────────────────────────────────────────
    scenarios = [
        {
            "title": "Scenario 1: Normal Operation (Mid-Life Battery, Cycle 200)",
            "voltage": 3.82,
            "current": -1.0,
            "temperature": 25.4,
            "cycle_number": 200,
        },
        {
            "title": "Scenario 2: Degraded Battery (Near EOL, Cycle 800)",
            "voltage": 3.55,
            "current": -0.8,
            "temperature": 32.0,
            "cycle_number": 800,
        },
        {
            "title": "Scenario 3: Over-Temperature Fault",
            "voltage": 3.90,
            "current": 2.5,
            "temperature": 52.0,
            "cycle_number": 350,
        },
        {
            "title": "Scenario 4: Over-Voltage Risk (Near Full Charge)",
            "voltage": 4.22,
            "current": 1.2,
            "temperature": 29.0,
            "cycle_number": 150,
        },
        {
            "title": "Scenario 5: Low SOC Alert (Deep Discharge Risk)",
            "voltage": 2.85,
            "current": -0.5,
            "temperature": 24.0,
            "cycle_number": 400,
        },
        {
            "title": "Scenario 6: Fresh Battery (Cycle 1)",
            "voltage": 4.18,
            "current": -0.5,
            "temperature": 22.0,
            "cycle_number": 1,
        },
    ]

    print("\n" + "=" * 62)
    print("  RUNNING 6 TEST SCENARIOS")
    print("=" * 62)

    for scenario in scenarios:
        title = scenario.pop("title")
        result = model.predict(**scenario)
        print_result(result, title)

    # ── Digital Twin Demo ────────────────────────────────────────────────────
    run_digital_twin_demo(model)

    print("\n" + "=" * 62)
    print("  Inference complete. Model is ready for production use.")
    print("=" * 62)


if __name__ == "__main__":
    main()
