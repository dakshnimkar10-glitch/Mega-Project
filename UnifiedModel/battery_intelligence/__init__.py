"""
Battery Intelligence Package
=============================
Unified Hierarchical Multi-Task Learning Framework for Li-ion Battery State Estimation.

Usage:
    import pickle
    with open("battery_intelligence.pkl", "rb") as f:
        model = pickle.load(f)

    result = model.predict(voltage=3.82, current=-1.0, temperature=25.4, cycle_number=100)
    print(result)
"""

from .model import BatteryIntelligenceModel

__all__ = ["BatteryIntelligenceModel"]
__version__ = "1.0.0"
