import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

class BatterySensorValidator:
    """
    Battery Voltage, Current, and Temperature Validation & Anomaly Model.
    Checks whether given (V, I, T) sensor measurements are 'RIGHT' (Valid) or 'NOT RIGHT' (Anomalous).
    Combines physical battery safety boundaries with an Isolation Forest ML Anomaly Detector.
    """
    
    # Physical operating thresholds for Kokam Li-ion Pouch Cell
    V_MIN = 2.50       # Volts
    V_MAX = 4.35       # Volts
    T_MIN = 0.0        # Deg C
    T_MAX = 55.0       # Deg C
    I_MAX_ABS = 4500.0 # mA (Max discharge/charge peak current)
    DV_DT_MAX = 0.25   # V/s max slope
    DT_DT_MAX = 2.0    # Deg C/s max slope

    def __init__(self, contamination=0.03, random_state=42):
        self.ml_anomaly_detector = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_cols = ['voltage_v', 'current_ma', 'temp_c', 'power_mw', 'dv_dt', 'dt_dt']
        self.is_fitted = False

    def check_physical_rules(self, voltage_v, current_ma, temp_c, dv_dt=0.0, dt_dt=0.0):
        """Checks physical operating bounds and rate of change limits."""
        faults = []
        
        # Voltage checks
        if voltage_v < self.V_MIN:
            faults.append(f"Under-voltage violation: {voltage_v:.3f}V is below safe limit ({self.V_MIN}V). Potential deep discharge/short.")
        elif voltage_v > self.V_MAX:
            faults.append(f"Over-voltage violation: {voltage_v:.3f}V exceeds safe limit ({self.V_MAX}V). Overcharge hazard!")
            
        # Temperature checks
        if temp_c < self.T_MIN:
            faults.append(f"Sub-zero temperature violation: {temp_c:.1f}°C is below safe threshold ({self.T_MIN}°C). Lithium plating risk!")
        elif temp_c > self.T_MAX:
            faults.append(f"Over-temperature alert: {temp_c:.1f}°C exceeds safe thermal limit ({self.T_MAX}°C). Thermal runaway hazard!")
            
        # Current checks
        if abs(current_ma) > self.I_MAX_ABS:
            faults.append(f"Over-current peak: {current_ma:.1f}mA exceeds max limit (±{self.I_MAX_ABS}mA). Over-current fault.")
            
        # Derivative checks
        if abs(dv_dt) > self.DV_DT_MAX:
            faults.append(f"Unrealistic voltage spike: dV/dt = {dv_dt:.3f} V/s exceeds physical limit ({self.DV_DT_MAX} V/s). Sensor noise/glitch.")
            
        if abs(dt_dt) > self.DT_DT_MAX:
            faults.append(f"Unrealistic thermal rise rate: dT/dt = {dt_dt:.2f} °C/s exceeds physical limit ({self.DT_DT_MAX} °C/s). Sensor disconnect.")
            
        return faults

    def fit(self, X_clean):
        """Fits Isolation Forest on clean/normal battery feature data."""
        X_feat = self._prepare_features(X_clean)
        self.ml_anomaly_detector.fit(X_feat)
        self.is_fitted = True
        return self

    def _prepare_features(self, df_or_dict):
        """Ensures all feature columns exist."""
        if isinstance(df_or_dict, dict):
            v = float(df_or_dict.get('voltage_v', 3.7))
            i = float(df_or_dict.get('current_ma', 0.0))
            t = float(df_or_dict.get('temp_c', 40.0))
            p = float(df_or_dict.get('power_mw', v * i))
            dv = float(df_or_dict.get('dv_dt', 0.0))
            dt = float(df_or_dict.get('dt_dt', 0.0))
            return pd.DataFrame([{
                'voltage_v': v,
                'current_ma': i,
                'temp_c': t,
                'power_mw': p,
                'dv_dt': dv,
                'dt_dt': dt
            }])
        else:
            df = df_or_dict.copy()
            if 'power_mw' not in df.columns:
                df['power_mw'] = df['voltage_v'] * df['current_ma']
            for col in ['dv_dt', 'dt_dt']:
                if col not in df.columns:
                    df[col] = 0.0
            return df[self.feature_cols]

    def validate_sample(self, voltage_v, current_ma, temp_c, dv_dt=0.0, dt_dt=0.0):
        """
        Validates a single (V, I, T) battery measurement point.
        Returns dictionary with validation outcome:
        {
          'is_valid': bool,
          'status': 'RIGHT' or 'NOT RIGHT',
          'faults': list[str],
          'anomaly_score_pct': float,
          'details': str
        }
        """
        # Step 1: Physical Rule Check
        physical_faults = self.check_physical_rules(voltage_v, current_ma, temp_c, dv_dt, dt_dt)
        
        # Step 2: ML Isolation Forest Check
        sample_dict = {
            'voltage_v': voltage_v,
            'current_ma': current_ma,
            'temp_c': temp_c,
            'power_mw': voltage_v * current_ma,
            'dv_dt': dv_dt,
            'dt_dt': dt_dt
        }
        
        ml_faults = []
        anomaly_score_pct = 0.0
        
        if self.is_fitted:
            X_samp = self._prepare_features(sample_dict)
            score = self.ml_anomaly_detector.score_samples(X_samp)[0]
            # IsolationForest score_samples: values near 0 or negative indicate anomalies
            # Map score to percentage (0% = perfectly normal, 100% = severe anomaly)
            anomaly_score_pct = float(np.clip((0.5 - score) * 100.0, 0.0, 100.0))
            
            is_ml_anomaly = self.ml_anomaly_detector.predict(X_samp)[0] == -1
            if is_ml_anomaly:
                ml_faults.append(f"ML Statistical Anomaly: Combination of V={voltage_v:.2f}V, I={current_ma:.0f}mA, T={temp_c:.1f}°C deviates significantly from normal battery operating profile (Anomaly Score: {anomaly_score_pct:.1f}%).")

        all_faults = physical_faults + ml_faults
        is_valid = len(all_faults) == 0
        
        return {
            'is_valid': is_valid,
            'status': 'RIGHT' if is_valid else 'NOT RIGHT',
            'faults': all_faults,
            'anomaly_score_pct': anomaly_score_pct if self.is_fitted else (0.0 if is_valid else 85.0),
            'summary': "All readings (Voltage, Current, Temperature) are RIGHT and within safe operating parameters." if is_valid else f"Readings NOT RIGHT: {len(all_faults)} fault(s) detected!"
        }

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'detector': self.ml_anomaly_detector,
            'feature_cols': self.feature_cols,
            'is_fitted': self.is_fitted
        }, filepath)

    @classmethod
    def load(cls, filepath):
        data = joblib.load(filepath)
        obj = cls()
        obj.ml_anomaly_detector = data['detector']
        obj.feature_cols = data['feature_cols']
        obj.is_fitted = data['is_fitted']
        return obj
