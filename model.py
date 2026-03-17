"""HELB Smart System — Fairness-Aware Band Allocation Model"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from config import Config


class HELBAllocationModel:
    """
    Fairness-aware multi-class classifier for HELB band allocation.
    Implements:
      - Softmax scoring (via GradientBoosting)
      - Fee Burden Index (FBI) fairness adjustment
      - Regional vulnerability weighting
      - Explainable feature contributions
    """

    FEATURES = ['household_income', 'num_dependents', 'siblings_in_uni',
                 'orphan_status', 'disability_status', 'school_type_enc',
                 'region_enc', 'sponsor_type_enc', 'annual_fees',
                 'income_per_dependent', 'fee_burden_index', 'region_weight']

    def __init__(self):
        self.model   = None
        self.scaler  = StandardScaler()
        self.le_school   = LabelEncoder()
        self.le_region   = LabelEncoder()
        self.le_sponsor  = LabelEncoder()
        self.is_trained  = False
        Config.create_dirs()

    # ── FEATURE ENGINEERING ──────────────────────────────────────────────────

    def _engineer(self, df):
        d = df.copy()
        d['income_per_dependent'] = d['household_income'] / (d['num_dependents'] + 1)
        d['fee_burden_index']     = d['annual_fees'] / (d['household_income'] + 1)
        d['region_weight']        = d['region'].map(Config.REGION_WEIGHTS).fillna(1.0)

        if not self.is_trained:
            d['school_type_enc']  = self.le_school.fit_transform(d['school_type'].fillna('County'))
            d['region_enc']       = self.le_region.fit_transform(d['region'].fillna('Nairobi'))
            d['sponsor_type_enc'] = self.le_sponsor.fit_transform(d['sponsor_type'].fillna('Government'))
        else:
            d['school_type_enc']  = self.le_school.transform(d['school_type'].fillna('County'))
            d['region_enc']       = self.le_region.transform(d['region'].fillna('Nairobi'))
            d['sponsor_type_enc'] = self.le_sponsor.transform(d['sponsor_type'].fillna('Government'))

        return d[self.FEATURES]

    # ── SYNTHETIC DATA GENERATION ─────────────────────────────────────────────

    def generate_training_data(self, n=3000):
        np.random.seed(42)
        records = []
        for _ in range(n):
            income   = np.random.choice([
                np.random.uniform(0, 20000),
                np.random.uniform(20000, 60000),
                np.random.uniform(60000, 150000),
                np.random.uniform(150000, 500000),
            ], p=[0.25, 0.35, 0.25, 0.15])
            deps     = np.random.randint(1, 8)
            siblings = np.random.randint(0, 3)
            orphan   = np.random.choice([0, 1], p=[0.85, 0.15])
            disabled = np.random.choice([0, 1], p=[0.92, 0.08])
            school   = np.random.choice(Config.SCHOOL_TYPES)
            region   = np.random.choice(Config.REGIONS)
            sponsor  = np.random.choice(Config.SPONSOR_TYPES, p=[0.65, 0.35])
            fees     = np.random.uniform(40000, 200000)
            employ   = np.random.choice(['Formal', 'Informal', 'Unemployed'])

            # Rule-based band label
            score = 0
            if income < 20000:   score += 40
            elif income < 60000: score += 30
            elif income < 150000:score += 15
            else:                score += 5
            score += deps * 3
            score += siblings * 4
            score += orphan * 15
            score += disabled * 12
            school_scores = {'National': 2, 'Extra-County': 4, 'County': 6, 'Sub-County': 9, 'Private': 1}
            score += school_scores.get(school, 5)
            score += (Config.REGION_WEIGHTS.get(region, 1.0) - 0.8) * 10
            fbi = fees / (income + 1)
            if fbi > 2: score += 10
            if employ == 'Unemployed': score += 8
            elif employ == 'Informal': score += 4

            if score >= 70:   band = 1
            elif score >= 55: band = 2
            elif score >= 40: band = 3
            elif score >= 25: band = 4
            else:             band = 5

            records.append({
                'household_income': income, 'num_dependents': deps,
                'siblings_in_uni': siblings, 'orphan_status': orphan,
                'disability_status': disabled, 'school_type': school,
                'region': region, 'sponsor_type': sponsor,
                'annual_fees': fees, 'parental_employment': employ,
                'band': band
            })
        return pd.DataFrame(records)

    # ── TRAINING ──────────────────────────────────────────────────────────────

    def train(self):
        print("Generating training data...")
        df = self.generate_training_data(3000)
        X  = self._engineer(df)
        y  = df['band']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_scaled = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        self.model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                max_depth=5, random_state=42)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True

        preds = self.model.predict(X_test_s)
        acc   = accuracy_score(y_test, preds)
        print(f"Model Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, target_names=[f'Band {i}' for i in range(1,6)]))
        self.save()
        return acc

    # ── PREDICTION ────────────────────────────────────────────────────────────

    def predict_band(self, applicant: dict):
        """
        Predict band + apply fairness adjustment + generate explanation.
        Returns dict with band, scores, explanation.
        """
        df  = pd.DataFrame([applicant])
        X   = self._engineer(df)
        Xs  = self.scaler.transform(X)

        proba = self.model.predict_proba(Xs)[0]
        band  = int(self.model.predict(Xs)[0])
        raw_score = float(np.max(proba) * 100)

        # ── Fairness Adjustment (Fee Burden Index) ──
        fbi = applicant['annual_fees'] / (applicant['household_income'] + 1)
        tau = 1.5
        lam = 0.3
        adjusted_score = raw_score
        if fbi > tau:
            adjustment = lam * (fbi - tau)
            adjusted_score = min(100, raw_score + adjustment * 10)
            if adjusted_score > raw_score + 5 and band < 5:
                band = max(1, band - 1)  # Move to more supported band

        # ── Regional vulnerability boost ──
        rw = Config.REGION_WEIGHTS.get(applicant.get('region', 'Nairobi'), 1.0)
        if rw >= 1.2 and band > 1:
            band = max(1, band - 1)

        # ── Explanation ──
        explanation = self._explain(applicant, band, fbi, rw)

        return {
            'band': band,
            'raw_score': round(raw_score, 2),
            'adjusted_score': round(adjusted_score, 2),
            'probabilities': {f'Band {i+1}': round(float(p)*100, 1) for i, p in enumerate(proba)},
            'explanation': explanation,
            'band_info': Config.BANDS[band]
        }

    def _explain(self, app, band, fbi, region_weight):
        reasons = []
        income = app['household_income']
        if income < 20000:
            reasons.append("very low household income (below KES 20,000)")
        elif income < 60000:
            reasons.append("low household income (below KES 60,000)")
        elif income < 150000:
            reasons.append("moderate household income")
        else:
            reasons.append("relatively high household income")

        if app['num_dependents'] >= 5:
            reasons.append(f"high number of dependents ({app['num_dependents']})")
        if app['orphan_status']:
            reasons.append("orphan status confirmed")
        if app['disability_status']:
            reasons.append("disability status confirmed")
        if app['siblings_in_uni'] > 0:
            reasons.append(f"{app['siblings_in_uni']} sibling(s) also in university")
        if fbi > 1.5:
            reasons.append(f"high fee burden index ({fbi:.2f}) relative to income")
        if region_weight >= 1.2:
            reasons.append(f"residence in a high-vulnerability region ({app.get('region','')})")
        if app['school_type'] in ['Sub-County', 'County']:
            reasons.append(f"attended a {app['school_type']} school")

        reason_str = "; ".join(reasons) if reasons else "overall socio-economic assessment"
        return (f"You have been placed in Band {band} based on: {reason_str}. "
                f"This entitles you to {Config.BANDS[band]['scholarship']}% government scholarship, "
                f"{Config.BANDS[band]['loan']}% HELB loan, and "
                f"{Config.BANDS[band]['household']}% household contribution.")

    # ── PERSISTENCE ───────────────────────────────────────────────────────────

    def save(self):
        path = Config.MODEL_PATH
        joblib.dump({'model': self.model, 'scaler': self.scaler,
                     'le_school': self.le_school, 'le_region': self.le_region,
                     'le_sponsor': self.le_sponsor}, path)
        print(f"Model saved: {path}")

    def load(self):
        path = Config.MODEL_PATH
        if os.path.exists(path):
            obj = joblib.load(path)
            self.model      = obj['model']
            self.scaler     = obj['scaler']
            self.le_school  = obj['le_school']
            self.le_region  = obj['le_region']
            self.le_sponsor = obj['le_sponsor']
            self.is_trained = True
            print("Model loaded.")
            return True
        return False
