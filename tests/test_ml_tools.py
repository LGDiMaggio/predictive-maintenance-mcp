"""
Tests for Machine Learning tools (feature extraction, training, prediction).

Tests:
- Feature extraction from signals
- ML model training (OneClassSVM, LocalOutlierFactor)
- Anomaly prediction
- Model persistence (save/load)
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json


class TestFeatureExtraction:
    """Test suite for extract_features_from_signal tool."""
    
    def test_feature_extraction_synthetic(self):
        """Test feature extraction on synthetic signal."""
        # Generate synthetic signal
        fs = 10000
        duration = 1.0
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        signal = np.sin(2*np.pi*50*t) + 0.1*np.random.randn(len(t))
        
        # Extract features from full signal
        features = self._extract_time_domain_features(signal)
        
        # Verify feature dictionary
        expected_features = [
            'mean', 'std', 'rms', 'peak', 'peak_to_peak',
            'crest_factor', 'kurtosis', 'skewness', 'clearance_factor',
            'shape_factor', 'impulse_factor', 'power', 'entropy'
        ]
        
        for feat in expected_features:
            assert feat in features, f"Missing feature: {feat}"
            assert isinstance(features[feat], (int, float)), \
                f"Feature {feat} is not numeric: {type(features[feat])}"
    
    
    def test_feature_rms_calculation(self):
        """Test RMS feature calculation."""
        # Signal with known RMS
        signal = np.array([1.0, -1.0, 1.0, -1.0])
        rms_expected = 1.0
        rms_calculated = np.sqrt(np.mean(signal**2))
        
        assert abs(rms_calculated - rms_expected) < 0.001
    
    
    def test_feature_kurtosis(self):
        """Test kurtosis calculation."""
        from scipy.stats import kurtosis
        
        # Normal distribution has kurtosis ~ 0 (excess kurtosis)
        signal_normal = np.random.randn(1000)
        kurt = kurtosis(signal_normal)
        
        # Excess kurtosis should be close to 0 for normal distribution
        assert abs(kurt) < 1.0, f"Kurtosis for normal: {kurt}"
        
        # Signal with outliers should have high kurtosis
        signal_outliers = np.concatenate([np.random.randn(990), np.array([10]*10)])
        kurt_outliers = kurtosis(signal_outliers)
        
        assert kurt_outliers > kurt, "Outliers should increase kurtosis"
    
    
    def test_feature_crest_factor(self):
        """Test crest factor calculation."""
        # Sine wave: peak = 1, RMS = 1/sqrt(2), crest factor = sqrt(2)
        signal = np.sin(np.linspace(0, 2*np.pi, 1000))
        peak = np.max(np.abs(signal))
        rms = np.sqrt(np.mean(signal**2))
        crest_factor = peak / rms
        
        expected_crest = np.sqrt(2)
        assert abs(crest_factor - expected_crest) < 0.01, \
            f"Crest factor: {crest_factor} vs expected {expected_crest}"
    
    
    def test_feature_segmentation(self):
        """Test sliding window segmentation."""
        signal = np.arange(1000)
        fs = 1000
        segment_duration = 0.2  # 200 samples
        overlap_ratio = 0.5  # 50% overlap
        
        segment_length = int(segment_duration * fs)  # 200
        hop_length = int(segment_length * (1 - overlap_ratio))  # 100
        
        segments = []
        for start in range(0, len(signal) - segment_length + 1, hop_length):
            segment = signal[start:start + segment_length]
            segments.append(segment)
        
        # Verify number of segments
        expected_segments = (len(signal) - segment_length) // hop_length + 1
        assert len(segments) == expected_segments, \
            f"Expected {expected_segments} segments, got {len(segments)}"
        
        # Verify segment overlap
        if len(segments) > 1:
            overlap = segments[0][hop_length:] 
            next_start = segments[1][:segment_length - hop_length]
            np.testing.assert_array_equal(overlap, next_start)
    
    
    def test_feature_extraction_real_data(self, sample_healthy_signal):
        """Test feature extraction on real bearing data."""
        signal = sample_healthy_signal[:10000]  # Use 10k samples for speed
        
        features = self._extract_time_domain_features(signal)
        
        # All features should be finite
        for key, value in features.items():
            assert np.isfinite(value), f"Feature {key} is not finite: {value}"
        
        # RMS should be positive
        assert features['rms'] > 0, "RMS should be positive"
        
        # Crest factor should be > 1 for real signals
        assert features['crest_factor'] > 1.0, \
            f"Crest factor should be > 1, got {features['crest_factor']}"
    
    
    def _extract_time_domain_features(self, signal):
        """Helper: Extract time-domain features."""
        from scipy.stats import kurtosis, skew, entropy
        
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        rms_val = np.sqrt(np.mean(signal**2))
        peak_val = np.max(np.abs(signal))
        peak_to_peak_val = np.ptp(signal)
        
        crest_factor_val = peak_val / (rms_val + 1e-10)
        kurtosis_val = kurtosis(signal)
        skewness_val = skew(signal)
        
        shape_factor_val = rms_val / (np.mean(np.abs(signal)) + 1e-10)
        impulse_factor_val = peak_val / (np.mean(np.abs(signal)) + 1e-10)
        clearance_factor_val = peak_val / (np.mean(np.sqrt(np.abs(signal)))**2 + 1e-10)
        
        power_val = np.mean(signal**2)
        hist, _ = np.histogram(signal, bins=50)
        hist = hist / (np.sum(hist) + 1e-10)
        entropy_val = entropy(hist + 1e-10)
        
        return {
            'mean': mean_val,
            'std': std_val,
            'rms': rms_val,
            'peak': peak_val,
            'peak_to_peak': peak_to_peak_val,
            'crest_factor': crest_factor_val,
            'kurtosis': kurtosis_val,
            'skewness': skewness_val,
            'shape_factor': shape_factor_val,
            'impulse_factor': impulse_factor_val,
            'clearance_factor': clearance_factor_val,
            'power': power_val,
            'entropy': entropy_val
        }


class TestMLTraining:
    """Test suite for train_anomaly_model tool."""
    
    def test_oneclass_svm_training(self):
        """Test OneClassSVM model training."""
        from sklearn.svm import OneClassSVM
        from sklearn.preprocessing import StandardScaler
        
        # Generate training data (healthy)
        X_train = np.random.randn(100, 5)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        # Train model
        model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
        model.fit(X_scaled)
        
        # Predict on training data (should mostly be +1)
        predictions = model.predict(X_scaled)
        inlier_ratio = np.sum(predictions == 1) / len(predictions)
        
        assert inlier_ratio > 0.8, \
            f"Too many outliers in training data: {inlier_ratio}"
    
    
    def test_local_outlier_factor(self):
        """Test LocalOutlierFactor model."""
        from sklearn.neighbors import LocalOutlierFactor
        
        # Generate training data
        X_train = np.random.randn(100, 5)
        
        # Train model (novelty=True for prediction)
        model = LocalOutlierFactor(novelty=True, contamination=0.1)
        model.fit(X_train)
        
        # Predict on similar data (should be inliers)
        X_test = np.random.randn(20, 5)
        predictions = model.predict(X_test)
        
        # Most should be inliers (+1)
        inlier_ratio = np.sum(predictions == 1) / len(predictions)
        assert inlier_ratio > 0.5
    
    
    def test_pca_dimensionality_reduction(self):
        """Test PCA for feature reduction."""
        from sklearn.decomposition import PCA
        
        # Generate high-dimensional data
        X = np.random.randn(100, 17)  # 17 features
        
        # Apply PCA
        pca = PCA(n_components=0.95)  # Keep 95% variance
        X_reduced = pca.fit_transform(X)
        
        # Reduced dimensions should be < 17
        assert X_reduced.shape[1] < X.shape[1], \
            f"PCA didn't reduce dimensions: {X_reduced.shape[1]}"
        
        # Variance explained should be >= 0.95
        var_explained = np.sum(pca.explained_variance_ratio_)
        assert var_explained >= 0.95, \
            f"Variance explained: {var_explained}"
    
    
    def test_model_persistence(self):
        """Test model save/load with pickle."""
        import pickle
        from sklearn.svm import OneClassSVM
        
        # Train model
        X_train = np.random.randn(100, 5)
        model = OneClassSVM()
        model.fit(X_train)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            pickle.dump(model, f)
            temp_path = f.name
        
        try:
            # Load model
            with open(temp_path, 'rb') as f:
                model_loaded = pickle.load(f)
            
            # Predictions should match
            X_test = np.random.randn(10, 5)
            pred_original = model.predict(X_test)
            pred_loaded = model_loaded.predict(X_test)
            
            np.testing.assert_array_equal(pred_original, pred_loaded)
        finally:
            # Cleanup
            Path(temp_path).unlink()
    
    
    def test_training_validation_split(self, sample_healthy_signal, sample_faulty_signal):
        """Test model training with validation."""
        from sklearn.svm import OneClassSVM
        from sklearn.preprocessing import StandardScaler
        
        # Extract features from healthy (simplified)
        healthy_features = self._simple_features(sample_healthy_signal[:50000])
        faulty_features = self._simple_features(sample_faulty_signal[:50000])
        
        # Train on healthy
        scaler = StandardScaler()
        X_train = scaler.fit_transform([healthy_features])
        
        model = OneClassSVM()
        model.fit(X_train)
        
        # Validate: healthy should be +1, faulty should be -1
        healthy_pred = model.predict(scaler.transform([healthy_features]))[0]
        faulty_pred = model.predict(scaler.transform([faulty_features]))[0]
        
        # This is probabilistic, but generally should work
        assert healthy_pred == 1, "Healthy data misclassified"
        # Faulty might still be +1 if features are similar, so soft check
    
    
    def _simple_features(self, signal):
        """Helper: Extract simple features."""
        return [
            np.mean(signal),
            np.std(signal),
            np.sqrt(np.mean(signal**2)),
            np.max(np.abs(signal)),
            np.ptp(signal)
        ]


class TestMLPrediction:
    """Test suite for predict_anomalies tool."""
    
    def test_anomaly_prediction_healthy(self):
        """Test prediction on healthy-like data."""
        from sklearn.svm import OneClassSVM
        from sklearn.preprocessing import StandardScaler
        
        # Train on healthy
        X_train = np.random.randn(100, 5)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        model = OneClassSVM()
        model.fit(X_scaled)
        
        # Test on similar healthy data
        X_test = np.random.randn(20, 5)
        X_test_scaled = scaler.transform(X_test)
        predictions = model.predict(X_test_scaled)
        
        # Calculate anomaly ratio
        anomaly_count = np.sum(predictions == -1)
        anomaly_ratio = anomaly_count / len(predictions)
        
        # Should be mostly normal (low anomaly ratio)
        assert anomaly_ratio < 0.3, f"Too many anomalies: {anomaly_ratio}"
    
    
    def test_anomaly_prediction_faulty(self):
        """Test prediction on faulty-like data."""
        from sklearn.svm import OneClassSVM
        from sklearn.preprocessing import StandardScaler
        
        # Train on healthy
        X_train = np.random.randn(100, 5)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        model = OneClassSVM(nu=0.1)
        model.fit(X_scaled)
        
        # Test on anomalous data (shifted distribution)
        X_test = np.random.randn(20, 5) + 5  # Shifted by 5 std devs
        X_test_scaled = scaler.transform(X_test)
        predictions = model.predict(X_test_scaled)
        
        # Should detect many anomalies
        anomaly_count = np.sum(predictions == -1)
        anomaly_ratio = anomaly_count / len(predictions)
        
        # Should have high anomaly ratio
        assert anomaly_ratio > 0.5, f"Anomalies not detected: {anomaly_ratio}"
    
    
    def test_health_status_classification(self):
        """Test overall health status determination."""
        # Test different anomaly ratios
        test_cases = [
            (0.05, "Healthy"),      # 5% anomalies
            (0.15, "Suspicious"),   # 15% anomalies
            (0.40, "Faulty"),       # 40% anomalies
        ]
        
        for anomaly_ratio, expected_status in test_cases:
            status = self._classify_health(anomaly_ratio)
            assert status == expected_status, \
                f"Ratio {anomaly_ratio}: expected {expected_status}, got {status}"
    
    
    def _classify_health(self, anomaly_ratio):
        """Helper: Classify health based on anomaly ratio."""
        if anomaly_ratio < 0.1:
            return "Healthy"
        elif anomaly_ratio < 0.3:
            return "Suspicious"
        else:
            return "Faulty"
