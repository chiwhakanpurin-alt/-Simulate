import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json

class BioMechanicsPreprocessor:
    """Preprocess real sensor data for model prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_sensor_data(self, csv_file):
        """Load data from force plate or insole sensor"""
        df = pd.read_csv(csv_file)
        return df
    
    def process_grf_signal(self, grf_signal, window_size=100):
        """
        Process raw GRF signal
        grf_signal: array of force values
        window_size: samples per step cycle
        """
        steps = []
        for i in range(0, len(grf_signal) - window_size, window_size // 2):
            step = grf_signal[i:i+window_size]
            if len(step) == window_size:
                steps.append(step)
        return np.array(steps)
    
    def extract_biomechanical_features(self, grf_signal):
        """Extract key metrics from single GRF signal"""
        features = {
            'peak_force': np.max(grf_signal),
            'mean_force': np.mean(grf_signal),
            'min_force': np.min(grf_signal),
            'std_dev': np.std(grf_signal),
            'loading_rate': np.max(np.gradient(grf_signal[:len(grf_signal)//3])),
            'impulse': np.sum(grf_signal),  # Area under curve
            'symmetry': np.sum(grf_signal[:len(grf_signal)//2]) / (np.sum(grf_signal) + 1e-6),
            'contact_time_ratio': np.sum(grf_signal > np.max(grf_signal) * 0.1) / len(grf_signal),
            'variability': np.std(np.diff(grf_signal)),
            'energy_absorption': np.sum(np.gradient(grf_signal)**2)
        }
        return features
    
    def detect_gait_events(self, grf_signal, threshold=0.05):
        """Detect heel strike and toe-off events"""
        max_force = np.max(grf_signal)
        threshold_force = max_force * threshold
        
        above_threshold = grf_signal > threshold_force
        diff = np.diff(above_threshold.astype(int))
        
        heel_strikes = np.where(diff == 1)[0]  # Transitions from below to above
        toe_offs = np.where(diff == -1)[0]    # Transitions from above to below
        
        return {
            'heel_strikes': heel_strikes,
            'toe_offs': toe_offs,
            'contact_time': len(np.where(above_threshold)[0])
        }
    
    def generate_training_ready_data(self, raw_grf_array, labels):
        """
        Convert raw sensor data to ML-ready format
        
        raw_grf_array: [n_samples, sequence_length] array of GRF values
        labels: [n_samples] array of class labels
        """
        features_list = []
        
        for i, grf in enumerate(raw_grf_array):
            features = self.extract_biomechanical_features(grf)
            features['label'] = labels[i]
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        return df

class PredictionEngine:
    """Make predictions on new sensor data"""
    
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.preprocessor = BioMechanicsPreprocessor()
        self.risk_labels = {0: 'Normal', 1: 'Overpronate', 2: 'Injured'}
    
    def predict_single_step(self, grf_signal):
        """Predict injury risk for single step"""
        features = self.preprocessor.extract_biomechanical_features(grf_signal)
        
        # Format for model
        feature_names = ['peak_force', 'mean_force', 'loading_rate', 'impact_peak',
                        'braking_force', 'propulsive_force', 'contact_time',
                        'speed', 'cadence', 'vertical_stiffness', 'symmetry_index']
        
        X = np.array([[features.get(f, 0) for f in feature_names]])
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled, verbose=0)[0]
        confidence = np.max(prediction)
        
        return {
            'predicted_class': self.risk_labels.get(np.argmax(prediction), 'Unknown'),
            'confidence': float(confidence),
            'probabilities': {
                'normal': float(prediction[0]),
                'overpronate': float(prediction[1]),
                'injured': float(prediction[2])
            },
            'features': features
        }
    
    def analyze_session(self, grf_signals):
        """Analyze multiple steps from running session"""
        predictions = []
        
        for i, grf in enumerate(grf_signals):
            pred = self.predict_single_step(grf)
            pred['step_number'] = i + 1
            predictions.append(pred)
        
        # Summary statistics
        classes = [p['predicted_class'] for p in predictions]
        confidence_scores = [p['confidence'] for p in predictions]
        
        summary = {
            'total_steps': len(predictions),
            'most_common_class': max(set(classes), key=classes.count),
            'class_distribution': dict(pd.Series(classes).value_counts()),
            'avg_confidence': float(np.mean(confidence_scores)),
            'min_confidence': float(np.min(confidence_scores)),
            'max_confidence': float(np.max(confidence_scores)),
            'risk_assessment': self._assess_risk(classes),
            'recommendations': self._get_recommendations(classes, confidence_scores)
        }
        
        return {
            'predictions': predictions,
            'summary': summary
        }
    
    def _assess_risk(self, classes):
        """Assess overall running risk"""
        injured_count = classes.count('Injured')
        overpronate_count = classes.count('Overpronate')
        total = len(classes)
        
        if injured_count / total > 0.3:
            return 'HIGH - Immediate attention needed'
        elif (injured_count + overpronate_count) / total > 0.4:
            return 'MODERATE - Monitor and correct form'
        else:
            return 'LOW - Keep up current form'
    
    def _get_recommendations(self, classes, confidences):
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if 'Injured' in classes:
            recommendations.append('⚠️ Reduce running volume and intensity')
            recommendations.append('🏥 Consider medical evaluation')
            recommendations.append('💪 Focus on strengthening exercises')
        
        if 'Overpronate' in classes:
            recommendations.append('👟 Consider motion control shoes')
            recommendations.append('🦶 Work on ankle stability')
            recommendations.append('🧘 Do foot and arch strengthening')
        
        if np.mean(confidences) < 0.7:
            recommendations.append('📊 Inconclusive - collect more data')
        
        if not recommendations:
            recommendations.append('✅ Continue current training program')
        
        return recommendations

class DataGenerator:
    """Generate synthetic sensor data for testing"""
    
    @staticmethod
    def simulate_force_plate_output(duration=10, sampling_rate=1000):
        """Simulate force plate readings"""
        n_samples = int(duration * sampling_rate)
        t = np.linspace(0, duration, n_samples)
        
        # Multiple steps
        steps = []
        for step_num in range(int(duration * 3)):  # ~3 steps per second
            step_start = step_num / 3
            step_duration = 0.3
            
            if step_start <= duration:
                step_t = np.linspace(0, 2*np.pi, int(step_duration * sampling_rate))
                step_force = 2.5 * (1 + np.sin(step_t)) + np.random.normal(0, 0.1, len(step_t))
                steps.append(step_force)
        
        return np.concatenate(steps) if steps else np.zeros(n_samples)
    
    @staticmethod
    def simulate_insole_pressure(n_sensors=8, duration=10, sampling_rate=100):
        """Simulate insole pressure sensor array"""
        n_samples = duration * sampling_rate
        pressure_map = np.zeros((n_samples, n_sensors))
        
        for sensor in range(n_sensors):
            # Different sensors activate at different times
            t = np.linspace(0, duration, n_samples)
            phase = sensor * np.pi / n_sensors
            pressure_map[:, sensor] = np.maximum(
                0, 
                2 * np.sin(t + phase) + np.random.normal(0, 0.2, n_samples)
            )
        
        return pressure_map

# ============= EXAMPLE USAGE =============
if __name__ == "__main__":
    print("🔧 Biomechanics Data Processing & Prediction System\n")
    print("=" * 60)
    
    # Example 1: Data preprocessing
    print("\n1️⃣ Data Preprocessing Example")
    print("-" * 60)
    preprocessor = BioMechanicsPreprocessor()
    
    # Simulate sensor data
    gen = DataGenerator()
    grf_data = gen.simulate_force_plate_output(duration=5)
    grf_steps = preprocessor.process_grf_signal(grf_data)
    
    print(f"Processed {len(grf_steps)} steps from {len(grf_data)} samples")
    print(f"Sample step shape: {grf_steps[0].shape}")
    
    # Extract features
    sample_step = grf_steps[0]
    features = preprocessor.extract_biomechanical_features(sample_step)
    print(f"\nExtracted Features from Step 1:")
    for key, value in features.items():
        print(f"  {key}: {value:.3f}")
    
    # Detect gait events
    events = preprocessor.detect_gait_events(sample_step)
    print(f"\nGait Events (Step 1):")
    print(f"  Heel strikes: {len(events['heel_strikes'])} events")
    print(f"  Toe-offs: {len(events['toe_offs'])} events")
    print(f"  Contact time: {events['contact_time']} samples")
    
    # Example 2: Simulated sensor output
    print("\n\n2️⃣ Sensor Simulation Example")
    print("-" * 60)
    
    pressure_data = gen.simulate_insole_pressure(n_sensors=8, duration=3)
    print(f"Insole pressure map shape: {pressure_data.shape}")
    print(f"Sensors: 8 | Duration: 3s | Sampling rate: 100 Hz")
    print(f"Pressure range: {pressure_data.min():.2f} - {pressure_data.max():.2f}")
    
    # Example 3: Create sample dataset for model training
    print("\n\n3️⃣ Training Dataset Generation Example")
    print("-" * 60)
    
    # Generate synthetic training data
    n_samples = 100
    grf_steps = np.array([gen.simulate_force_plate_output(duration=0.5) 
                         for _ in range(n_samples)])
    
    labels = np.array([0]*60 + [1]*20 + [2]*20)  # 60% normal, 20% overpronate, 20% injured
    
    training_df = preprocessor.generate_training_ready_data(grf_steps, labels)
    
    print(f"Training dataset shape: {training_df.shape}")
    print(f"\nDataset Summary:")
    print(training_df.groupby('label')[['peak_force', 'mean_force', 'loading_rate']].mean())
    
    # Save example training data
    training_df.to_csv('/mnt/user-data/outputs/example_training_data.csv', index=False)
    print("\n✅ Example training data saved: example_training_data.csv")
    
    print("\n" + "=" * 60)
    print("✨ All utilities ready for use!")
