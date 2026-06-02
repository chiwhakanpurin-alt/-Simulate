import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class RunningBiomechanicsSimulator:
    """Simulate running mechanics data with foot pressure and injury risk"""
    
    def __init__(self, n_samples=1000, seed=42):
        np.random.seed(seed)
        self.n_samples = n_samples
        self.data = None
        
    def simulate_ground_reaction_force(self, runner_type='normal'):
        """
        Simulate Ground Reaction Force (GRF)
        runner_type: 'normal', 'overpronate', 'injured'
        """
        grf_data = []
        
        for _ in range(self.n_samples):
            # Stance phase (0-1 second)
            time = np.linspace(0, 1, 100)
            
            if runner_type == 'normal':
                # Normal GRF profile - double peak
                grf = (2.5 * np.exp(-((time-0.2)**2)/0.01) + 
                       2.0 * np.exp(-((time-0.7)**2)/0.02))
                grf += np.random.normal(0, 0.1, len(time))
                
            elif runner_type == 'overpronate':
                # Over-pronation - higher impact
                grf = (3.2 * np.exp(-((time-0.15)**2)/0.008) + 
                       2.3 * np.exp(-((time-0.65)**2)/0.02))
                grf += np.random.normal(0, 0.15, len(time))
                
            elif runner_type == 'injured':
                # Injured - asymmetric load
                grf = (2.8 * np.exp(-((time-0.25)**2)/0.015) + 
                       1.5 * np.exp(-((time-0.75)**2)/0.03))
                grf += np.random.normal(0, 0.2, len(time))
                
            grf_data.append(grf)
        
        return np.array(grf_data)
    
    def extract_features(self, grf_data, speed, cadence):
        """Extract biomechanical features from GRF"""
        features = []
        
        for i, grf in enumerate(grf_data):
            peak_force = np.max(grf)
            mean_force = np.mean(grf)
            loading_rate = np.gradient(grf)[:20].max()  # First 20ms
            braking_force = grf[10:30].mean()  # Mid-stance
            propulsive_force = grf[70:90].mean()  # Push-off
            impact_peak = grf[0:10].mean()  # Initial contact
            
            # Contact time
            contact_time = np.sum(grf > np.max(grf) * 0.1) / 100
            
            feature_row = {
                'peak_force': peak_force,
                'mean_force': mean_force,
                'loading_rate': loading_rate,
                'impact_peak': impact_peak,
                'braking_force': braking_force,
                'propulsive_force': propulsive_force,
                'contact_time': contact_time,
                'speed': speed[i],
                'cadence': cadence[i],
                'vertical_stiffness': peak_force / contact_time if contact_time > 0 else 0,
                'symmetry_index': propulsive_force / (braking_force + 0.1),
            }
            features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def generate_dataset(self, ratio={'normal': 0.6, 'overpronate': 0.2, 'injured': 0.2}):
        """Generate complete running dataset"""
        all_data = []
        all_labels = []
        
        for runner_type, proportion in ratio.items():
            n = int(self.n_samples * proportion)
            
            # Simulate GRF
            grf_data = self.simulate_ground_reaction_force(runner_type)
            
            # Generate running parameters
            speed = np.random.uniform(3, 6, n)  # m/s
            cadence = np.random.uniform(160, 180, n)  # steps/min
            
            # Extract features
            features_df = self.extract_features(grf_data[:n], speed, cadence)
            
            # Add label
            if runner_type == 'normal':
                label = 0
            elif runner_type == 'overpronate':
                label = 1
            else:  # injured
                label = 2
            
            features_df['injury_risk'] = label
            all_data.append(features_df)
            
            print(f"✓ Generated {n} samples: {runner_type}")
        
        self.data = pd.concat(all_data, ignore_index=True)
        return self.data
    
    def train_model(self):
        """Train ML model to predict injury risk"""
        X = self.data.drop(['injury_risk'], axis=1)
        y = self.data['injury_risk']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        print(f"\n📊 Model Performance:")
        print(f"  Train Accuracy: {train_score:.3f}")
        print(f"  Test Accuracy: {test_score:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🎯 Top Features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Normal', 'Overpronate', 'Injured']))
        
        return {
            'model': model,
            'scaler': scaler,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'feature_importance': feature_importance
        }
    
    def visualize_data(self):
        """Create visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Distribution of Peak Force by Risk Level
        risk_labels = {0: 'Normal', 1: 'Overpronate', 2: 'Injured'}
        for risk in [0, 1, 2]:
            subset = self.data[self.data['injury_risk'] == risk]
            axes[0, 0].hist(subset['peak_force'], alpha=0.6, label=risk_labels[risk], bins=20)
        axes[0, 0].set_xlabel('Peak Force (BW)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Peak Ground Reaction Force Distribution')
        axes[0, 0].legend()
        
        # 2. Loading Rate vs Impact Peak
        for risk in [0, 1, 2]:
            subset = self.data[self.data['injury_risk'] == risk]
            axes[0, 1].scatter(subset['loading_rate'], subset['impact_peak'], 
                             alpha=0.5, label=risk_labels[risk], s=30)
        axes[0, 1].set_xlabel('Loading Rate (BW/s)')
        axes[0, 1].set_ylabel('Impact Peak (BW)')
        axes[0, 1].set_title('Loading Rate vs Impact Peak')
        axes[0, 1].legend()
        
        # 3. Running Speed vs Cadence
        for risk in [0, 1, 2]:
            subset = self.data[self.data['injury_risk'] == risk]
            axes[1, 0].scatter(subset['speed'], subset['cadence'], 
                             alpha=0.5, label=risk_labels[risk], s=30)
        axes[1, 0].set_xlabel('Speed (m/s)')
        axes[1, 0].set_ylabel('Cadence (steps/min)')
        axes[1, 0].set_title('Running Speed vs Cadence')
        axes[1, 0].legend()
        
        # 4. Symmetry Index by Risk Level
        risk_data = [self.data[self.data['injury_risk'] == i]['symmetry_index'].values 
                     for i in [0, 1, 2]]
        axes[1, 1].boxplot(risk_data, labels=['Normal', 'Overpronate', 'Injured'])
        axes[1, 1].set_ylabel('Symmetry Index')
        axes[1, 1].set_title('Gait Symmetry by Risk Level')
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/running_biomechanics_analysis.png', dpi=300)
        print("\n📈 Visualization saved: running_biomechanics_analysis.png")
        plt.show()

# ============= RUN SIMULATION =============
if __name__ == "__main__":
    print("🏃 Running Biomechanics Simulator\n")
    print("=" * 50)
    
    # Create simulator
    simulator = RunningBiomechanicsSimulator(n_samples=1000)
    
    # Generate dataset
    print("\n🔧 Generating Dataset...")
    dataset = simulator.generate_dataset()
    
    # Show sample data
    print("\n📊 Sample Data (First 5 rows):")
    print(dataset.head())
    
    print(f"\n📈 Dataset Summary:")
    print(dataset.describe())
    
    # Save dataset
    dataset.to_csv('/mnt/user-data/outputs/running_biomechanics_data.csv', index=False)
    print("\n💾 Dataset saved: running_biomechanics_data.csv")
    
    # Train model
    print("\n" + "=" * 50)
    print("🤖 Training Injury Risk Prediction Model...")
    print("=" * 50)
    results = simulator.train_model()
    
    # Visualize
    print("\n" + "=" * 50)
    print("📊 Creating Visualizations...")
    print("=" * 50)
    simulator.visualize_data()
    
    print("\n✅ Complete! All files saved to /mnt/user-data/outputs/")
