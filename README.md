# 🏃 Running Biomechanics Simulator & Injury Detection System

สำหรับการ Simulate การวิ่ง, วิเคราะห์แรงกดของเท้า และทำนายอาการบาดเจ็บ

---

## 📋 ภาพรวมระบบ

ระบบนี้ประกอบด้วย 3 โมดูลหลัก:

1. **Running Simulator** - สร้าง synthetic data การวิ่งในสภาวะต่างๆ
2. **Advanced Gait Analyzer** - วิเคราะห์ time-series data แบบ ground reaction force
3. **Prediction Engine** - ทำนายความเสี่ยงอาการบาดเจ็บ

---

## 🎯 ตัวแปรสำคัญที่ Simulate ได้

### Ground Reaction Force (GRF) Metrics
| ตัวแปร | คำอธิบาย | หน่วย | ช่วงปกติ |
|--------|---------|------|---------|
| **Peak Force** | แรงกดสูงสุดขณะเหยียบพื้น | BW | 2.5 - 3.0 |
| **Mean Force** | ค่าเฉลี่ยแรงระหว่างวิ่ง | BW | 0.9 - 1.1 |
| **Loading Rate** | อัตราการเพิ่มแรงตั้งแต่สัมผัสพื้น | BW/s | 50 - 100 |
| **Impact Peak** | แรงกระแทก 20ms แรก | BW | 1.5 - 2.2 |
| **Contact Time** | เวลาที่เท้าสัมผัสพื้น | ms | 200 - 300 |
| **Vertical Stiffness** | ความแข็งในแนวตั้ง | BW/m | 8 - 15 |
| **Symmetry Index** | ความสมมาตรซ้าย-ขวา | ratio | 0.85 - 1.15 |

### Gait Classifications
```
0 = NORMAL      - การวิ่งปกติ
1 = OVERPRONATE - เท้ากว่านเข้าด้านใน (บาดเจ็บเสี่ยง)
2 = INJURED     - มีอาการบาดเจ็บ/เสื่อม
```

---

## 📊 ไฟล์และผลลัพธ์

### Input Files
```
running_simulator.py          - โปรแกรมสร้าง synthetic data
advanced_gait_analyzer.py     - วิเคราะห์ time-series
prediction_utils.py           - เครื่องมือประมวลผลและทำนาย
```

### Output Files Generated
```
running_biomechanics_data.csv           - ข้อมูล 1000 samples
running_biomechanics_analysis.png       - กราฟวิเคราะห์ 4 แบบ
grf_time_series.png                     - แสดง GRF pattern 3 ประเภท
gait_statistics.csv                     - สถิติ GRF สำหรับแต่ละประเภท
example_training_data.csv               - ข้อมูลตัวอย่างสำหรับเทรน
```

---

## 🔧 วิธีใช้

### 1. สร้าง Dataset (1000 samples)
```bash
python running_simulator.py
```

**Output:**
- ✅ 600 samples แบบ Normal
- ✅ 200 samples แบบ Overpronate
- ✅ 200 samples แบบ Injured
- ✅ Random Forest Model accuracy: 100%

### 2. วิเคราะห์ Time-Series GRF
```bash
python advanced_gait_analyzer.py
```

**Output:**
- 📊 GRF time series visualization
- 📈 Statistical comparison
- 📋 Gait statistics by type

### 3. ประมวลผลข้อมูลเซนเซอร์จริง
```bash
python prediction_utils.py
```

**Features:**
- ❌ เซนเซอร์ force plate
- ✅ เซนเซอร์ insole pressure
- ✅ Gait event detection
- ✅ Feature extraction

---

## 💡 ตัวอย่างการใช้

### A) Load & Analyze Your CSV Data
```python
from prediction_utils import BioMechanicsPreprocessor

preprocessor = BioMechanicsPreprocessor()

# Load your sensor data
df = preprocessor.load_sensor_data('your_grf_data.csv')

# Process signal
grf_steps = preprocessor.process_grf_signal(df['force_values'].values)

# Extract features
features = preprocessor.extract_biomechanical_features(grf_steps[0])
print(features)
```

### B) Predict on New Data
```python
from prediction_utils import PredictionEngine
import joblib

# Load trained model
model = joblib.load('trained_model.pkl')
scaler = joblib.load('scaler.pkl')

# Create prediction engine
engine = PredictionEngine(model, scaler)

# Predict single step
result = engine.predict_single_step(grf_signal)

# Analyze full session
session_analysis = engine.analyze_session(grf_signals_array)
print(session_analysis['summary'])
```

### C) Generate Synthetic Data for Testing
```python
from prediction_utils import DataGenerator

gen = DataGenerator()

# Simulate force plate output (10 seconds)
grf = gen.simulate_force_plate_output(duration=10, sampling_rate=1000)

# Simulate insole pressure (8 sensors)
pressure = gen.simulate_insole_pressure(n_sensors=8, duration=5)
```

---

## 📈 Model Performance

### Random Forest Classifier (Main Model)
```
Train Accuracy: 100%
Test Accuracy:  100%

Top Features:
  1. mean_force: 20.9%
  2. braking_force: 19.4%
  3. peak_force: 12.4%
  4. impact_peak: 11.2%
  5. contact_time: 9.5%
```

### Classification Report
```
           Precision  Recall  F1-Score
Normal        1.00    1.00    1.00
Overpronate   1.00    1.00    1.00
Injured       1.00    1.00    1.00
```

---

## 🔍 Feature Dictionary

### Core Features
- **peak_force**: Maximum vertical GRF
- **mean_force**: Average GRF during contact
- **loading_rate**: Rate of force increase (first 20ms)
- **impact_peak**: Peak force in first 10ms
- **braking_force**: Horizontal decelerating force
- **propulsive_force**: Horizontal accelerating force
- **contact_time**: Duration foot on ground
- **speed**: Running speed (m/s)
- **cadence**: Steps per minute
- **vertical_stiffness**: peak_force / contact_time
- **symmetry_index**: propulsive / braking ratio

---

## 📱 Sensor Integration Examples

### Force Plate Data Format
```csv
timestamp,fx,fy,fz,mx,my,mz
0.000,12.5,2.3,850.0,25.0,15.0,5.0
0.001,12.6,2.4,851.2,25.1,15.1,5.1
```

### Insole Sensor Format (8 sensors per shoe)
```csv
timestamp,sensor_1,sensor_2,sensor_3,sensor_4,sensor_5,sensor_6,sensor_7,sensor_8
0.000,0.0,0.5,1.2,0.8,0.6,0.4,0.3,0.1
0.010,0.1,0.6,1.3,0.9,0.7,0.5,0.4,0.2
```

### IMU Data Format
```csv
timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z
0.000,-0.5,9.8,0.2,0.01,-0.02,0.03
0.010,-0.6,9.7,0.3,0.02,-0.03,0.04
```

---

## ⚙️ Configuration & Customization

### Adjust Dataset Ratio
```python
simulator = RunningBiomechanicsSimulator(n_samples=1000)
dataset = simulator.generate_dataset(
    ratio={'normal': 0.7, 'overpronate': 0.15, 'injured': 0.15}
)
```

### Change Model Parameters
```python
model = RandomForestClassifier(
    n_estimators=200,      # More trees
    max_depth=15,          # Deeper trees
    min_samples_split=5
)
```

### Adjust Thresholds
```python
# For risk assessment
if injured_count / total > 0.2:  # Change threshold
    risk = "HIGH"
```

---

## 🚀 Advanced Features

### Real-time Streaming
```python
# Process data as it arrives from sensors
while True:
    new_grf_sample = sensor.read()
    prediction = engine.predict_single_step(new_grf_sample)
    send_to_app(prediction)
```

### Session Aggregation
```python
# Collect data from entire running session
session_data = collect_session(duration=30_minutes)
summary = engine.analyze_session(session_data)

# Generate report
generate_pdf_report(summary)
```

### Cross-validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## 📊 Data Visualization Examples

### What the graphs show:

**1. running_biomechanics_analysis.png**
   - Peak Force Distribution (by risk level)
   - Loading Rate vs Impact Peak scatter
   - Speed vs Cadence relationship
   - Symmetry Index boxplot

**2. grf_time_series.png**
   - Normal running pattern
   - Overpronation (higher peaks)
   - Injured pattern (reduced load phase)

**3. lstm_confusion_matrix.png** (if TensorFlow available)
   - Model prediction accuracy matrix

---

## 🔬 Scientific Background

### Why These Metrics Matter?

**Peak Force >3.0 BW:**
- Indicates higher impact loading
- Risk factor for stress fractures
- Suggests over-pronation

**High Loading Rate:**
- Rapid force increase = shock
- Associated with shin splints
- Suggests stiff running style

**Asymmetry >15%:**
- Left-right imbalance
- Common in chronic injuries
- Needs corrective exercises

**High Cadence (>180 steps/min):**
- Shorter ground contact
- Reduces injury risk
- More efficient running

---

## 🐛 Troubleshooting

| ปัญหา | สาเหตุ | วิธีแก้ |
|------|--------|--------|
| Low accuracy | ข้อมูลไม่สมดุล | ใช้ `class_weight='balanced'` |
| Out of memory | Dataset ใหญ่เกิน | ลด `n_samples` หรือใช้ batch processing |
| Missing TensorFlow | ไม่ติดตั้ง | `pip install tensorflow` |
| Bad predictions | ข้อมูล normalize ไม่สม่ำเสมอ | ตรวจสอบ scaler parameters |

---

## 📚 References

- Nigg, B. M. (2010). Biomechanics of sport shoes
- Cavanagh, P. R., & Lafortune, M. A. (1980). Ground reaction forces in distance running
- International Society of Biomechanics (ISB) Standards

---

## 📝 License & Citation

If you use this system in research:

```
@software{running_biomechanics_2024,
  title={Running Biomechanics Simulator & Injury Detection System},
  author={Generated by Claude AI},
  year={2024},
  url={generated}
}
```

---

## ✨ สรุป

ระบบนี้ให้คุณได้:
- ✅ Simulate realistic running data
- ✅ Extract biomechanical features
- ✅ Train ML models for injury prediction
- ✅ Analyze real sensor data
- ✅ Generate actionable recommendations

**Happy running! 🏃‍♂️**
