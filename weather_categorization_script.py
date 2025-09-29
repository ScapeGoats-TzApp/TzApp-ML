# %%
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from lightgbm import LGBMClassifier

# %%
weather_df = pd.read_csv("weather_daily_2021_2022.csv")
weather_df.head(5)

# %%
weather_df.drop(columns=['latitude', 'longitude', 'date'], inplace=True, axis=1)
# weather_df.head(5)
weather_df.shape

# %%
weather_df.info()

# %%
weather_df.describe()

# %%
# Create categories for meteorological variables
print("Creating categories for meteorological variables...")

def create_categories(weather_df):
	'''Define 6 categories for weather'''

	conditions = [
		# SNOW
		(weather_df['temperature'] <= 0) & (weather_df['precipitation'] > 0.001),
		# HEAVY_RAIN
        (weather_df['precipitation'] >= 0.03),

        # LIGHT_RAIN
        (weather_df['precipitation'] >= 0.005) & (weather_df['precipitation'] < 0.03),

        # HOT
        (weather_df['temperature'] >= 20),

        # COLD
        (weather_df['temperature'] < 10) & (weather_df['precipitation'] < 0.005),

        # SUNNY
        (weather_df['temperature'] >= 10) & (weather_df['temperature'] < 20) & (weather_df['precipitation'] < 0.005)
	]

	choices = ['snow', 'heavy_rain', 'light_rain', 'hot', 'cold', 'sunny']
	weather_df['weather_category'] = np.select(conditions, choices, default='sunny')

	return weather_df

weather_df = create_categories(weather_df)
weather_df['weather_category'].value_counts()
# weather_df.head(5)

# %%
# FEATURE ENGINEERING
print("Feature engineering...")

def create_optimized_features(weather_df):

	# Physical formulas
    weather_df['wind_chill'] = 13.12 + 0.6215*weather_df['temperature'] - 11.37*(weather_df['wind']**0.16) + 0.3965*weather_df['temperature']*(weather_df['wind']**0.16)
    weather_df['feels_like'] = weather_df['temperature'] + 0.3 * weather_df['relative_humidity'] - 0.7 * weather_df['wind']

    # Important interactions
    weather_df['precip_humidity_interaction'] = weather_df['precipitation'] * weather_df['relative_humidity']
    weather_df['temp_pressure_interaction'] = weather_df['temperature'] * (weather_df['air_pressure'] / 1000)

    # Clear binary conditions
    weather_df['is_freezing'] = (weather_df['temperature'] <= 0).astype(int)
    weather_df['is_raining'] = (weather_df['precipitation'] > 0.005).astype(int)
    weather_df['is_windy'] = (weather_df['wind'] > 8).astype(int)
    weather_df['is_humid'] = (weather_df['relative_humidity'] > 2.0).astype(int)

    # Simplified altitude zone
    weather_df['is_high_altitude'] = (weather_df['altitude'] > 800).astype(int)

    return weather_df

weather_df = create_optimized_features(weather_df)
weather_df.info()


# %%
weather_df['weather_category'].value_counts()

# %%
# FEATURES SELECTION
print("Features selection...")

optimal_features = [
	# original features
	'temperature', 'precipitation', 'wind', 'relative_humidity', 'altitude', 'air_pressure',
	# engineered features
	'feels_like', 'precip_humidity_interaction', 'temp_pressure_interaction',
	# binary features
	'is_freezing', 'is_raining', 'is_windy', 'is_humid', 'is_high_altitude'
]

len(optimal_features)

# %%
# LABEL ENCODING
print("Label encoding...")
label = LabelEncoder()
weather_df['weather_encoded'] = label.fit_transform(weather_df['weather_category'])

weather_df.head(5)


# %%
# Verify distribution
print("\n📈 Distribution of categories:")
category_dist = weather_df['weather_category'].value_counts()
for category, count in category_dist.items():
    percentage = (count / len(weather_df)) * 100
    print(f"   {category}: {count} samples ({percentage:.2f}%)")

# %%
# Splitting dataset
print("\nSplitting dataset...")
X = weather_df[optimal_features]
y = weather_df['weather_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# %%
# MODEL CREATION
print("\nTraining LightGBM model...")

model = LGBMClassifier(
	n_estimators=800,
	max_depth=7,
	learning_rate=0.1,
	subsample=0.7,
	colsample_bytree=0.7,
	reg_alpha=0.3,
	reg_lambda=0.3,
	random_state=42,
	class_weight='balanced',
	min_child_samples=25,
	min_split_gain=0.01,
	n_jobs=-1,
	objective='multiclass',
	boosting_type='gbdt',
	metric='multi_logloss'
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='multi_logloss', callbacks=[
        lgb.early_stopping(50, verbose=0),   
        lgb.log_evaluation(100)        
    ])

print("Model training completed.")

# %%
# EVALUATION
print("\nEvaluating model...")

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("\n--- Training Set Evaluation ---")
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

# %%
accuracy_gap = train_accuracy - test_accuracy
print(f"Accuracy Gap (Train - Test): {accuracy_gap:.4f}")

if accuracy_gap < 0.05:
	print("The model is well-generalized.")
else:
	print("Warning: The model may be overfitting.")

print("\n--- Classification Report (Test Set) ---")
print(classification_report(y_test, y_pred_test, target_names=label.classes_))

cm = confusion_matrix(y_test, y_pred_test, labels=model.classes_)
cm_df = pd.DataFrame(cm, index=label.classes_, columns=label.classes_)
print("Confusion Matrix:")
print(cm_df)

# %%
def predict_weather_optimized(temperature, precipitation, wind, humidity, altitude, pressure):
    """
    Funcție optimizată pentru predicții în producție
    """
    # Calculează feature-urile derivate
    feels_like = temperature + 0.3 * humidity - 0.7 * wind
    precip_humidity_interaction = precipitation * humidity
    temp_pressure_interaction = temperature * (pressure / 1000)
    
    # Condiții binare
    is_freezing = 1 if temperature <= 0 else 0
    is_raining = 1 if precipitation > 0.005 else 0
    is_windy = 1 if wind > 8 else 0
    is_humid = 1 if humidity > 2.0 else 0
    is_high_altitude = 1 if altitude > 800 else 0
    
    # Feature vector în ordinea corectă
    features = np.array([[
        temperature, precipitation, wind, humidity, altitude, pressure,
        feels_like, precip_humidity_interaction, temp_pressure_interaction,
        is_freezing, is_raining, is_windy, is_humid, is_high_altitude
    ]])
    
    # Predicție
    prediction_encoded = model.predict(features)[0]
    prediction = label.inverse_transform([prediction_encoded])[0]
    
    return prediction

# %%
print("🧪 20 MANUAL TESTS WITH VALIDATION")
print("=" * 60)

test_cases_manual = [
	# HOT - Warm conditions
	(25, 0.001, 3, 1.5, 200, 1013, "hot", "Warm and dry"),
	(28, 0.0001, 2, 1.3, 150, 1010, "hot", "Very hot"),
	(22, 0.0005, 4, 1.6, 300, 1012, "hot", "Moderate warmth"),
    
	# SNOW - Below zero temperatures with precipitation
	(-5, 0.02, 8, 2.1, 800, 980, "snow", "Moderate snow"),
	(-2, 0.015, 6, 2.0, 500, 990, "snow", "Light snow"),
	(-8, 0.025, 10, 2.2, 600, 970, "snow", "Blizzard"),

	# HEAVY_RAIN - Abundant precipitation
	(15, 0.04, 12, 2.3, 300, 1002, "heavy_rain", "Torrential rain"),
	(18, 0.035, 10, 2.4, 250, 1005, "heavy_rain", "Heavy rain"),
	(12, 0.032, 8, 2.2, 350, 1008, "heavy_rain", "Constant rain"),

	# LIGHT_RAIN - Light/moderate precipitation
	(10, 0.01, 5, 2.0, 200, 1010, "light_rain", "Moderate rain"),
	(8, 0.008, 4, 1.9, 180, 1012, "light_rain", "Drizzle"),
	(14, 0.015, 6, 2.1, 220, 1008, "light_rain", "Light rain"),

	# COLD - Cold temperatures without precipitation
	(-3, 0.0001, 5, 1.7, 300, 1005, "cold", "Cold and clear"),
	(5, 0.0002, 6, 1.6, 400, 1008, "cold", "Cool"),
	(-10, 0.0003, 7, 1.8, 600, 985, "cold", "Intense cold"),

	# NORMAL - Normal/temperate conditions
	(16, 0.0001, 3, 1.4, 100, 1013, "normal", "Normal/Sunny"),
	(19, 0.0003, 4, 1.5, 150, 1011, "normal", "Pleasant and clear"),
	(17, 0.0002, 5, 1.6, 120, 1012, "normal", "Normal with light wind"),

	# Edge cases
	(0, 0.0005, 3, 1.8, 200, 1008, "cold", "At freezing point without precipitation"),
	(0, 0.002, 4, 2.0, 300, 1005, "snow", "At freezing point with precipitation")
]

correct_predictions = 0
total_tests = len(test_cases_manual)

print("🧪 REZULTATE TESTE:")
print("-" * 60)

for i, (temp, precip, wind, humid, alt, press, expected, description) in enumerate(test_cases_manual, 1):
    try:
        prediction = predict_weather_optimized(temp, precip, wind, humid, alt, press)
        is_correct = prediction == expected
        
        if is_correct:
            correct_predictions += 1
        
        status_emoji = "✅" if is_correct else "❌"
        
        print(f"Test {i:2d}: {status_emoji} {description}")
        print(f"   🌡️  {temp}°C | 💧 {precip:.3f} precip | 💨 {wind} vânt | 💦 {humid} umid")
        print(f"   🎯 Așteptat: {expected}")
        print(f"   🤖 Predicted: {prediction}")
        
        if not is_correct:
            print(f"   ⚠️  DISCREPANȚĂ!")
        print()
        
    except Exception as e:
        print(f"Test {i:2d}: ❌ EROARE - {description}")
        print(f"   💥 Eroare: {e}")
        print()



# %%
accuracy_final = (correct_predictions / total_tests) * 100

print("FINAL PERFORMANCE SUMMARY")
print("=" * 60)
print(f"TEST SET ACCURACY: {test_accuracy:.4f}")
print(f"MANUAL TEST ACCURACY: {accuracy_final:.1f}%")
print(f"OVERFITTING (gap): {accuracy_gap:.4f}")
print(f"CLASSES: {len(label.classes_)} clear categories")
print(f"FEATURES: {len(optimal_features)} (optimized)")

if accuracy_gap < 0.02 and accuracy_final >= 85:
	print("\nSUCCESS! Model is optimized and generalizes excellently!")
elif accuracy_gap < 0.05 and accuracy_final >= 80:
	print("\nGOOD! Model has good performance with minimal overfitting!")
else:
	print("\nACCEPTABLE! Model works but can be improved!")

# %%
# Evaluation message based on accuracy
print("\nEVALUATION BASED ON ACCURACY")

accuracy = (correct_predictions / total_tests) * 100

if accuracy == 100:
	print("PERFECT! All predictions are correct!")
elif accuracy >= 90:
	print("EXCELLENT! The model is very accurate!")
elif accuracy >= 80:
	print("VERY GOOD! Solid performance!")
elif accuracy >= 70:
	print("GOOD! The model works decently.")
elif accuracy >= 60:
	print("ACCEPTABLE! There is room for improvement.")
else:
	print("NEEDS IMPROVEMENT! Accuracy is too low.")

# %%
# Manual test with new data
print("\n🧪 MANUAL TEST WITH NEW DATA\n")

new_data = [
	(30, 0.0001, 3, 1.2, 100, 1013),  # Hot and dry
]

for i, (temp, precip, wind, humid, alt, press) in enumerate(new_data, 1):
	try:
		prediction = predict_weather_optimized(temp, precip, wind, humid, alt, press)
		print(f"New Data {i}: 🌡️ {temp}°C | 💧 {precip:.4f} precip | 💨 {wind} vânt | 💦 {humid} umid | 🏔️ {alt} alt | � pressure {press}")
		print(f"   🤖 Predicted Weather Category: {prediction}\n")
	except Exception as e:
		print(f"New Data {i}: ❌ EROARE")
		print(f"   💥 Eroare: {e}\n")


