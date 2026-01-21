import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# Go to your 'Soccer_Training_Results' folder and copy the name of the Master file
FILENAME = r'C:/Users/alext/PycharmProjects/Group2-AI-ML/results/soccer/Soccer_Training_Results/soccer_training_master_20260114_161701.csv'


try:
    df = pd.read_csv(FILENAME)
    print(f"Successfully loaded {len(df)} training runs.")
except FileNotFoundError:
    print(f"Could not find the file: {FILENAME}. Check your folder!")
    exit()

# Features (The inputs the SVR uses to learn patterns)
X = df[['batch_size', 'hidden_units', 'num_layers']]


# We map the display labels to the actual column names in our CSV
targets = {
    'Wall-Clock Time (Duration)': 'duration_sec',
    'Peak RAM Usage': 'peak_ram_kb',
    'Average RAM Usage': 'avg_ram_kb'
}

print("\n SVR results : ")

for label, col_name in targets.items():
    y = df[col_name]

    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    # Split: 80% to learn, 20% to test if it can guess correctly
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Train SVR (RBF kernel is best for non-linear hardware performance)
    model = SVR(kernel='rbf', C=10, epsilon=0.1)
    model.fit(X_train, y_train)


    preds_scaled = model.predict(X_test)

    # REVERSE the scaling to get real-world numbers (Seconds or KB)
    preds_final = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    actuals_final = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    mae = mean_absolute_error(actuals_final, preds_final)
    r2 = r2_score(actuals_final, preds_final)

    unit = "seconds" if "Time" in label else "KB"

    print(f"\nTarget: {label}")
    print(f"  Model Accuracy (R2 Score): {r2:.4f}")
    print(f"  Average Prediction Error (MAE): {mae:.2f} {unit}")

    # show an example
    print(f"  Example -> Predicted: {preds_final[0]:.2f} | Actual: {actuals_final[0]:.2f}")

print("\nAnalysis Finished.")
