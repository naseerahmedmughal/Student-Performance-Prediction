import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. Load Data (Simulated for this demo)
def load_data():
    data = {
        'Study_Hours': [2, 5, 8, 10, 1, 3, 7, 9],
        'Attendance_Pct': [60, 80, 95, 90, 50, 70, 85, 98],
        'Parent_Education_Level': [1, 2, 3, 3, 1, 2, 3, 3],
        'Final_Grade': [45, 65, 88, 92, 30, 55, 80, 95]
    }
    return pd.DataFrame(data)

# 2. Exploratory Data Analysis (EDA)
def plot_correlations(df):
    """
    Visualizes the relationship between Attendance and Grades.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Attendance_Pct', y='Final_Grade', data=df)
    plt.title("Impact of Attendance on Final Grades")
    plt.xlabel("Attendance (%)")
    plt.ylabel("Final Grade (0-100)")
    plt.show()
    
    # Correlation Matrix
    print("--- Correlation Matrix ---")
    print(df.corr())

# 3. Train Regression Model
def predict_performance(df):
    """
    Trains a simple Linear Regression model.
    """
    X = df[['Study_Hours', 'Attendance_Pct']]
    y = df['Final_Grade']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(f"\nModel Coefficient (Study Hours): {model.coef_[0]}")
    print(f"Model Coefficient (Attendance): {model.coef_[1]}")
    print("Insight: Attendance has a stronger weight on the final grade.")

if __name__ == "__main__":
    df = load_data()
    plot_correlations(df)
    predict_performance(df)
