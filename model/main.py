import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import optuna

df = pd.read_csv('Sales-Data.csv')

df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Week'] = df['Order Date'].dt.isocalendar().week
df.columns = df.columns.str.strip()
grouped_data = df.groupby(['City', 'Week', 'Product']).agg({'Quantity Ordered': 'sum'}).reset_index()

grouped_data['City'] = grouped_data['City'].str.strip()


onehot_encoder_city = OneHotEncoder(sparse_output=False)
onehot_encoder_product = OneHotEncoder(sparse_output=False)

encoded_cities = onehot_encoder_city.fit_transform(grouped_data[['City']])
encoded_products = onehot_encoder_product.fit_transform(grouped_data[['Product']])

encoded_cities_df = pd.DataFrame(encoded_cities, columns=onehot_encoder_city.get_feature_names_out(['City']))
encoded_products_df = pd.DataFrame(encoded_products, columns=onehot_encoder_product.get_feature_names_out(['Product']))

encoded_cities_df.columns = encoded_cities_df.columns.str.replace(' ', '_', regex=False)
encoded_products_df.columns = encoded_products_df.columns.str.replace(' ', '_', regex=False)

grouped_data_encoded = pd.concat([grouped_data, encoded_cities_df, encoded_products_df], axis=1)

grouped_data_encoded.drop(columns=['City', 'Product'], inplace=True)

joblib.dump(onehot_encoder_city, 'city_onehot_encoder.pkl')
joblib.dump(onehot_encoder_product, 'product_onehot_encoder.pkl')

grouped_data_encoded['Quantity Ordered'] = grouped_data_encoded['Quantity Ordered'].astype(float)

# na zakresie 1-6 uczymy się na historii z 5 tygodni wstecz ile bylo trzeba zamowic. Tutaj zmieniamy do ilu wstecz patrzymy
for i in range(1, 6):
    grouped_data_encoded[f'Quantity_Ordered_T-{i}'] = grouped_data_encoded['Quantity Ordered'].shift(i)

grouped_data_encoded = grouped_data_encoded.dropna()
X = grouped_data_encoded.drop(['Quantity Ordered'], axis=1)
y = grouped_data_encoded['Quantity Ordered']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
grouped_data_encoded.to_csv('nazwa_pliku.csv', index=False, sep=',', encoding='utf-8')
numeric_features = X.select_dtypes(include=['float64']).columns.tolist()
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
    ])


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.005, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 10, 50),
        'max_depth': trial.suggest_int('max_depth', 5, 25),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 70),
        'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0)
    }

    model = lgb.LGBMRegressor(**params, random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return mse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("Best parameters found for LightGBM:")
print(study.best_params)

best_params = study.best_params
best_model = lgb.LGBMRegressor(**best_params, random_state=42)

pipeline_best = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', best_model)
])

pipeline_best.fit(X_train, y_train)
y_pred_best = pipeline_best.predict(X_test)

mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

results = [{
    'Model': 'LightGBM',
    'Params': study.best_params,
    'Mean Squared Error': mse_best,
    'R2 Score': r2_best
}]

with open(r'results.txt', 'a') as f:
    for result in results:
        f.write(f"Model: {result['Model']}_1\n")
        f.write(f"Params: {result['Params']}\n")
        f.write(f"Mean Squared Error: {result['Mean Squared Error']:.4f}\n")
        f.write(f"R2 Score: {result['R2 Score']:.4f}\n\n")

print("Results saved to results.txt")

joblib.dump(pipeline_best, 'best_model.pkl')
print("Model saved to best_model.pkl")

selected_city = 'atlanta'
selected_product = 'AAA Batteries (4-pack)'

filter_mask = (grouped_data_encoded['City_Atlanta'] == 1) & (grouped_data_encoded['Product_AAA_Batteries_(4-pack)'] == 1)
filtered_data = grouped_data_encoded[filter_mask]

filtered_X = filtered_data.drop(columns=['Quantity Ordered'])
predicted_quantities = pipeline_best.predict(filtered_X)

plt.figure(figsize=(12, 6))
plt.plot(filtered_data['Week'], filtered_data['Quantity Ordered'], label='Rzeczywista Sprzedaż', marker='o')
plt.plot(filtered_data['Week'], predicted_quantities, label='Przewidziana Sprzedaż', marker='x')
plt.title(f'Porównanie Sprzedaży dla {selected_product} w {selected_city}')
plt.xlabel('Tydzień')
plt.ylabel('Ilość Zamówionych Produktów')
plt.legend()
plt.grid()
plt.show()
