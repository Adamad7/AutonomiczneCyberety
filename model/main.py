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

from flask import Flask, request, render_template, redirect, url_for
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        df = pd.read_csv(file)
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Week'] = df['Order Date'].dt.isocalendar().week
        df.columns = df.columns.str.strip()
        grouped_data = df.groupby(['City', 'Week', 'Product']).agg({'Quantity Ordered': 'sum'}).reset_index()
        grouped_data['City'] = grouped_data['City'].str.strip()

        # Save the processed data to a CSV file
        grouped_data.to_csv('processed_data.csv', index=False)

        # Get the unique weeks
        weeks = grouped_data['Week'].unique()

        return render_template('select_week.html', weeks=weeks)



# def train_model():
#     start_week = int(request.form['start_week'])
#     end_week = int(request.form['end_week'])
#     selected_weeks = list(range(start_week, end_week + 1))
#
#     # Load the processed data
#     grouped_data = pd.read_csv('processed_data.csv')
#
#     # Filter the data based on selected weeks
#     grouped_data = grouped_data[grouped_data['Week'].isin(selected_weeks)]
#
#     print(grouped_data)
#
#     if grouped_data.empty:
#         return "No data available for the selected weeks."
#
#     onehot_encoder_city = OneHotEncoder(sparse_output=False)
#     onehot_encoder_product = OneHotEncoder(sparse_output=False)
#
#     encoded_cities = onehot_encoder_city.fit_transform(grouped_data[['City']])
#     encoded_products = onehot_encoder_product.fit_transform(grouped_data[['Product']])
#
#     encoded_cities_df = pd.DataFrame(encoded_cities, columns=onehot_encoder_city.get_feature_names_out(['City']))
#     encoded_products_df = pd.DataFrame(encoded_products,
#                                        columns=onehot_encoder_product.get_feature_names_out(['Product']))
#
#     encoded_cities_df.columns = encoded_cities_df.columns.str.replace(' ', '_', regex=False)
#     encoded_products_df.columns = encoded_products_df.columns.str.replace(' ', '_', regex=False)
#
#     grouped_data_encoded = pd.concat([grouped_data, encoded_cities_df, encoded_products_df], axis=1)
#
#     grouped_data_encoded.drop(columns=['City', 'Product'], inplace=True)
#
#     joblib.dump(onehot_encoder_city, 'city_onehot_encoder.pkl')
#     joblib.dump(onehot_encoder_product, 'product_onehot_encoder.pkl')
#
#     grouped_data_encoded['Quantity Ordered'] = grouped_data_encoded['Quantity Ordered'].astype(float)
#
#
#     for i in range(1, end_week - start_week):
#         grouped_data_encoded[f'Quantity_Ordered_T-{i}'] = grouped_data_encoded['Quantity Ordered'].shift(i)
#
#     grouped_data_encoded = grouped_data_encoded.dropna()
#     X = grouped_data_encoded.drop(['Quantity Ordered'], axis=1)
#     y = grouped_data_encoded['Quantity Ordered']
#
#     if X.empty or y.empty:
#         return "No data available after preprocessing."
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
#     if X_train.empty or y_train.empty or X_test.empty or y_test.empty:
#         return "Training or testing data is empty."
#
#     numeric_features = X.select_dtypes(include=['float64']).columns.tolist()
#     numeric_transformer = Pipeline(steps=[
#         ('scaler', StandardScaler())
#     ])
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numeric_transformer, numeric_features),
#         ])
#
#     def objective(trial):
#         params = {
#             'n_estimators': trial.suggest_int('n_estimators', 50, 300),
#             'learning_rate': trial.suggest_uniform('learning_rate', 0.005, 0.3),
#             'num_leaves': trial.suggest_int('num_leaves', 10, 50),
#             'max_depth': trial.suggest_int('max_depth', 5, 25),
#             'min_child_samples': trial.suggest_int('min_child_samples', 10, 70),
#             'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
#             'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
#             'verbose': -1
#         }
#
#         model = lgb.LGBMRegressor(**params, random_state=42, )
#
#
#         pipeline = Pipeline(steps=[
#             ('preprocessor', preprocessor),
#             ('regressor', model)
#         ])
#
#         pipeline.fit(X_train, y_train)
#         y_pred = pipeline.predict(X_test)
#         mse = mean_squared_error(y_test, y_pred)
#
#         return mse
#
#     study = optuna.create_study(direction='minimize')
#     study.optimize(objective, n_trials=20)
#
#     best_params = study.best_params
#     best_model = lgb.LGBMRegressor(**best_params, random_state=42)
#
#     pipeline_best = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('regressor', best_model)
#     ])
#
#     pipeline_best.fit(X_train, y_train)
#
#     next_week = max(grouped_data_encoded['Week']) + 1
#     grouped_data_encoded['Week'] = next_week
#
#     grouped_data_encoded = grouped_data_encoded.drop(columns=['Quantity Ordered'])
#     predicted_quantities = pipeline_best.predict(grouped_data_encoded)
#
#     predictions_df = grouped_data[['City', 'Product']].drop_duplicates().reset_index(drop=True)
#     predictions_df['Predicted Quantity Ordered'] = predicted_quantities[:len(predictions_df)]
#
#     city_tables = []
#     for city, city_df in predictions_df.groupby('City'):
#         city_tables.append((city, city_df.drop(columns=['City']).to_html(classes='table table-striped', header="true")))
#
#     return render_template('predictions.html', city_tables=city_tables)
#
#
# def train_model2():
#     start_week = 1
#     end_weeek = 10
#
#     start_week = int(request.form['start_week'])
#     end_week = int(request.form['end_week'])
#     selected_weeks = list(range(start_week, end_week + 1))
#
#     # Load the processed data
#     grouped_data = pd.read_csv('processed_data.csv')
#
#     # Filter the data based on selected weeks
#     grouped_data = grouped_data[grouped_data['Week'].isin(selected_weeks)]
#
#     onehot_encoder_city = OneHotEncoder(sparse_output=False)
#     onehot_encoder_product = OneHotEncoder(sparse_output=False)
#
#     encoded_cities = onehot_encoder_city.fit_transform(grouped_data[['City']])
#     encoded_products = onehot_encoder_product.fit_transform(grouped_data[['Product']])
#
#     encoded_cities_df = pd.DataFrame(encoded_cities, columns=onehot_encoder_city.get_feature_names_out(['City']))
#     encoded_products_df = pd.DataFrame(encoded_products,
#                                        columns=onehot_encoder_product.get_feature_names_out(['Product']))
#
#     encoded_cities_df.columns = encoded_cities_df.columns.str.replace(' ', '_', regex=False)
#     encoded_products_df.columns = encoded_products_df.columns.str.replace(' ', '_', regex=False)
#
#     grouped_data_encoded = pd.concat([grouped_data, encoded_cities_df, encoded_products_df], axis=1)
#
#     grouped_data_encoded.drop(columns=['City', 'Product'], inplace=True)
#
#     joblib.dump(onehot_encoder_city, 'city_onehot_encoder.pkl')
#     joblib.dump(onehot_encoder_product, 'product_onehot_encoder.pkl')
#
#     grouped_data_encoded['Quantity Ordered'] = grouped_data_encoded['Quantity Ordered'].astype(float)
#
#     # na zakresie 1-6 uczymy się na historii z 5 tygodni wstecz ile bylo trzeba zamowic. Tutaj zmieniamy do ilu wstecz patrzymy
#     for i in range(1, end_weeek - start_week):
#         grouped_data_encoded[f'Quantity_Ordered_T-{i}'] = grouped_data_encoded['Quantity Ordered'].shift(i)
#
#     grouped_data_encoded = grouped_data_encoded.dropna()
#     X = grouped_data_encoded.drop(['Quantity Ordered'], axis=1)
#     y = grouped_data_encoded['Quantity Ordered']
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#     grouped_data_encoded.to_csv('nazwa_pliku.csv', index=False, sep=',', encoding='utf-8')
#     numeric_features = X.select_dtypes(include=['float64']).columns.tolist()
#     numeric_transformer = Pipeline(steps=[
#         ('scaler', StandardScaler())
#     ])
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numeric_transformer, numeric_features),
#         ])
#
#     def objective(trial):
#         params = {
#             'n_estimators': trial.suggest_int('n_estimators', 50, 300),
#             'learning_rate': trial.suggest_uniform('learning_rate', 0.005, 0.3),
#             'num_leaves': trial.suggest_int('num_leaves', 10, 50),
#             'max_depth': trial.suggest_int('max_depth', 5, 25),
#             'min_child_samples': trial.suggest_int('min_child_samples', 10, 70),
#             'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
#             'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
#             'verbose': -1
#         }
#
#         model = lgb.LGBMRegressor(**params, random_state=42)
#
#         pipeline = Pipeline(steps=[
#             ('preprocessor', preprocessor),
#             ('regressor', model)
#         ])
#
#         pipeline.fit(X_train, y_train)
#         y_pred = pipeline.predict(X_test)
#         mse = mean_squared_error(y_test, y_pred)
#
#         return mse
#
#     study = optuna.create_study(direction='minimize')
#     study.optimize(objective, n_trials=20)
#
#     best_params = study.best_params
#     best_model = lgb.LGBMRegressor(**best_params, random_state=42)
#
#     pipeline_best = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('regressor', best_model)
#     ])
#
#     pipeline_best.fit(X_train, y_train)
#
#     predicted_quantities = pipeline_best.predict(grouped_data_encoded.drop(columns=['Quantity Ordered']))
#     print(predicted_quantities)
#     # Zmienna, która określa tydzień dla predykcji
#     week_pred = end_weeek + 1
#
#     selected_city = 'atlanta'
#     selected_product = 'AAA Batteries (4-pack)'
#     filter_mask = (grouped_data_encoded['City_Atlanta'] == 1) & (
#             grouped_data_encoded['Product_AAA_Batteries_(4-pack)'] == 1)
#     filtered_data = grouped_data_encoded[filter_mask]
#
#     if week_pred <= filtered_data['Week'].max() + 1:
#         relevant_data = filtered_data[filtered_data['Week'] < week_pred].copy()
#
#         # Tworzymy zestaw danych do predykcji dla wybranego tygodnia
#         new_data = relevant_data[relevant_data['Week'] == relevant_data['Week'].max()].copy()
#
#         # Ustawiamy nowy tydzień jako 'week_pred'
#         new_data['Week'] = week_pred
#
#         # Zaktualizuj przesunięcia danych, np. T-1, T-2, itp.
#         for i in range(1, 6):  # Zakładamy 5 przesuniętych tygodni
#             new_data[f'Quantity_Ordered_T-{i}'] = relevant_data[f'Quantity_Ordered_T-{i}'].values[-1]
#
#         # Przewidujemy dla danego tygodnia
#         new_data = new_data.drop(columns=['Quantity Ordered'])  # Usuwamy rzeczywiste wartości sprzedaży
#         predicted_quantity = pipeline_best.predict(new_data)
#
#         print(
#             f"Przewidziana ilość zamówień dla {selected_product} w {selected_city} na tydzień {week_pred}: {predicted_quantity[0]}")
#
#
#         predictions_df = grouped_data[['City', 'Product']].drop_duplicates().reset_index(drop=True)
#         predictions_df['Predicted Quantity Ordered'] = predicted_quantities[:len(predictions_df)]
#
#         city_tables = []
#         for city, city_df in predictions_df.groupby('City'):
#             city_tables.append(
#                 (city, city_df.drop(columns=['City']).to_html(classes='table table-striped', header="true")))
#
#         return render_template('predictions.html', city_tables=city_tables)
#
#     else:
#         print(
#             f"Brak wystarczających danych, aby przewidzieć tydzień {week_pred}. Upewnij się, że dostępne są dane historyczne.")

@app.route('/train', methods=['POST'])
def train_model3():


    start_week = int(request.form['start_week'])
    end_week = int(request.form['end_week'])
    selected_weeks = list(range(start_week, end_week + 1))
    grouped_data = pd.read_csv('processed_data.csv')

    onehot_encoder_city = OneHotEncoder(sparse_output=False)
    onehot_encoder_product = OneHotEncoder(sparse_output=False)

    encoded_cities = onehot_encoder_city.fit_transform(grouped_data[['City']])
    encoded_products = onehot_encoder_product.fit_transform(grouped_data[['Product']])

    encoded_cities_df = pd.DataFrame(encoded_cities, columns=onehot_encoder_city.get_feature_names_out(['City']))
    encoded_products_df = pd.DataFrame(encoded_products,
                                       columns=onehot_encoder_product.get_feature_names_out(['Product']))

    encoded_cities_df.columns = encoded_cities_df.columns.str.replace(' ', '_', regex=False)
    encoded_products_df.columns = encoded_products_df.columns.str.replace(' ', '_', regex=False)
    # print(encoded_cities_df)
    grouped_data_encoded = pd.concat([grouped_data, encoded_cities_df, encoded_products_df], axis=1)
    # print(grouped_data_encoded)
    grouped_data_encoded.drop(columns=['City', 'Product'], inplace=True)

    grouped_data_encoded = grouped_data_encoded[grouped_data_encoded['Week'].isin(selected_weeks)]

    # grouped_data_encoded = grouped_data_encoded[grouped_data_encoded['Week'].isin(range(start_week, end_week + 1))]

    grouped_data_encoded['Quantity Ordered'] = grouped_data_encoded['Quantity Ordered'].astype(float)

    # na zakresie 1-6 uczymy się na historii z 5 tygodni wstecz ile bylo trzeba zamowic. Tutaj zmieniamy do ilu wstecz patrzymy
    for i in range(1, 6):
        grouped_data_encoded[f'Quantity_Ordered_T-{i}'] = grouped_data_encoded['Quantity Ordered'].shift(i)

    grouped_data_encoded = grouped_data_encoded.dropna()
    X = grouped_data_encoded.drop(['Quantity Ordered'], axis=1)
    y = grouped_data_encoded['Quantity Ordered']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'verbose': -1
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

    best_params = study.best_params
    best_model = lgb.LGBMRegressor(**best_params, random_state=42)

    pipeline_best = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', best_model)
    ])

    pipeline_best.fit(X_train, y_train)

    import os

    # Zmienna, która określa tydzień dla predykcji
    week_pred = end_week + 1

    # Wyodrębniamy wszystkie unikalne miasta i produkty z danych
    city_columns = [col for col in grouped_data_encoded.columns if col.startswith('City_')]
    product_columns = [col for col in grouped_data_encoded.columns if col.startswith('Product_')]

    # Ścieżka do pliku, w którym zapisujemy wyniki
    report_file_path = 'predicted_sales_report.txt'

    # Tworzymy nowy plik i zapisujemy nagłówki
    with open(report_file_path, 'w') as file:
        file.write(f'Przewidywania na tydzień: {week_pred}\n')
        file.write('Miasto\tProdukt\tPrzewidziana Ilość Zamówień\n')

        # Iterujemy po każdym mieście
        for city_col in city_columns:
            city_name = city_col.replace('City_', '').replace('_', ' ')

            # Iterujemy po każdym produkcie
            for product_col in product_columns:
                product_name = product_col.replace('Product_', '').replace('_', ' ')

                # Tworzymy maskę dla wybranego miasta i produktu
                filter_mask = (grouped_data_encoded[city_col] == 1) & (grouped_data_encoded[product_col] == 1)
                filtered_data = grouped_data_encoded[filter_mask]

                # Sprawdzamy, czy dane istnieją dla danego miasta i produktu
                if len(filtered_data) > 0 and week_pred <= filtered_data['Week'].max() + 1:
                    # Filtrujemy dane dla tygodni poprzedzających week_pred
                    relevant_data = filtered_data[filtered_data['Week'] < week_pred].copy()

                    # Tworzymy zestaw danych do predykcji dla wybranego tygodnia
                    new_data = relevant_data[relevant_data['Week'] == relevant_data['Week'].max()].copy()

                    # Ustawiamy nowy tydzień jako 'week_pred'
                    new_data['Week'] = week_pred

                    # Zaktualizuj przesunięcia danych, np. T-1, T-2, itp.
                    for i in range(1, 6):  # Zakładamy 5 przesuniętych tygodni
                        if f'Quantity_Ordered_T-{i}' in relevant_data.columns:
                            new_data[f'Quantity_Ordered_T-{i}'] = relevant_data[f'Quantity_Ordered_T-{i}'].values[-1]

                    # Przewidujemy dla danego tygodnia
                    new_data = new_data.drop(columns=['Quantity Ordered'],
                                             errors='ignore')  # Usuwamy rzeczywiste wartości sprzedaży, jeśli są
                    predicted_quantity = pipeline_best.predict(new_data)[0]

                    # Zapisujemy wynik do pliku
                    file.write(f'{city_name}\t{product_name}\t{predicted_quantity:.2f}\n')

                    print(
                        f"Przewidziana ilość zamówień dla {product_name} w {city_name} na tydzień {week_pred}: {predicted_quantity:.2f}")
                else:
                    print(f"Brak wystarczających danych dla {product_name} w {city_name} na tydzień {week_pred}.")

    print(f"\nRaport został zapisany do pliku: {os.path.abspath(report_file_path)}")
    with open(report_file_path, 'r') as file:
        lines = file.readlines()

    predictions = []
    for line in lines[2:]:  # Skip the first two lines (header)
        city, product, quantity = line.strip().split('\t')
        predictions.append({'city': city, 'product': product, 'quantity': float(quantity)})

    return render_template('predictions.html', predictions=predictions)

def _test():
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



def _test2():
    df = pd.read_csv('Sales-Data.csv')
    start_week = 1
    end_weeek = 10

    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Week'] = df['Order Date'].dt.isocalendar().week
    df.columns = df.columns.str.strip()
    grouped_data = df.groupby(['City', 'Week', 'Product']).agg({'Quantity Ordered': 'sum'}).reset_index()

    grouped_data['City'] = grouped_data['City'].str.strip()
    grouped_data = grouped_data[grouped_data['Week'].isin(range(start_week, end_weeek))]


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
    for i in range(1, end_weeek):
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
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'verbose': -1
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


    best_params = study.best_params
    best_model = lgb.LGBMRegressor(**best_params, random_state=42)

    pipeline_best = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', best_model)
    ])

    pipeline_best.fit(X_train, y_train)




    predicted_quantities = pipeline_best.predict(grouped_data_encoded.drop(columns=['Quantity Ordered']))
    print(predicted_quantities)
    predictions_df = grouped_data_encoded.copy()
    predictions_df['Predicted Quantity Ordered'] = predicted_quantities

    # Print the DataFrame with column names
    print(predictions_df)
    # filter_mask = (predicted_quantities['City_Atlanta'] == 1)
    # filtered_data = predicted_quantities[filter_mask]
    # print(filtered_data)


def test3():
    df = pd.read_csv('Sales-Data.csv')
    start_week = 1
    end_weeek = 10

    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Week'] = df['Order Date'].dt.isocalendar().week
    df.columns = df.columns.str.strip()
    grouped_data = df.groupby(['City', 'Week', 'Product']).agg({'Quantity Ordered': 'sum'}).reset_index()

    grouped_data['City'] = grouped_data['City'].str.strip()
    grouped_data = grouped_data[grouped_data['Week'].isin(range(start_week, end_weeek))]

    onehot_encoder_city = OneHotEncoder(sparse_output=False)
    onehot_encoder_product = OneHotEncoder(sparse_output=False)

    encoded_cities = onehot_encoder_city.fit_transform(grouped_data[['City']])
    encoded_products = onehot_encoder_product.fit_transform(grouped_data[['Product']])

    encoded_cities_df = pd.DataFrame(encoded_cities, columns=onehot_encoder_city.get_feature_names_out(['City']))
    encoded_products_df = pd.DataFrame(encoded_products,
                                       columns=onehot_encoder_product.get_feature_names_out(['Product']))

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
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'verbose': -1
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

    best_params = study.best_params
    best_model = lgb.LGBMRegressor(**best_params, random_state=42)

    pipeline_best = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', best_model)
    ])

    pipeline_best.fit(X_train, y_train)

    predicted_quantities = pipeline_best.predict(grouped_data_encoded.drop(columns=['Quantity Ordered']))
    print(predicted_quantities)
    # Zmienna, która określa tydzień dla predykcji
    week_pred = end_weeek + 1



    selected_city = 'seattle'
    selected_product = '34in Ultrawide Monitor'
    filter_mask = (grouped_data_encoded['City_Seattle'] == 1) & (
                grouped_data_encoded['Product_34in_Ultrawide_Monitor'] == 1)
    filtered_data = grouped_data_encoded[filter_mask]

    if week_pred <= filtered_data['Week'].max() + 1:
        relevant_data = filtered_data[filtered_data['Week'] < week_pred].copy()

        # Tworzymy zestaw danych do predykcji dla wybranego tygodnia
        new_data = relevant_data[relevant_data['Week'] == relevant_data['Week'].max()].copy()

        # Ustawiamy nowy tydzień jako 'week_pred'
        new_data['Week'] = week_pred

        # Zaktualizuj przesunięcia danych, np. T-1, T-2, itp.
        for i in range(1, 6):  # Zakładamy 5 przesuniętych tygodni
            new_data[f'Quantity_Ordered_T-{i}'] = relevant_data[f'Quantity_Ordered_T-{i}'].values[-1]

        # Przewidujemy dla danego tygodnia
        new_data = new_data.drop(columns=['Quantity Ordered'])  # Usuwamy rzeczywiste wartości sprzedaży
        predicted_quantity = pipeline_best.predict(new_data)

        print(
            f"Przewidziana ilość zamówień dla {selected_product} w {selected_city} na tydzień {week_pred}: {predicted_quantity[0]}")

        # Rysowanie wykresu: Rzeczywista sprzedaż vs przewidywana sprzedaż
        plt.figure(figsize=(12, 6))

        # Rysowanie rzeczywistej sprzedaży dla wcześniejszych tygodni
        plt.plot(relevant_data['Week'], relevant_data['Quantity Ordered'], label='Rzeczywista Sprzedaż', marker='o')

        # Dodaj przewidywaną sprzedaż dla tygodnia week_pred
        plt.plot(week_pred, predicted_quantity, label='Przewidziana Sprzedaż', marker='x', color='red', markersize=10)

        plt.title(f'Porównanie Sprzedaży dla {selected_product} w {selected_city} (Predykcja dla tygodnia {week_pred})')
        plt.xlabel('Tydzień')
        plt.ylabel('Ilość Zamówionych Produktów')
        plt.legend()
        plt.grid()
        plt.show()

    else:
        print(
            f"Brak wystarczających danych, aby przewidzieć tydzień {week_pred}. Upewnij się, że dostępne są dane historyczne.")

def test4():
    df = pd.read_csv('Sales-Data.csv')
    start_week = 1
    end_week = 15

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
    encoded_products_df = pd.DataFrame(encoded_products,
                                       columns=onehot_encoder_product.get_feature_names_out(['Product']))

    encoded_cities_df.columns = encoded_cities_df.columns.str.replace(' ', '_', regex=False)
    encoded_products_df.columns = encoded_products_df.columns.str.replace(' ', '_', regex=False)
    print(encoded_cities_df)
    grouped_data_encoded = pd.concat([grouped_data, encoded_cities_df, encoded_products_df], axis=1)
    print(grouped_data_encoded)
    grouped_data_encoded.drop(columns=['City', 'Product'], inplace=True)

    grouped_data_encoded = grouped_data_encoded[grouped_data_encoded['Week'].isin(range(start_week, end_week+1))]

    grouped_data_encoded['Quantity Ordered'] = grouped_data_encoded['Quantity Ordered'].astype(float)


    # na zakresie 1-6 uczymy się na historii z 5 tygodni wstecz ile bylo trzeba zamowic. Tutaj zmieniamy do ilu wstecz patrzymy
    for i in range(1, 6):
        grouped_data_encoded[f'Quantity_Ordered_T-{i}'] = grouped_data_encoded['Quantity Ordered'].shift(i)

    grouped_data_encoded = grouped_data_encoded.dropna()
    X = grouped_data_encoded.drop(['Quantity Ordered'], axis=1)
    y = grouped_data_encoded['Quantity Ordered']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'verbose': -1
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

    best_params = study.best_params
    best_model = lgb.LGBMRegressor(**best_params, random_state=42)

    pipeline_best = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', best_model)
    ])

    pipeline_best.fit(X_train, y_train)

    predicted_quantities = pipeline_best.predict(grouped_data_encoded.drop(columns=['Quantity Ordered']))
    print(predicted_quantities)
    # Zmienna, która określa tydzień dla predykcji
    week_pred = end_week + 1

    selected_city = 'seattle'
    selected_product = 'AAA Batteries (4-pack)'
    filter_mask = (grouped_data_encoded['City_Seattle'] == 1) & (
                grouped_data_encoded['Product_iPhone'] == 1)
    filtered_data = grouped_data_encoded[filter_mask]
    print(filtered_data)

    if week_pred <= filtered_data['Week'].max() + 1:
        relevant_data = filtered_data[filtered_data['Week'] < week_pred].copy()

        # Tworzymy zestaw danych do predykcji dla wybranego tygodnia
        new_data = relevant_data[relevant_data['Week'] == relevant_data['Week'].max()].copy()

        # Ustawiamy nowy tydzień jako 'week_pred'
        new_data['Week'] = week_pred

        # Zaktualizuj przesunięcia danych, np. T-1, T-2, itp.
        for i in range(1, 6):  # Zakładamy 5 przesuniętych tygodni
            new_data[f'Quantity_Ordered_T-{i}'] = relevant_data[f'Quantity_Ordered_T-{i}'].values[-1]

        # Przewidujemy dla danego tygodnia
        new_data = new_data.drop(columns=['Quantity Ordered'])  # Usuwamy rzeczywiste wartości sprzedaży
        predicted_quantity = pipeline_best.predict(new_data)

        print(
            f"Przewidziana ilość zamówień dla {selected_product} w {selected_city} na tydzień {week_pred}: {predicted_quantity[0]}")

        # Rysowanie wykresu: Rzeczywista sprzedaż vs przewidywana sprzedaż
        plt.figure(figsize=(12, 6))

        # Rysowanie rzeczywistej sprzedaży dla wcześniejszych tygodni
        plt.plot(relevant_data['Week'], relevant_data['Quantity Ordered'], label='Rzeczywista Sprzedaż', marker='o')

        # Dodaj przewidywaną sprzedaż dla tygodnia week_pred
        plt.plot(week_pred, predicted_quantity, label='Przewidziana Sprzedaż', marker='x', color='red', markersize=10)

        plt.title(f'Porównanie Sprzedaży dla {selected_product} w {selected_city} (Predykcja dla tygodnia {week_pred})')
        plt.xlabel('Tydzień')
        plt.ylabel('Ilość Zamówionych Produktów')
        plt.legend()
        plt.grid()
        plt.show()

    else:
        print(
            f"Brak wystarczających danych, aby przewidzieć tydzień {week_pred}. Upewnij się, że dostępne są dane historyczne.")

if __name__ == '__main__':
    app.run(debug=True)
    # _test2()

    # test4()