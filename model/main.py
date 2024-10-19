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

EXPECTED_COLUMNS = [
    '', 'Order ID', 'Product', 'Quantity Ordered', 'Price Each', 'Order Date',
    'Purchase Address', 'Month', 'Sales', 'City', 'Hour'
]

def validate_csv(file):
    df = pd.read_csv(file)
    if list(df.columns) != EXPECTED_COLUMNS:
        return False
    return True
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
    grouped_data_encoded = pd.concat([grouped_data, encoded_cities_df, encoded_products_df], axis=1)

    grouped_data_encoded.drop(columns=['City', 'Product'], inplace=True)

    grouped_data_encoded = grouped_data_encoded[grouped_data_encoded['Week'].isin(selected_weeks)]

    # grouped_data_encoded = grouped_data_encoded[grouped_data_encoded['Week'].isin(range(start_week, end_week + 1))]

    grouped_data_encoded['Quantity Ordered'] = grouped_data_encoded['Quantity Ordered'].astype(float)

    # na zakresie 1-6 uczymy się na historii z 5 tygodni wstecz ile bylo trzeba zamowic. Tutaj zmieniamy do ilu wstecz patrzymy
    for i in range(start_week, end_week):
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
    # filtered_historical_data = ''
    # for city in grouped_data['City'].unique():
    #     for product in grouped_data['Product'].unique():
    #         for day in range(start_week, end_week + 1):
    #             mask = (grouped_data['City'] == city) & (grouped_data['Product'] == product) & (
    #                         grouped_data['Week'] == day)
    #             filtered_historical_data = grouped_data.loc[mask, 'Quantity Ordered']

    start_date = start_week
    end_date = end_week
    report_file_path = 'historical_sales_report.txt'
    with open(report_file_path, 'w') as file:
        for city in grouped_data['City'].unique():
            for product in grouped_data['Product'].unique():
                file.write(f'{city};{product}')

                for day in range(start_date, end_date + 1):
                    mask = (grouped_data['City'] == city) & (grouped_data['Product'] == product) & (
                                grouped_data['Week'] == day)
                    filtered_data = grouped_data.loc[mask, 'Quantity Ordered']

                    # Jeżeli są dane dla tego dnia, zapisz liczbę zamówień; jeśli nie, zapisz 0
                    if not filtered_data.empty:
                        file.write(f'{int(filtered_data.values[0])};')
                    else:
                        file.write('0;')  # Brak danych - 0

                # Nowa linia po każdym produkcie
                file.write('\n')

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
                    for i in range(1, end_week):  # Zakładamy 5 przesuniętych tygodni
                        if f'Quantity_Ordered_T-{i}' in relevant_data.columns:
                            new_data[f'Quantity_Ordered_T-{i}'] = relevant_data[f'Quantity_Ordered_T-{i}'].values[-1]

                    # Przewidujemy dla danego tygodnia

                    historical_quantities = relevant_data[['Week', 'Quantity Ordered']].sort_values(by='Week')

                    if not historical_quantities.empty:
                        quantities_list = historical_quantities['Quantity Ordered'].tolist()

                        full_quantities_list = [str(quantities_list[i]) if i < len(quantities_list) else '0' for i in
                                                range(week_pred - 1)]

                        file.write(f'{city_name}\t{product_name}\t' + '\t'.join(full_quantities_list))

                    new_data = new_data.drop(columns=['Quantity Ordered'],
                                             errors='ignore')  # Usuwamy rzeczywiste wartości sprzedaży, jeśli są
                    predicted_quantity = pipeline_best.predict(new_data)[0]

                    # Zapisujemy wynik do pliku
                    file.write(f'\t{predicted_quantity:.2f}\n')

                    print(
                        f"Przewidziana ilość zamówień dla {product_name} w {city_name} na tydzień {week_pred}: {predicted_quantity:.2f}")
                else:
                    print(f"Brak wystarczających danych dla {product_name} w {city_name} na tydzień {week_pred}.")

    print(f"\nRaport został zapisany do pliku: {os.path.abspath(report_file_path)}")
    with open(report_file_path, 'r') as file:
        lines = file.readlines()

    predictions = []
    for line in lines[2:]:  # Skip the first two lines (header)
        parts = line.strip().split('\t')
        city = parts[0]
        product = parts[1]
        historical_sales = list(map(float, parts[2:-1]))
        predicted_quantity = float(parts[-1])
        predictions.append({
            'city': city,
            'product': product,
            'historical_sales': historical_sales,
            'predicted_quantity': predicted_quantity
        })

    return render_template('predictions.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
    # _test2()

    # test4()