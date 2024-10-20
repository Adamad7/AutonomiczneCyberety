from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import optuna
from flask import Flask, request, render_template, redirect
import pandas as pd
import numpy as np
app = Flask(__name__)

product_prices_map = {}
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
        product_prices = df[['Product', 'Price Each']].drop_duplicates()
        global product_prices_map
        product_prices_map = product_prices.set_index('Product')['Price Each'].to_dict()
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
    grouped_data_encoded['Quantity Ordered'] = grouped_data_encoded['Quantity Ordered'].astype(float)

    for i in range(1, min(4, end_week - start_week)):
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

    week_pred = end_week + 1

    city_columns = [col for col in grouped_data_encoded.columns if col.startswith('City_')]
    product_columns = [col for col in grouped_data_encoded.columns if col.startswith('Product_')]
    print(product_prices_map)
    historical_sales = {}
    for city in grouped_data['City'].unique():
        historical_sales[city] = {}
        for product in grouped_data['Product'].unique():
            historical_sales[city][product] = []

            for day in range(start_week, end_week + 1):
                mask = (grouped_data['City'] == city) & (grouped_data['Product'] == product) & (
                            grouped_data['Week'] == day)
                filtered_data = grouped_data.loc[mask, 'Quantity Ordered']

                if not filtered_data.empty:
                    historical_sales[city][product].append(int(filtered_data.values[0]))
                else:
                    historical_sales[city][product].append(0)

    prediction_data = {}
    total_sales_in_cities = {}
    total_order_values_in_cities = {}
    for city_col in city_columns:
        city_name = city_col.replace('City_', '').replace('_', ' ')
        prediction_data[city_name] = {}

        for product_col in product_columns:
            product_name = product_col.replace('Product_', '').replace('_', ' ')

            filter_mask = (grouped_data_encoded[city_col] == 1) & (grouped_data_encoded[product_col] == 1)
            filtered_data = grouped_data_encoded[filter_mask]

            if len(filtered_data) > 0 and week_pred <= filtered_data['Week'].max() + 1:
                relevant_data = filtered_data.loc[(filtered_data['Week'] < week_pred) & (filtered_data['Week'] > start_week)].copy()
                prediction_data[city_name][product_name] = {}

                next_week_data = relevant_data[relevant_data['Week'] == relevant_data['Week'].max()].copy()
                next_week_data['Week'] = week_pred

                for i in range(1, min(4, end_week - start_week)):
                    if f'Quantity_Ordered_T-{i}' in relevant_data.columns:
                        next_week_data[f'Quantity_Ordered_T-{i}'] = relevant_data[f'Quantity_Ordered_T-{i}'].values[-1]


                prediction_data[city_name][product_name]['historical_sales'] = historical_sales[city_name][product_name]



                next_week_data = next_week_data.drop(columns=['Quantity Ordered'], errors='ignore')
                predicted_quantity = pipeline_best.predict(next_week_data)[0]
                prediction_data[city_name][product_name]['predicted_sales'] = int(max(np.ceil(predicted_quantity), 0))
                # prediction_data[city_name][product_name]['product_price'] = next_week_data['Price Each'].values[0]
                prediction_data[city_name][product_name]['unit_price'] = product_prices_map[product_name]

            else:
                print(f"Brak wystarczających danych dla {product_name} w {city_name} na tydzień {week_pred}.")
        total_sales_in_cities[city_name] = sum(
            [data['predicted_sales'] for data in prediction_data[city_name].values()])
        total_order_values_in_cities[city_name] = sum(
            [data['predicted_sales'] * data['unit_price'] for data in prediction_data[city_name].values()])

    print(prediction_data)

    product_sizes = {
        '20in Monitor': 0.05,
        '27in 4K Gaming Monitor': 0.07,
        '27in FHD Monitor': 0.06,
        '34in Ultrawide Monitor': 0.1,
        'AA Batteries (4-pack)': 0.001,
        'AAA Batteries (4-pack)': 0.001,
        'Apple Airpods Headphones': 0.002,
        'Bose SoundSport Headphones': 0.002,
        'Flatscreen TV': 0.9,
        'Google Phone': 0.003,
        'LG Dryer': 0.8,
        'LG Washing Machine': 1.0,
        'Lightning Charging Cable': 0.001,
        'Macbook Pro Laptop': 0.01,
        'ThinkPad Laptop': 0.015,
        'USB-C Charging Cable': 0.001,
        'Vareebadd Phone': 0.003,
        'Wired Headphones': 0.002,
        'iPhone': 0.003
    }

    package_capacity = 0.7  # cubic meters


    for city_name, products in prediction_data.items():
        current_volume = 0
        car_number = 0
        num_packages = 0
        for product_name, data in products.items():
            if 'predicted_sales' in data:
                predicted_sales = data['predicted_sales']
                for x in range(predicted_sales):
                    product_volume = product_sizes.get(product_name, 0)
                    if product_volume > package_capacity:
                        car_number += 1
                    elif current_volume + product_volume < package_capacity:
                        current_volume += product_volume
                    else:
                        num_packages += 1
                        current_volume = product_volume

        if current_volume > 0:
            num_packages += 1

        prediction_data[city_name]['transport'] = {
            'packages': int(num_packages),
            'cars': int(car_number)
        }

    return render_template('predictions.html', predictions=prediction_data, start_week=start_week, end_week=end_week, total_sales=total_sales_in_cities, total_order_values=total_order_values_in_cities)

if __name__ == '__main__':
    app.run(debug=True)