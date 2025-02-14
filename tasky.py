from pyspark import SparkContext
import pandas as pd
import numpy as np
import json
import csv
import sys
import time
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV  # Added GridSearchCV
from datetime import datetime

def main():
    if len(sys.argv) != 4:
        print("Usage: competition.py <folder_path> <val_file_path> <output_file_path>")
        sys.exit(-1)

    folder_path = sys.argv[1]
    val_path = sys.argv[2]
    output_path = sys.argv[3]
    user_json = folder_path + "/user.json"
    business_json = folder_path + "/business.json"
    review_json = folder_path + "/review_train.json"
    yelp_train = folder_path + "/yelp_train.csv"
    photo_json = folder_path + "/photo.json"
    tip_json = folder_path + "/tip.json"

    sc = SparkContext("local[*]", "competition")
    sc.setLogLevel("ERROR")
    start_time = time.time()
    raw_user_data = sc.textFile(user_json)
    raw_business_data = sc.textFile(business_json)
    raw_review_data = sc.textFile(review_json)
    raw_train_data = sc.textFile(yelp_train)
    raw_photo_data = sc.textFile(photo_json)
    raw_tip_data = sc.textFile(tip_json)

    unique_cities = raw_business_data.map(json.loads).map(lambda row: row.get('city', 'Unknown')).filter(lambda city: city != "None").distinct().collect()
    unique_states = raw_business_data.map(json.loads).map(lambda row: row.get('state', 'Unknown')).filter(lambda state: state != "None").distinct().collect()
    city_mapping = {city: idx for idx, city in enumerate(sorted(unique_cities))}
    city_mapping['Unknown'] = -1
    state_mapping = {state: idx for idx, state in enumerate(sorted(unique_states))}
    state_mapping['Unknown'] = -1 
    broadcast_city_mapping = sc.broadcast(city_mapping)
    broadcast_state_mapping = sc.broadcast(state_mapping)
    
    class User:
        @staticmethod
        def user_feature(row):
            if row["elite"] != "None":
                row['elite_cnt'] = len(row["elite"].split(","))
            else:
                row['elite_cnt'] = 0
            yelping_since = datetime.strptime(row["yelping_since"], "%Y-%m-%d")
            row['yelping_time'] = datetime.now().year - yelping_since.year
            if row["friends"] != "None":
                row['friends_cnt'] = len(row["friends"].split(","))
            else:
                row['friends_cnt'] = 0
            return row

    user_rdd = raw_user_data.map(json.loads).map(User.user_feature).map(
        lambda row: (
            row['user_id'], 
            (
                float(row['average_stars']), 
                np.log1p(row['review_count']), 
                row['elite_cnt'], 
                row['yelping_time'],
                row["useful"], 
                row["funny"], 
                row["cool"], 
                row["fans"], 
                row['friends_cnt']
            )
        )
    ).cache().collectAsMap()
        
    class Business:
        @staticmethod
        def business_feature(row, city_mapping, state_mapping):
            row['attributes_cnt'] = len(row['attributes']) if row.get('attributes') else 0
            row['categories_cnt'] = len(row['categories'].split(",")) if row.get('categories') else 0
            city = row.get('city', 'Unknown')
            row['city_encoded'] = city_mapping.get(city, -1)
            state = row.get('state', 'Unknown')
            row['state_encoded'] = state_mapping.get(state, -1)
            return row

    # Corrected business_rdd mapping
    business_rdd = raw_business_data.map(json.loads).map(
        lambda row: Business.business_feature(row, broadcast_city_mapping.value, broadcast_state_mapping.value)
    ).map(
        lambda row: (
            row['business_id'], 
            (
                float(row['stars']), 
                np.log1p(row['review_count']), 
                row['attributes_cnt'], 
                row['categories_cnt'], 
                row['city_encoded'], 
                row['state_encoded'], 
                row['is_open']
            )
        )
    ).cache().collectAsMap()
    
    class Photo:
        @staticmethod
        def photo_feature(row):
            business_id = row.get('business_id', None)
            if business_id is not None:
                return (business_id, 1)
            else:
                return (None, 0)
    
    def process_photos(raw_photo_data):
        photo_rdd = raw_photo_data.map(json.loads).map(Photo.photo_feature).filter(lambda x: x[0] is not None).reduceByKey(lambda a, b: a + b).map(lambda x: (x[0], x[1])).cache().collectAsMap()
        return photo_rdd
    
    photo_rdd = process_photos(raw_photo_data)

    class Review:
        def __init__(self, raw_review_rdd):
            self.review_rdd = raw_review_rdd.map(json.loads)\
                .map(lambda row: (
                    row['business_id'],
                    (
                        float(row.get('stars', 0)),
                        float(row.get('useful', 0)),
                        float(row.get('funny', 0)),
                        float(row.get('cool', 0)),
                        1  # Count of reviews
                    )
                ))\
                .reduceByKey(lambda a, b: (
                    a[0] + b[0],  # Sum of stars
                    a[1] + b[1],  # Sum of useful
                    a[2] + b[2],  # Sum of funny
                    a[3] + b[3],  # Sum of cool
                    a[4] + b[4]   # Count
                ))\
                .mapValues(lambda sums: (
                    sums[0] / sums[4],  # Average stars
                    sums[1] / sums[4],  # Average useful
                    sums[2] / sums[4],  # Average funny
                    sums[3] / sums[4]   # Average cool
                ))\
                .cache()\
                .collectAsMap()
        
        def get_review_rdd(self):
            return self.review_rdd
            
    review_rdd = Review(raw_review_data).get_review_rdd()

    header_train = raw_train_data.first()
    raw_train_data = raw_train_data.filter(lambda line: line != header_train)
    train_rdd = raw_train_data.map(lambda line: line.split(",")).map(lambda cols: (cols[0], cols[1], float(cols[2])))

    user_rdd_bc = sc.broadcast(user_rdd)
    business_rdd_bc = sc.broadcast(business_rdd)
    review_rdd_bc = sc.broadcast(review_rdd)
    photo_rdd_bc = sc.broadcast(photo_rdd)

    def extract_features(record):
        user_id, business_id, stars = record
        user_features = user_rdd_bc.value.get(user_id, (3.0, 0, 0, 0, 0, 0, 0, 0, 0))
        business_features = business_rdd_bc.value.get(business_id, (3.0, 0, 0, 0, -1, -1, 0))
        review_features = review_rdd_bc.value.get(business_id, (0.0, 0.0, 0.0, 0.0))
        photo_features = photo_rdd_bc.value.get(business_id, 0)
        return (
            user_features[0], user_features[1], user_features[2], user_features[3], user_features[4], user_features[5], user_features[6], user_features[7], user_features[8],
            business_features[0], business_features[1], business_features[2], business_features[3], business_features[4], business_features[5], business_features[6],
            review_features[1], review_features[2], review_features[3],
            photo_features, stars
        )
    
    training_features_labels_rdd = train_rdd.map(extract_features)
    training_features_labels = training_features_labels_rdd.collect()

    X_train = np.array([record[:-1] for record in training_features_labels])
    y_train = np.array([record[-1] for record in training_features_labels])

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)

    # Define the parameter grid for Grid Search
    param_grid = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [6, 10, 12],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.7, 1],
        'reg_lambda': [1, 1.5, 2],
        'reg_alpha': [0.5, 1]
    }

    # Initialize the XGBRegressor
    xgb = XGBRegressor(
        objective='reg:linear',  # Updated to the recommended objective
        random_state=42,
        verbosity=1
    )

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=2,
        n_jobs=-1  # Utilize all available cores
    )

    # Fit GridSearchCV on the training data
    grid_search.fit(X_train, y_train)

    # Retrieve the best estimator
    best_model = grid_search.best_estimator_
    print("Best Parameters found: ", grid_search.best_params_)
    print("Best CV Score (Negative MSE): ", grid_search.best_score_)

    # Proceed with the best model for predictions
    def extract_features_test(record):
        user_id, business_id = record
        user_features = user_rdd_bc.value.get(user_id, (3.0, 0, 0, 0, 0, 0, 0, 0, 0))
        business_features = business_rdd_bc.value.get(business_id, (3.0, 0, 0, 0, -1, -1,0))
        review_features = review_rdd_bc.value.get(business_id, (0.0, 0.0, 0.0, 0.0))
        photo_features = photo_rdd_bc.value.get(business_id, 0)
        return (
            user_id, business_id,
            user_features[0], user_features[1], user_features[2], user_features[3], user_features[4], user_features[5], user_features[6], user_features[7], user_features[8],
            business_features[0], business_features[1], business_features[2], business_features[3], business_features[4], business_features[5], business_features[6],
            review_features[1], review_features[2], review_features[3],
            photo_features
        )
    
    val_data = sc.textFile(val_path)
    header_val = val_data.first()
    val_data = val_data.filter(lambda line: line != header_val)
    val_data = val_data.map(lambda line: line.split(",")).map(lambda x: (x[0], x[1]))
    validation_features_labels_rdd = val_data.map(extract_features_test)
    validation_features_labels = validation_features_labels_rdd.collect()

    X_val = np.array([record[2:] for record in validation_features_labels])
    X_val = scaler.transform(X_val)

    y_pred = best_model.predict(X_val)  # Use the best model from Grid Search
    user_business_ids = [(record[0], record[1]) for record in validation_features_labels]
    output = zip(user_business_ids, y_pred)
    
    with open(output_path, 'w', newline='') as f:  # Added newline='' for Windows compatibility
        writer = csv.writer(f)
        writer.writerow(["user_id", "business_id", "prediction"])
        for (user_id, business_id), prediction in output:
            writer.writerow([user_id, business_id, prediction])
    
    print(f"Execution time: {time.time() - start_time} seconds")

if __name__ == "__main__":
    main()

