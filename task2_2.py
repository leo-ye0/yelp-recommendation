from pyspark import SparkContext
import sys
import time
import json
import numpy as np
import xgboost
from xgboost import XGBRegressor

def main():
    if len(sys.argv) != 4:
        print("Usage: task2_2.py <folder_path> <val_file_path> <output_file_path>")
        sys.exit(-1)

    folder_path = sys.argv[1]
    val_path = sys.argv[2]
    output_path = sys.argv[3]
    user_json = folder_path + "/user.json"
    business_json = folder_path + "/business.json"
    review_json = folder_path + "/review_train.json"
    yelp_train = folder_path + "/yelp_train.csv"

    sc = SparkContext("local[*]", "task2_2")
    sc.setLogLevel("ERROR")
    start_time = time.time()
    raw_train_data = sc.textFile(yelp_train)
    header = raw_train_data.first()
    train_data = raw_train_data.filter(lambda line: line != header)
    train_data = train_data.map(lambda line: line.split(",")).map(lambda x: (x[0], x[1], float(x[2])))
    review_rdd = sc.textFile(review_json).map(json.loads).map(lambda row: (row['business_id'], (float(row['useful']), float(row['funny']), float(row['cool'])))
                                                              ).groupByKey().mapValues(list).mapValues(lambda x: np.mean(x, axis=0)).cache()
    review_map = review_rdd.collectAsMap()
    business_map = sc.textFile(business_json).map(json.loads).map(lambda row: (row['business_id'], (float(row['stars']), float(row['review_count'])))).collectAsMap()
    user_map = sc.textFile(user_json).map(json.loads).map(lambda row: (row['user_id'], (float(row['average_stars']), float(row['review_count'])))).collectAsMap()

    bc_review_map = sc.broadcast(review_map)
    bc_business_map = sc.broadcast(business_map)
    bc_user_map = sc.broadcast(user_map)

    def extract_features(record):
        user_id, business_id, stars = record
        user_features = bc_user_map.value.get(user_id, (3.0, 0))
        business_features = bc_business_map.value.get(business_id, (3.0, 0))
        review_features = bc_review_map.value.get(business_id, (0.0, 0.0, 0.0))
        return (
            user_features[0], user_features[1], business_features[0], business_features[1], review_features[0], review_features[1], review_features[2], stars                       
        )
    training_features_labels_rdd = train_data.map(extract_features)
    training_features_labels = training_features_labels_rdd.collect()
    X_train = np.array([
        [record[0], record[1], record[2], record[3], record[4], record[5], record[6]] for record in training_features_labels
        ])
    y_train = np.array([record[7] for record in training_features_labels])
    model = XGBRegressor(
        objective='reg:linear',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    def extract_features_test(record):
        user_id, business_id = record
        user_features = bc_user_map.value.get(user_id, (3.0, 0))
        business_features = bc_business_map.value.get(business_id, (3.0, 0))
        review_features = bc_review_map.value.get(business_id, (0.0, 0.0, 0.0))
        return (
            user_id,                # Keep the user_id
            business_id,            # Keep the business_id
            user_features[0],       # user_average_stars
            user_features[1],       # user_review_count
            business_features[0],   # business_stars
            business_features[1],   # business_review_count
            review_features[0],     # business_avg_useful
            review_features[1],     # business_avg_funny
            review_features[2]      # business_avg_cool
        )

    val_data = sc.textFile(val_path)
    header_val = val_data.first()
    val_data = val_data.filter(lambda line: line != header_val)
    val_data = val_data.map(lambda line: line.split(",")).map(lambda x: (x[0], x[1]))
    validation_features_labels_rdd = val_data.map(extract_features_test)
    validation_features_labels = validation_features_labels_rdd.collect()
    user_business_ids = [
    (record[0], record[1]) for record in validation_features_labels
    ]
    X_val = np.array([
    [
        record[2],  # user_average_stars
        record[3],  # user_review_count
        record[4],  # business_stars
        record[5],  # business_review_count
        record[6],  # business_avg_useful
        record[7],  # business_avg_funny
        record[8]   # business_avg_cool
    ]
    for record in validation_features_labels
])
    y_val = np.array([record[7] for record in validation_features_labels])
    y_pred = model.predict(X_val)

    with open(output_path, 'w') as f:
        f.write("user_id, business_id, prediction\n")
        for i, record in enumerate(validation_features_labels):
            f.write(f"{record[0]},{record[1]},{y_pred[i]}\n")   
    sc.stop()
    end_time = time.time()
    print(f"Execution Time: {end_time - start_time} seconds")
if __name__ == "__main__":
    main()


        


    



