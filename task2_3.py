from pyspark import SparkContext
import sys
import time
import json
import numpy as np
import xgboost
from xgboost import XGBRegressor

def item_based(folder_path, val_file_path, output_file_path,sc):
    def user_business_map(data):
        mapped_rdd = data.map(lambda x: (x[0], (x[1], float(x[2]))))
        user_business_rdd = mapped_rdd.groupByKey().mapValues(lambda vals: {"business": {business: rating for business, rating in vals},
                                                                            "avg_rating": sum(rating for _, rating in vals) / len(vals) if len(vals) > 0 else 0.0})
        user_business_dict = user_business_rdd.collectAsMap()
        return user_business_dict

    def business_user_map(data):
        mapped_rdd = data.map(lambda x: (x[1], (x[0], float(x[2]))))
        business_user_rdd = mapped_rdd.groupByKey().mapValues(lambda vals: {"users": {user: rating for user, rating in vals},
                                                                            "avg_rating": sum(rating for _, rating in vals) / len(vals) if len(vals) > 0 else 0.0})
        business_user_dict = business_user_rdd.collectAsMap()
        return business_user_dict

    def get_pearson_similarity(item1, item2, item_user_map):
        corated_users = set(item_user_map[item1]['users'].keys()).intersection(set(item_user_map[item2]['users'].keys()))
        if len(corated_users) <= 1:
            avg1 = item_user_map[item1].get('avg_rating', 3)  # Default avg_rating if missing
            avg2 = item_user_map[item2].get('avg_rating', 3)
            w = (5 - abs(avg1 - avg2)) / 5
            return w
        
        rating1 = []
        rating2 = []
        for user in corated_users:
            rating1.append(float(item_user_map[item1]['users'][user]))
            rating2.append(float(item_user_map[item2]['users'][user]))

        avg_rating1 = sum(rating1) / len(rating1)
        avg_rating2 = sum(rating2) / len(rating2)
        rating1 = [rating - avg_rating1 for rating in rating1]
        rating2 = [rating - avg_rating2 for rating in rating2]
        numerator = sum([rating1[i] * rating2[i] for i in range(len(rating1))])
        denominator1 = sum([rating ** 2 for rating in rating1]) ** 0.5
        denominator2 = sum([rating ** 2 for rating in rating2]) ** 0.5
        if denominator1 == 0 or denominator2 == 0:
            return 0.0
        w = numerator / (denominator1 * denominator2)
        return w

    def predict_rating_amplified(num_neighbors, user, target_business, user_business_map, business_user_map, p=2.5):
        if user not in user_business_map or target_business not in business_user_map:
            return 3.0
        pearson_similarities = []
        user_info = user_business_map[user]
        user_rated_businesses = user_info['business']
        for rated_business, user_rating in user_rated_businesses.items():
            if rated_business == target_business:
                continue
            similarity = get_pearson_similarity(target_business, rated_business, business_user_map)
            amplified_similarity = abs(similarity) ** p
            pearson_similarities.append((amplified_similarity, rated_business, user_rating))
        if not pearson_similarities:
            return business_user_map[target_business]['avg_rating']
        pearson_similarities.sort(key=lambda x: x[0], reverse=True)
        top_similarities = pearson_similarities[:num_neighbors]
        numerator = 0
        denominator = 0
        for similarity, business, rating in top_similarities:
            numerator += similarity * rating
            denominator += abs(similarity)
        if denominator == 0:
            return business_user_map[target_business]['avg_rating']
        predicted_rating = numerator / denominator
        predicted_rating = max(1.0, min(5.0, predicted_rating))
        return predicted_rating
    
    train_file_name = folder_path + "/yelp_train.csv"
    test_file_name = val_file_path
    raw_train_data = sc.textFile(train_file_name)
    header = raw_train_data.first()
    train_data = raw_train_data.filter(lambda line: line != header)
    train_data = train_data.map(lambda line: line.split(",")).map(lambda x: (x[0], x[1], x[2]))
    user_business_map_train = user_business_map(train_data)
    business_user_map_train = business_user_map(train_data)

    raw_val_data = sc.textFile(test_file_name)
    header_val = raw_val_data.first()
    val_data = raw_val_data.filter(lambda line: line != header_val)
    val_data = val_data.map(lambda line: line.split(",")).map(lambda x: (x[0], x[1]))
    predictions = val_data.map(lambda x: (x[0], x[1], predict_rating_amplified(20, x[0], x[1], user_business_map_train, business_user_map_train)))
    predictions = predictions.collect()
    item_pred = []
    for user, business, pred in predictions:
        item_pred.append((user, business, pred))

    return item_pred

def model_based(folder_path, val_file_path, output_file_path, sc):
    user_json = folder_path + "/user.json"
    business_json = folder_path + "/business.json"
    review_json = folder_path + "/review_train.json"
    yelp_train = folder_path + "/yelp_train.csv"

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

    val_data = sc.textFile(val_file_path)
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

    model_pred = []
    for i, record in enumerate(validation_features_labels):
        model_pred.append((record[0], record[1], y_pred[i]))
    return model_pred

def logistic_alpha(review_count, rating_variance, x0, y0, k1=0.01, k2=0.05):
    alpha_review = 1 / (1 + np.exp(k1 * (review_count - x0)))
    alpha_rating = 1 / (1 + np.exp(k2 * (rating_variance - y0)))
    alpha = 0.5 * alpha_review + 0.5 * alpha_rating
    alpha = max(0, min(1, alpha)) 
    return alpha

def main():
    if len(sys.argv) != 4:
        print("Usage: task2_3.py <folder_path> <val_file_path> <output_file_path>")
        sys.exit(-1)

    folder_path = sys.argv[1]
    val_file_path = sys.argv[2]
    output_file_path = sys.argv[3]
    reviews_file = folder_path + "/review_train.json"

    sc = SparkContext("local[*]", "task2_3")
    sc.setLogLevel("ERROR")
    def compute_variance(ratings):
        n = len(ratings)
        if n <= 1:
            return 0
        mean = sum(ratings) / n
        return sum((rating - mean) ** 2 for rating in ratings) / (n - 1)
    start_time = time.time()
    raw_reviews = sc.textFile(reviews_file)
    reviews = raw_reviews.map(json.loads).map(lambda review: (review['business_id'], review['stars'])).groupByKey().mapValues(lambda stars: {
    'review_count': len(stars), 'rating_variance': compute_variance(list(stars))})
    reviews = reviews.collectAsMap()
    review_counts = [v['review_count'] for v in reviews.values()]
    rating_variances = [v['rating_variance'] for v in reviews.values()]

    x0 = np.mean(review_counts) if review_counts else 0
    y0 = np.mean(rating_variances) if rating_variances else 0

    item_based_predictions = item_based(folder_path, val_file_path, output_file_path,sc)
    model_based_predictions = model_based(folder_path, val_file_path, output_file_path, sc)
    item_based_dict = { (user, business): pred for user, business, pred in item_based_predictions}
    model_based_dict = { (user, business): pred for user, business, pred in model_based_predictions}
    
    predictions = []

    for (user, business) in item_based_dict.keys():
        item_pred = item_based_dict.get((user, business), 3.0)
        model_pred = model_based_dict.get((user, business), 3.0)
        business_info = reviews.get(business, {'review_count': 0, 'rating_variance': 0.0})
        review_count = business_info['review_count']
        rating_variance = business_info['rating_variance']
        alpha = logistic_alpha(review_count, rating_variance, x0, y0, k1=0.01, k2=0.1)

        prediction = alpha * item_pred + (1 - alpha) * model_pred
        predictions.append((user, business, prediction))

    with open(output_file_path, 'w') as f:
        f.write("user_id, business_id, prediction\n")
        for user, business, prediction in predictions:
            f.write(f"{user},{business},{prediction}\n")
    sc.stop()
    end_time = time.time()
    print(f"Execution Time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()




