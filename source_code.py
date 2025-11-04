"""
Matrix factorization
=======================================================

This implementation performs collaborative filtering using matrix factorization, enhanced with user and item bias terms.
The algorithm is particularly suited to recommendation systems, where we attempt to predict how a user would rate an item they haven't yet seen, based on patterns in existing user-item interactions.

Each observed rating `r_ui` (user u rating item i) is modelled as:

    r_ui ≈ μ + b_u + b_i + p_uᵀ q_i

Where:
- μ is the global average rating across the dataset.
- b_u is a learned bias term for user u (e.g., some users tend to give higher or lower ratings overall).
- b_i is a learned bias term for item i (e.g., some items are generally liked more).
- p_u is a vector representing latent features of user u (e.g., genre preferences, interests).
- q_i is a vector representing latent features of item i (e.g., attributes like "comedy", "drama", "thrill").

We aim to learn these latent features and biases such that the dot product `p_uᵀ q_i` captures how well item i fits user u's preferences.

To learn these parameters, we minimize the regularized squared error loss using Stochastic Gradient Descent (SGD):

    Loss = Σ (r_ui - μ - b_u - b_i - p_uᵀ q_i)^2 
           + λ (||p_u||² + ||q_i||² + b_u² + b_i²)

Where λ is the regularization factor, which penalizes overly complex models to avoid overfitting.

The predicted ratings are clamped to the range [0.5, 5.0] in increments of 0.5.
"""

import numpy as np
import csv
import random

# Set model hyperparameters
NUM_FACTORS = 40        # Number of latent features for each user and item
LEARNING_RATE = 0.01    # Step size for gradient descent updates
REGULARIZATION = 0.03   # Regularization strength (lambda)
EPOCHS = 10             # Number of passes over the training data
BATCH_SIZE = 512        # Number of samples per SGD batch

def load_train_data(filename):
    """
    Load training data from a CSV file.

    Each line in the file is expected to contain:
        user_id, item_id, rating, timestamp
    """
    ratings = []
    users, items = set(), set()
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 4:
                continue  # skip malformed lines
            user_id, item_id, rating = int(parts[0]), int(parts[1]), float(parts[2])
            ratings.append((user_id, item_id, rating))
            users.add(user_id)
            items.add(item_id)
    return ratings, users, items

def load_test_data(filename):
    """
    Load test data from a CSV file.

    Each line contains:
        user_id, item_id, timestamp
    
    """
    test_data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 3:
                continue
            user_id, item_id, timestamp = int(parts[0]), int(parts[1]), int(parts[2])
            test_data.append((user_id, item_id, timestamp))
    return test_data

def clamp_rating(r):
    """
    Restrict predicted ratings to the valid range [0.5, 5.0], rounded to the nearest 0.5.
    """
    return round(min(5.0, max(0.5, r)) * 2) / 2.0

def train_mf_bias(ratings, users, items):
    """
    Train the matrix factorization model with user and item biases using stochastic gradient descent.
    """
    # Map user and item IDs to matrix indices
    user_to_index = {u: i for i, u in enumerate(sorted(users))}
    item_to_index = {i: j for j, i in enumerate(sorted(items))}
    num_users = len(users)
    num_items = len(items)

    # Initialize user (P) and item (Q) latent factor matrices with small Gaussian noise
    P = np.random.normal(0, 0.1, (num_users, NUM_FACTORS))
    Q = np.random.normal(0, 0.1, (num_items, NUM_FACTORS))

    # Initialize user and item biases to zero
    user_bias = np.zeros(num_users)
    item_bias = np.zeros(num_items)

    # Compute global mean rating from training data
    global_mean = np.mean([r for _, _, r in ratings])

    # Begin training over multiple epochs
    for epoch in range(EPOCHS):
        random.shuffle(ratings)  # Shuffle ratings each epoch to help SGD
        total_error = 0

        for batch_start in range(0, len(ratings), BATCH_SIZE):
            batch = ratings[batch_start:batch_start + BATCH_SIZE]

            for user_id, item_id, rating in batch:
                if user_id not in user_to_index or item_id not in item_to_index:
                    continue  # Skip unknown users/items

                # Lookup matrix indices
                u_idx = user_to_index[user_id]
                i_idx = item_to_index[item_id]

                # Compute predicted rating using matrix factorization 
                pred = global_mean + user_bias[u_idx] + item_bias[i_idx] + np.dot(P[u_idx], Q[i_idx])

                # Calculate prediction error
                err = rating - pred
                total_error += abs(err)

                # Update user and item biases (gradient descent step)
                user_bias[u_idx] += LEARNING_RATE * (err - REGULARIZATION * user_bias[u_idx])
                item_bias[i_idx] += LEARNING_RATE * (err - REGULARIZATION * item_bias[i_idx])

                # Update latent feature vectors
                P[u_idx] += LEARNING_RATE * (err * Q[i_idx] - REGULARIZATION * P[u_idx])
                Q[i_idx] += LEARNING_RATE * (err * P[u_idx] - REGULARIZATION * Q[i_idx])

        # Print mean absolute error after each epoch 
        mae = total_error / len(ratings)
        print(f"Epoch {epoch + 1}/{EPOCHS} — MAE: {mae:.4f}")

    # Return all learned parameters in a dictionary for prediction
    return {
        'P': P,
        'Q': Q,
        'user_bias': user_bias,
        'item_bias': item_bias,
        'global_mean': global_mean,
        'user_to_index': user_to_index,
        'item_to_index': item_to_index
    }

def predict(model, user_id, item_id):
    """
    Predict a rating for the given user-item pair.
    If either is unknown, return the global average as fallback.
    """
    if user_id in model['user_to_index'] and item_id in model['item_to_index']:
        u = model['user_to_index'][user_id]
        i = model['item_to_index'][item_id]
        pred = model['global_mean'] + model['user_bias'][u] + model['item_bias'][i] + np.dot(model['P'][u], model['Q'][i])
    else:
        # Fallback prediction if user or item was unseen during training
        pred = model['global_mean']
    return clamp_rating(pred)

def generate_predictions(test_data, model, output_file):
    """
    Use the trained model to predict ratings for all user-item pairs in the test set.
    Write the results to a CSV file in the expected format.
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for user_id, item_id, timestamp in test_data:
            rating = predict(model, user_id, item_id)
            writer.writerow([user_id, item_id, rating, timestamp])
    print(f"Predictions written to {output_file}")

def run(train_file, test_file):
    """
    Load training and test data, train the model, and generate predictions.
    """
    print("Loading training data...")
    ratings, users, items = load_train_data(train_file)

    print("Training model...")
    model = train_mf_bias(ratings, users, items)

    print("Loading test data...")
    test_data = load_test_data(test_file)

    print("Generating predictions...")
    generate_predictions(test_data, model, "results.csv")

if __name__ == '__main__':
    # File paths for training and test datasets
    TRAIN_FILE = 'train_20M_withratings.csv'
    TEST_FILE = 'test_20M_withoutratings.csv'
    run(TRAIN_FILE, TEST_FILE)

