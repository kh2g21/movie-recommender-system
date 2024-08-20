
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import argparse

# Load Data
def load_data():
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', usecols=[0, 1], names=['movie_id', 'title'])
    return ratings, movies

# Preprocess Data
def preprocess_data(ratings, movies):
    movie_data = pd.merge(ratings, movies, on='movie_id')
    user_movie_matrix = movie_data.pivot_table(index='user_id', columns='title', values='rating')
    return user_movie_matrix

# Calculate User Similarity
def calculate_user_similarity(user_movie_matrix):
    user_similarity = cosine_similarity(user_movie_matrix.fillna(0))
    return pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# Recommend Movies
def recommend_for_user(user_id, user_similarity_df, user_movie_matrix, num_recommendations=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    top_similar_users = similar_users.index[1:num_recommendations + 1]

    recommendations = []
    for user in top_similar_users:
        user_recommendations = user_movie_matrix.loc[user].dropna().index.tolist()
        recommendations.extend(user_recommendations)

    recommendations = list(set(recommendations) - set(user_movie_matrix.loc[user_id].dropna().index))
    return recommendations[:num_recommendations]

# Main Function
def main():
    parser = argparse.ArgumentParser(description="User-Based Movie Recommender System")
    parser.add_argument('--user_id', type=int, required=True, help="User ID for recommendations")
    parser.add_argument('--num_recommendations', type=int, default=5, help="Number of movie recommendations")
    args = parser.parse_args()

    ratings, movies = load_data()
    user_movie_matrix = preprocess_data(ratings, movies)
    user_similarity_df = calculate_user_similarity(user_movie_matrix)

    try:
        recommendations = recommend_for_user(args.user_id, user_similarity_df, user_movie_matrix, args.num_recommendations)
        print(f"Movies recommended for user {args.user_id}:")
        for movie in recommendations:
            print(movie)
    except KeyError:
        print(f"User ID {args.user_id} not found in the dataset.")

if __name__ == "__main__":
    main()


