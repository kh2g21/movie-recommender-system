# content_based_recommender.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse

# Load Data
def load_data():
    movie_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 
                     'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
                     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    movies = pd.read_csv('ml-100k/u.item', sep='|', names=movie_columns, encoding='latin-1')
    return movies

# Content-Based Recommendation
def content_based_recommendation(movie_title, movies, num_recommendations=5):
    movies['genres'] = movies[movies.columns[5:]].apply(lambda row: ' '.join(row.index[row == 1]), axis=1)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=movies['title'], columns=movies['title'])

    similar_movies = cosine_sim_df[movie_title].sort_values(ascending=False)
    return similar_movies.iloc[1:num_recommendations + 1]

# Main Function
def main():
    parser = argparse.ArgumentParser(description="Content-Based Movie Recommender System")
    parser.add_argument('--movie_title', type=str, required=True, help="Movie title for recommendations")
    parser.add_argument('--num_recommendations', type=int, default=5, help="Number of movie recommendations")
    args = parser.parse_args()

    movies = load_data()

    try:
        recommendations = content_based_recommendation(args.movie_title, movies, args.num_recommendations)
        print(f"Movies similar to '{args.movie_title}':")
        for movie in recommendations.index:
            print(movie)
    except KeyError:
        print(f"Movie '{args.movie_title}' not found in the dataset.")

if __name__ == "__main__":
    main()
