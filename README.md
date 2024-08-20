# Movie Recommender System

## Overview

This repository contains two Python scripts for recommending movies using different methodologies:

1. **User-Based Collaborative Filtering**: Recommends movies to users by analyzing the preferences of similar users.
2. **Content-Based Filtering**: Recommends movies similar to a given movie based on genre similarities.

## Features

- **User-Based Collaborative Filtering**:
  - Provides movie recommendations based on user preferences.
  - Identifies users with similar tastes and suggests movies they liked.

- **Content-Based Filtering**:
  - Recommends movies similar to a given movie based on genre information.
  - Uses TF-IDF vectorization to compute similarity between movies.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python packages: `pandas`, `scikit-learn`

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/movie-recommender-system.git
   cd movie-recommender-system

2. **Install Dependencies**:
   Install required packages using pip:
   ```bash
   pip install pandas scikit-learn

3. **Download Data**:
   Download the MovieLens 100k dataset from MovieLens and place the data files (u.data and u.item) in the ml-100k directory.

## Usage

- **User-Based Recommendations**:
  - **Description**: Get movie recommendations for a user based on the preferences of similar users.
  - **Command**:
    ```bash
    python user_based_recommender.py --user_id 5 --num_recommendations 5
    ```
  - **Parameters**:
    - `--user_id`: The ID of the user for whom recommendations are to be generated.
    - `--num_recommendations`: The number of movie recommendations to display.
  - **Example Output**:
    ```plaintext
    Movies recommended for user 5:
    Movie 1
    Movie 2
    Movie 3
    ```

- **Content-Based Recommendations**:
  - **Description**: Get movie recommendations similar to a given movie based on genre similarity.
  - **Command**:
    ```bash
    python content_based_recommender.py --movie_title "Star Wars (1977)" --num_recommendations 5
    ```
  - **Parameters**:
    - `--movie_title`: The title of the movie for which similar recommendations are to be made.
    - `--num_recommendations`: The number of similar movies to display.
  - **Example Output**:
    ```plaintext
    Movies similar to 'Star Wars (1977)':
    Movie A
    Movie B
    Movie C
    ```
