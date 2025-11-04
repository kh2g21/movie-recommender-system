# Movie Recommender System

## Overview

This is a **collaborative filtering movie recommender system** using **matrix factorization with bias terms**. The system predicts user ratings for movies based on historical user-item interactions. It is designed to provide personalized movie recommendations for users.

## source_code.py

- Predicts movie ratings for users based on past ratings.
- Learns patterns for both users and movies (some users rate higher/lower, some movies are generally more liked).
- Handles datasets with 100k users and movies efficiently.
- Saves predictions in a CSV file for easy analysis.

The system predicts a rating for a movie using a combination of:

- The **average rating** across all users  
- The **user's personal bias** (some users give higher/lower ratings)  
- The **movie's popularity bias** (some movies are generally liked more)  
- **Hidden factors** that represent user preferences and movie features  

The program **learns these patterns** from the training data to make accurate predictions for the test data.


## Repo Structure

train_100k_withratings.csv - Training data: user, movie, rating, timestamp
test_100k_withoutratings.csv - Test data: user, movie, timestamp
source_code.py - Code for the recommender system; full pipeline for training and predicting ratings
README.md (this file)

### Prerequisites

- Python 3.x
- Required Python packages: `pandas`, `scikit-learn`

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/kh2g21/movie-recommender-system.git
   cd movie-recommender-system

2. **Install Dependencies**:
   Install required packages using pip:
   ```bash
   pip install numpy 

3. **Run recommender system**:
  Either run using Python IDE or if using the terminal:
 ```bash
python source_code.py

