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

The program **learns these patterns** from the training data to make accurate predictions for the test data, saving the predictions into a file `results.csv`.


## Repo Structure

```text
movie-recommender-system/
│
├── train_100k_withratings.csv
├── test_100k_withoutratings.csv
├── source_code.py
└── README.md

```

## Prerequisites

- Python 3.x
- Required Python packages: `numpy`

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
  ```

The script will:
- Load the training data (``train_100k_withratings.csv`).
- Train the matrix factorization model with user and item biases.
- Load the test data (`test_100k_withoutratings.csv`).
- Generate predictions for each user-item pair in the test set (into a file called `results.csv`).
