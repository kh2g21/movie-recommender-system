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

Install Dependencies:

Install required packages using pip:
pip install pandas scikit-learn

Download Data:

Download the MovieLens 100k dataset from MovieLens and place the data files (u.data and u.item) in the ml-100k directory.

User-Based Recommendations:

python user_based_recommender.py --user_id 5 --num_recommendations 5

This command will recommend movies for user ID 5 based on what similar users have liked.

Content-Based Recommendations:

python content_based_recommender.py --movie_title "Star Wars (1977)" --num_recommendations 5

This command will recommend movies similar to "Star Wars (1977)" based on genre similarity.
