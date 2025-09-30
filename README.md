# 🎬 IMDB Ratings Analysis — Top 1000 Films and Shows

> A data-driven exploration and classification of top-rated films and TV shows using Python and IMDB data.  
> 📁 Includes data cleaning, feature engineering, visualization, and report generation.

---

## 📊 Project Overview

This project analyzes the **Top 1000 IMDB-rated movies and shows** by leveraging Python-based data processing and statistical analysis. The primary goals are to:

- Clean and enrich the raw dataset from Kaggle
- Engineer meaningful features (e.g., genre-wise ratings, director impact, low-rated clusters)
- Create visual summaries of trends and outliers
- Generate an overview report of findings

---

## 🌐 Data Source

- 📦 Dataset: [Top 1000 IMDB Films and Shows](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows)
- 📁 File: `imdb_top_1000.csv`

---

## 🛠️ Tech Stack

- **Language**: Python 3.13.2
- **Tools**: Jupyter Notebook, Pandas, NumPy, Matplotlib, Seaborn
- **Environment**: Install dependencies via `requirements.txt`

---

## 🔍 Analysis Pipeline

### 1. 📥 Data Acquisition, 🧹 Cleaning and 🧠 Feature Engineering
- Downloaded `imdb_top_1000.csv` from Kaggle
- Imported and explored structure and null values

- Removed missing values, duplicates
- Standardized column names and data formats
- Output: `IMDB_cleaned_dataset.csv`

### 2. 📈 Exploratory Analysis and 📊 Visualization
- Extracted **genre-wise averages** and visualization
- Visualized distribution of ratings, runtime, and gross earnings
- Identified top-rated genres and directors
- Highlighted outliers and rating anomalies
- Aggregated **director-level gross earnings** (`Director_Gross.csv`) and visualization

### 5. 📝 Report Generation
- All findings compiled in `Sesethu_report.pdf`
- Submission notebook: `markdown.ipynb`

---

## 🚀 How to Run

1. Clone the repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt

---

## 🙋🏽‍♂️ Author

Sesethu M. Bango
Aspiring Data Scientist & Film Analytics Enthusiast

📫 Connect on LinkedIn ([https://www.linkedin.com/in/sesethu-bango-197856380/])


