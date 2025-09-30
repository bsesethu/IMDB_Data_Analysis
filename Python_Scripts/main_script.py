import pandas as pd
import numpy as np
from functions_Assignment_2 import Data_Collection as DaC 
from functions_Assignment_2 import Data_Preperation as DaP
from functions_Assignment_2 import Data_Visualisation as DaV
import matplotlib.pyplot as plt
import seaborn as sns

# Downloading file from kagglehub
# path = kagglehub.dataset_download("harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")
# print( "Path to dataset files:" , path)
print('File downloaded successfully')
    
df = pd.read_csv('imdb_top_1000.csv') # Doesn't want to read imdb file in the .cache folder. Had to move it to the local folder
print('Original DataFrame first 5 entries')
print(df.head())
print('\nDataFrame core info:')
print(df.info())

# Check for mising values and document their percantage
    # Using functions written into functions_Assignment_2.py file
print('\n')
missing = DaC.check_missing(df) 

# Remove rows with missing critical data, from the following columns
df_new = DaC.remove_rows(df, 'Meta_score')
df_new = DaC.remove_rows(df_new, 'Gross')
print('\nAfter removal of null values')
missing_df_new = DaC.check_missing(df_new)


# Phase 2 Data preperation
# Drop duplicates
df_noDuplicates = DaC.dropDuplicates(df_new)
print('\nAfter dropped duplicates, Core info again:')
print(df_noDuplicates.info()) # There are no duplicates. Function would've kept the first occurance

# Testing the 'dropDuplicates' function
# df_G = pd.DataFrame({'Name': ['Cash', 'Money', 'Money', 'Ross'],
#                      'Sex': ['Yes', 'No', 'No', 'Yes'],
#                      'Number': [1, 5, 5, 8]
#                      })
# df_noDup = DaC.dropDuplicates(df_G)
# print(df_noDup) # The function works

# Convert runtime to numerical
newRuntime = []
cnt = 0
for row in df_noDuplicates['Runtime']:
    newRuntime.append(float(row.split(' ')[0])) # New list of numerical values only
    
# Reset indices so there are no skipped values
df_reset = df_noDuplicates
df_reset.reset_index(inplace= True) #NOTE Index values are now reset for all affilliated DFs being used from now on

df_noDuplicates['Runtime'] = pd.DataFrame({'Runtime': newRuntime}) # Make newRuntime list a DF column by overriding the original
print('\nCheck Runtime column values dtype:')
print(df_noDuplicates['Runtime'].info())

# For some reason I can't explain it's returning NaN values for the last 200 or so values of the 'Runtime' column, Let's fix this.
# Sorted: Reason for the issue is the index values of the df_duplicates DF, there are 750 values but indices go up to 997

# Extract Decade from Released_Year
df_res = DaP.newColumn_ReleasedDecade(df_noDuplicates)
print('\nFirst 5 entries in Released_Year and Released_Decade columns')
print(df_res[['Released_Year', 'Released_Decade']].head()) # It checks out

# Create a Lead_Actors column combining Star1, Star2, Star3, Star4.                          #NOTE IMPORTANT FUNCTION
df_res['Lead_Actors'] = df_res[['Star1', 'Star2', 'Star3', 'Star4']].agg(', '.join, axis= 1) # New column, all 4 cell values in one, using aggregate funtion where all values are joined along the columns axis
print('\nFirst 5 entries in new Lead_Actors column')
print(df_res['Lead_Actors'].head())


# Phase 3: Data Visualisation
# Histogram
# Convert 'IMDB_Rating' to a comparative rating with 'Meta_score', rating out of 100 not 10
df_res['IMDB_Rating'] = df_res['IMDB_Rating'] * 10
hist = DaV.doubleHistogram(df_res, 'IMDB_Rating', 'Meta_score', 24)

# Bar plot of Genre frequency
    # First need to find the top 10 genre frequency of occurance
df_Genre_top10 = DaP.genre_Frequencies(df_res) # Applying the function
print('\nTop 10 genres by Frequency')
print(df_Genre_top10)
# Generating the bar plot
bar = DaV.bar_plot(df_Genre_top10, 'Film_Genre', 'Frequency')

# Scatter plot of Gross vs. No_of_votes.
    # First convert Gross to number value
df_res['Gross'] = df_res['Gross'].str.replace(',', '') # Remove ','
df_res['Gross'] = pd.to_numeric(df_res['Gross'], errors= 'coerce') # 'coerce' to make errors encountered NaN
# print(df_res[df_res['Gross'] > 800e6]) # Just checking
scatter = DaV.scatterPlot(df_res, 'Gross', 'No_of_Votes')

# Box plot of IMDB_Rating by Certificate
df_res['Certificate'] = df_res['Certificate'].fillna('Unknown') # fil NaN values
# Return IMDB to a rating out of 10
df_res['IMDB_Rating'] = df_res['IMDB_Rating'] / 10 # Converting column values back to original state
box = DaV.boxplots(df_res, 'IMDB_Rating', 'Certificate')


# Phase 4: Applied Statistical Analysis
    # Compute mean, median, std for Gross, No_of_votes, IMDB_Rating.
print('\nComputed values for mean, median and standard deviation for the following columns:')
# print(df_res.info()) # Check dtypes
G_mean = round(np.mean(df_res['Gross']), 0)
G_median = round(np.median(df_res['Gross']), 0)
G_std = round(np.std(df_res['Gross']), 0)
print(f'Gross; mean: {G_mean}, median: {G_median}, std: {G_mean}') # G_std = G_mean for an exponential distributiion, of which this is.

# Histogram of Gross distribution
# hist_G = DaV.singleHistogram(df_res, 'Gross', 24, 'Gross', 'Frequency') # Checking the distribution, it's exponential hence std == mean
# Histogram of No_of_votes distribution
# hist_G = DaV.singleHistogram(df_res, 'No_of_Votes', 24, 'No_of_Votes', 'Frequency') # Checking the distribution, also exponential.

N_mean = round(np.mean(df_res['No_of_Votes']), 0)
N_median = round(np.median(df_res['No_of_Votes']), 0)
N_std = round(np.std(df_res['No_of_Votes']), 0)
print(f'No_of_Votes; mean: {N_mean}, median: {N_median}, std: {N_mean}') # Also exponential distribution

I_mean = round(np.mean(df_res['IMDB_Rating']), 1)
I_median = round(np.median(df_res['IMDB_Rating']), 1)
I_std = round(np.std(df_res['IMDB_Rating']), 1)
print(f'IMDB_Rating; mean: {I_mean}, median: {I_median}, std: {I_std}')

# Calculate Pearson correlation between Gross and No_of_votes
corr = round(df_res['Gross'].corr(df_res['No_of_Votes']), 5) 
print('\nCorrelation between Gross and No_of_Votes: ', corr)

# Use IQR to identify outliers in Gross
q1 = np.percentile(df_res['Gross'], 25)
q3 = np.percentile(df_res['Gross'], 75)
IQR = q3 - q1
bound_lower = q1 - (1.5 * IQR)
bound_upper = q3 + (1.5 * IQR)
outliers = []
for row in df_res['Gross']:
    if row < bound_lower:
        outliers.append(row)
    elif row > bound_upper:
        outliers.append(row)
print('\nBoundaries for outliers in the Gross column: ', bound_lower, bound_upper)


# Phase 5 Advanced Analysis
    # Director Analysis
df_directors = df_res.groupby('Director').agg(Sum_Gross= ('Gross', 'sum'), Average_Gross= ('Gross', 'mean'))
df_dir_sorted = df_directors.sort_values(by= 'Average_Gross', ascending= False)
print("\nComparing directors' Gross profits:")
print(df_dir_sorted.head(10))
print('Anthony Russo is the director with the highest average Gross')

# Plot using Bar plot
df_dir_5 = df_dir_sorted.drop(df_dir_sorted.index[5:]) # Top 5 rows only
# df_dir_5.to_csv('Director_Gross.csv') #NOTE Done. Now to read it again. Did this to solve the 'KeyError'
df_5 = pd.read_csv('Director_Gross.csv') # KeyError navigated successfully
bar = DaV.bar_plot(df_5, 'Director', 'Average_Gross')

    # Actor analysis
df_res['IMDB_Rating'] = df_res['IMDB_Rating'] 
df = (df_res[df_res['IMDB_Rating'] > 8.5]) 
df_Actors = df.groupby('Star1').agg(Count_IMDB= ('IMDB_Rating', 'count'))
df_Actors.sort_values(by= 'Count_IMDB', ascending= False, inplace= True)
print('\nTop lead actors by IMDB rating above 8.5:')
print(df_Actors.head())
print('Actors Tom Hanks and Elijah Wood appear most frequntly, each have 3 films in the IMDB count of scores above 8.5')

print('\nGross profit of actor pairs [Star1 + Star2]')
df_pairs = df_res.groupby(['Star1', 'Star2']).agg(Avg_Gross= ('Gross', 'mean'))
# df_pairs.to_csv('two_stars.csv')
df_2Stars = pd.read_csv('two_stars.csv') # Re-loading the df_pairs DF
# print(df_2Stars.head(30))
df_2Stars.sort_values(by= 'Avg_Gross', ascending= False, inplace= True)
print(df_2Stars.head(12))
print("There doesn't seem to be any data that suggests two specific actors working together result in consistently good gross profit figures.")

    # Genre preference
print('\nGenre most associated with high IMDB rating:')
df_genre = round(df_res.groupby('Genre').agg(Avg_IMDB_Rating= ('IMDB_Rating', 'mean')), 1)
# df_genre.to_csv('genre_rating.csv')
df_genre_rating = pd.read_csv('genre_rating.csv') # Re-loaded
df_genre_rating.sort_values(by= 'Avg_IMDB_Rating', ascending= False, inplace= True)
df_rating = df_genre_rating[df_genre_rating['Avg_IMDB_Rating'] > 8.2] # Limit the result
print(df_rating)
print('The Genre most associated with high IMDB rating is [Crime, Mystery, Thriller], by a very slim margin over the others')

    # Heatmap to show genre vs rating trends
# Create a pivot table
pivot_df = df_rating.pivot_table(index='Genre', values='Avg_IMDB_Rating', aggfunc='mean')
# Plot heatmap
plt.figure(figsize=(8, 6))
plt.tick_params(axis='x', labelsize=7) # Smaller font
sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt=".2f", linewidths=0.5)
plt.title('Heatmap of Genre vs IMDB_Rating')
plt.show()


# Cleaned DataFrame
# df_cleaned = df_res.to_csv('IMDB_cleaned_dataset.csv')
# FIN --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------