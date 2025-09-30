import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

    # Check for missing values
class Data_Collection:
    def check_missing(df):
        """
        Function checks for missing values, counts the total number of missing values,
        compares them to the total number of values and prints the results.
        """
        df_check = df.isna()
        count1 = 0 # To count the number of missing and non missing values
        count2 = 0
        for col in df_check.columns:
            for row in df_check[col]:
                if row == True: # Returns a missing value
                    count1 += 1
                    # print(row, col) # Shows where the None values are. NOTE They are in the 'parental_education_level'
                else:
                    count2 += 1# Returns a non missing value
        print('Number of missing values', count1)
        print('Total number of values in the table', count1 + count2)

        missing_percentage = 100.0 * count1 / (count1 + count2)
        print(f'Proportion of missing data to the total: {round(missing_percentage, 2)}% of data is missing.') # percentage is well below 1%, Unlikely to have a major impact on overall findings
        # print(df.isna()) # Returns missing values as true

    # (Function) Remove rows with critical missing data, critical data column specified
    def remove_rows(df, column): 
        for i, row in df.iterrows():
            if pd.isna(row[column]): # How to equate to None/NaN in pandas. Pandas has no None, it's NaN in pandas
                # print(1)
                df.drop(index= i, inplace= True) #NOTE Inplace = True changes the DF
        return df

    # (Function) Remove duplicates
    def dropDuplicates(df):
        df_noDuplicates = df.drop_duplicates() # Removes any row that is a duplicate of any other, leaving only the first occurance
        return df_noDuplicates

class Data_Preperation:
    # (Funtion) Creating a new 'Relesed_Decade' column from the 'Released_Year' column
    def newColumn_ReleasedDecade(df):
        # Convert 'Released_Year' from string to int
        df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors= 'coerce') # 'coerce' to make errors encountered NaN
        df['Released_Year'] = df['Released_Year'].fillna(0)
        df['Released_Year'] = df['Released_Year'].astype(int)
        # Extract Decade from Released_Year
        list1 = []
        for row in df['Released_Year']:
            if row == 0:
                list1.append('0')
            elif row >= 1920 and row < 1930:
                list1.append('1920s')
            elif row >= 1930 and row < 1940:
                list1.append('1930s')
            elif row >= 1940 and row < 1950:
                list1.append('1940s')
            elif row >= 1950 and row < 1960:
                list1.append('1950s')
            elif row >= 1960 and row < 1970:
                list1.append('1960s')
            elif row >= 1970 and row < 1980:
                list1.append('1970s')
            elif row >= 1980 and row < 1990:
                list1.append('1980s')
            elif row >= 1990 and row < 2000:
                list1.append('1990s')
            elif row >= 2000 and row < 2010:
                list1.append('2000s')
            elif row >= 2010 and row < 2020:
                list1.append('2010s')
            elif row >= 2020 and row < 2030:
                list1.append('2020s')
        df_col = pd.DataFrame({'Released_Decade': list1})
        df_res = pd.concat([df, df_col], axis= 1) # Concat the decade column to the main DF
        return df_res
    
    # (Function) Find the top 10 film genres by frequency of occurance
    def genre_Frequencies(df):
        cntDr = 0; cntCr = 0; cntAc = 0; cntAd = 0; cntSc = 0; cntBi = 0; cntHi = 0; cntWe = 0; cntTh = 0
        cntCo = 0; cntRo = 0; cntAn = 0; cntFam = 0; cntWa = 0; cntMu = 0; cntFan = 0; cntMy = 0; cntHo = 0

        for row in df['Genre']:
            if 'Drama' in row: cntDr += 1
            if 'Crime' in row: cntCr += 1 # Not elif because a film may be of more than one genre
            if 'Action' in row: cntAc += 1
            if 'Adventure' in row: cntAd += 1
            if 'Sci-Fi' in row: cntSc += 1
            if 'Biography' in row: cntBi += 1
            if 'History' in row: cntHi += 1
            if 'Western' in row: cntWe += 1
            if 'Thriller' in row: cntTh += 1
            if 'Comedy' in row: cntCo += 1
            if 'Romance' in row: cntRo += 1
            if 'Animation' in row: cntAn += 1
            if 'Family' in row: cntFam += 1
            if 'War' in row: cntWa += 1
            if 'Music' in row: cntMu += 1
            if 'Fantasy' in row: cntFan += 1
            if 'Mystery' in row: cntMy += 1
            if 'Horror' in row: cntHo += 1
        list_freq = pd.array([cntDr, cntCr, cntAc, cntAd, cntSc, cntBi, cntHi, cntWe, cntTh, cntCo, cntRo, cntAn, cntFam, cntWa, cntMu, cntFan, cntMy, cntHo])        
        list_Genre = pd.array(['Drama','Crime','Action','Adventure','Sci-Fi','Biography','History','Western','Thriller',
                            'Comedy','Romance','Animation','Family','War','Music','Fantasy','Mystery','Horror'])
        df_Genre = pd.DataFrame({'Film_Genre': list_Genre, 'Frequency': list_freq})
        
        df_Genre.sort_values('Frequency', ascending = False, inplace = True)
        df_Genre_top10 = df_Genre[df_Genre['Frequency'] >= 64] # 64 being the 10th highest value, hence everything below is not important
        return df_Genre_top10

                    
class Data_Visualisation:
    # (Function) Create a histogram of column_1 vs column_2 of a dataframe
    def doubleHistogram(df, column1, column2, num_of_bins): # specify these characteristics of your histogram
        df_col1 = df[column1]
        df_col2 = df[column2]
        title1 = 'Histogram of ' + column1
        title2 = 'Histogram of ' + column2
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4)) # 1 row, 2 columns of subplots

        axes[0].hist(df_col1, bins= num_of_bins, color= 'skyblue', edgecolor= 'black')
        axes[0].set_title(title1)
        axes[0].set_xlabel('Adjusted ' + column1)
        axes[0].set_ylabel('Frequency')
        
        axes[1].hist(df_col2, bins= num_of_bins, color= 'red', edgecolor= 'black')
        axes[1].set_title(title2)
        axes[1].set_xlabel(column2)
        axes[1].set_ylabel('Frequency')

        axes[1].grid(True)
        axes[0].grid(True)
        plt.show()
    
    # (Function) Create a histogram of one of the columns of a dataframe
    def singleHistogram(df, column_name, num_of_bins, xlabel, ylabel): 
        study_time = df[column_name]
        title = 'Histogram of ' + column_name
        plt.hist(study_time, bins= num_of_bins, color= 'skyblue', edgecolor= 'black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()
    
    # (Function) Creating a box plot using DF and column name
    def bar_plot(df, x_column, y_column):
        colours = ['red', 'orange', 'yellow']
        plt.tick_params(axis='x', labelsize=7) #NOTE change tick labels in the plot
        plt.bar(df[x_column], df[y_column], color= colours)
        
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title('Bar plot of ' + x_column + ' vs ' + y_column)
        
        def millions(x, pos): # I can do this! Interesting.
            return f'{float(x/1_000_000.0)}M'
        
        # Custom set ticker
        if y_column == 'Average_Gross':
            plt.gca().yaxis.set_major_formatter(FuncFormatter(millions)) # Only for a specific column name
                
        plt.tight_layout()
        plt.show()
    
    
        
    # (Function) Create a scatter plot of one column vs another
    def scatterPlot(df, column_name1, column_name2): # Specify these characteristics of your skatter plot
        x = df[column_name1]
        y = df[column_name2]
        title_name = 'Skatter plot of ' + column_name1 +  ' vs ' +  column_name2

        plt.scatter(x, y, alpha= 0.7, cmap= 'viridis', marker= '.')
        plt.xlabel(column_name1)
        plt.ylabel(column_name2)
        plt.title(title_name)
        
        def millions(x, pos): # I can do this! Interesting.
            return f'{float(x/1_000_000.0)}M'
        
        # Custom set ticker
        plt.gca().xaxis.set_major_formatter(FuncFormatter(millions))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(millions))
        
        plt.show()
        
    # (Function) Create boxplots of scores by diet quality
    def boxplots(df, column_1, column_2):
        sns.boxplot(x= column_1, y= column_2, data= df[[column_1, column_2]])
        title_str = 'Box Plot of ' + column_1 + ' by ' + column_2
        plt.title(title_str)
        plt.xlabel(column_1)
        plt.ylabel(column_2)
        plt.show()