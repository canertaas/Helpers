## Helpers
Helper functions I use frequently.



## Explore and Visualize the Data

##### Exploratory Data Analysis
````
#Showing pairwise relationships in a dataset
sns.pairplot(df, hue="x", kind="reg", palette="husl")
````

````
#Subplotting with histogram, violin plot and box plot
f, axes = plt.subplots(4, 3, figsize=(15, 15))
sns.despine(right=True)
count=0
for i in range(4):      
    sns.histplot(df.iloc[:,count], ax=axes[i,0], kde=True)
    sns.boxplot(x=df.iloc[:,count], ax=axes[i,1])
    sns.violinplot(x=df.iloc[:,count], ax=axes[i,2])
    count=count+1
````

````
#Normality checking with QQ Plot
import statsmodels.api as sm 

sm.qqplot(df.iloc[:,count], line="s")

````

````
#Number of outliers in each column
numeric_columns=["x","y"]
for columns in numeric_columns:
    quantiles=df[columns].quantile(q=[0.25,0.50,0.75]).values
    q1=quantiles[0]
    q2=quantiles[1]
    q3=quantiles[2]
    iqr=q3-q1
    outliers=df[(df[columns] < q1-1.5*iqr ) | (df[columns] >  q3+ 1.5*iqr)][columns]
    print("number of outliers in", columns, ":", (len(outliers))) 
````

````
#To see the effect of outliers on mean and standard deviation in selecting the column
remove_outliers=df[(df["x"] >= q1-1.5*iqr ) &(df["x"] <=  q3+ 1.5*iqr)]["x"]
mean_change= (np.mean(df["x"]) - np.mean(remove_outliers) )/ np.mean(df["x"])
std_change= (np.std(df["x"]) - np.std(remove_outliers) )/ np.std(df["x"])
print("change in mean: %.2f "% mean_change)
print("change in std: %.2f "% std_change)
````
##### Correlation
````
#Finding the relations between the variables with correlation matrix.
plt.figure(figsize=(10,10))
correlation = df.corr()
sns.heatmap(correlation, cmap= "BrBG", annot= True)
plt.show()
````

````
#Looking Pearson Correlation of two varaibles
from scipy.stats import pearsonr

corr, p_value = pearsonr(df["x"],df["y"])
print("Pearsons correlation: %.6f" % corr)
````
##### Missing Data
````
#Show the missing data in a bar chart
import missingno as msno

msno.bar(df)

#Show the missing data missingness pattern
msno.matrix(df)

#Show the missing data in a heatmap
msno.heatmap(df)
````

## Preprocessing
##### Encoders
````
#Label Encoder
from sklearn.preprocessing import LabelEncoder

le = preprocessing.LabelEncoder()
df["x_label"] = le.fit_transform(df["x"]) 
````

````
#One Hot Encoder
from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder()
encoded = onehot_encoder.fit_transform(df[['x1','x2']])
df_newpredictors = onehot_encoder.get_feature_names(['x2', 'x3'])
df_add = pd.DataFrame(encoded.toarray(), columns=df_newpredictors)
df_ohe = df.merge(df_add, left_index=True, right_index=True)
````

##### Scaling
````
#Use it in a normal distribution
from sklearn.preprocessing import StandardScaler

standardscaler= StandardScaler()
scaled_data=standardscaler.fit_transform(df)
scaled_data=pd.DataFrame(scaled_data, columns=df.columns)
````

````
from sklearn.preprocessing import MinMaxScaler

scaler= MinMaxScaler()
scaled_data=scaler.fit_transform(df)
scaled_data=pd.DataFrame(scaled_data, columns=df.columns)
````
## Machine Learning
