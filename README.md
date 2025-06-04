<h1 align="center">Spotify Project</h1>

### Project Overview

Spotify's music streaming service allows users to create personalized playlists and manage family accounts.  
Due to a server hack, data regarding `user` and `top_year` for some songs were deleted.  
The task is to:
 - Reconstruct the Top-of-the-Year playlists for each user from 2018 to 2024 using the available data
 - Reconstruct the missing user: Recover the missing songs for each user’s playlist

### Project summary

The Jupyter Notebook is structured in 5 ways:
1. Exploratory Data Analysis (EDA)
2. Data Preparation (preprocessing)
3. User Playlist Reconstruction using Machine Learning
4. Top-Year Playlist Reconstruction using Machine Learning
5. Reconstruction the Playlist Data



<h2 align="center">  :star: Key Result  :star: </h2>

This **data imputation** problem was approached as a **classification problem**.  

**User playlist** was reconstructed using **OneVSRestClassifier** (Logistic Regression) with F1-score of **87%**.  
- **5 class** to predict: {`alpha`, `beta`, `gamma`, `delta`, `epsilon`}
- Before feature engineering, model performance was at best **40%** using more complex model (SVC, Random Forest, OvR with degree-3 poly kernel)
- To prevent over-reliance on the newly feature created, **L2 penalty** (Ridge penalization) was added. It reduce overfitting and give more room for other features

**Top-year playlist** was reconstructed using **Decision Tree Classifier** with F1-score of **76%**.
- **7 class** to predict: {`2018`, `201`9, `2020`, `2021`, `2022`, `2023`, `2024`}
- Before feature engineering, model performance was at best **40%** using more complex model (k-NN, Random Forest Classifier, Gradient Boosting)
- To prevent overfitting, **hyperparameter tuning** using grid was performed and 

**Output**: The dataset with the reconstructed user and top_year predicted by both models.

___
Comprehensive Analysis below
___

<h1 align="center">Exploratory Data Analysis EDA</h1>

The Exploratory Data Analysis is structured in 9 sub-sections:

**1.** Data Exploration - **2.** Handling missing values - **3.** Check for duplicate - **4.** Explore the Target variable - **5.** Data Type - **6.** Descriptive Analysis - **7.** Correlation Analysis - **8.** Feature-Target Relationship Analysis (User) - **9.** Feature-Target Relationship Analysis (Top_year)

### Exploratory from the Dataset
We are trying to predict the missing `user` and `top_year` (far-left of the dataframe)  
The first row is missing the user and top_year that was deleted and that we need to predict

We have 3 mains features: 
- metadata about the track: name, album, artist, duration (length)
- release date and top_year
- song characteristic like tempon energy, accousticness

Target to predict are the missing value in:
- user: {`alpha`, `beta`, `gamma`, `delta`, `epsilon`}
- top_year: {`2018`, `201`9, `2020`, `2021`, `2022`, `2023`, `2024`}

*__Table__: Sample of the dataset*
| name | album | artist | release_date | length | popularity | acousticness | danceability | energy | instrumentalness | liveness | loudness | speechiness | tempo | valence | time_signature | key | mode | uri | release_year | top_year | user |
|------|-------|--------|--------------|--------|------------|--------------|--------------|--------|------------------|---------|----------|-------------|-------|---------|----------------|-----|------|-----|--------------|----------|------|
| Variations on a Polish Theme, Op. 10: No. 5 Andantino | Szymanowski: Piano Works, Vol. 2 | Karol Szymanowski | 06/12/1996 | 76933 | 53 | 0.9960 | 0.329 | 0.00695 | 0.866000 | 0.0906 | -34.227 | 0.0448 | 70.295 | 0.238 | 4 | 11 | 0 | spotify:track:3bcdLMrAxrfn5dxInjIdI2 | 1996 | unknown | unknown |
| Je vous trouve un charme fou - En duo avec Gaëtan Roussel | Il suffit d'y croire (Version deluxe) | Hoshi | 2018-11-30 | 172626 | 62 | 0.6220 | 0.615 | 0.59900 | 0.000008 | 0.1920 | -8.715 | 0.2530 | 86.976 | 0.626 | 4 | 1 | 1 | spotify:track:0C2yaSWVgCUiiqPyYxSOkd | 2018 | 2024 | delta |
| Me Gusta | On ira où ? | DTF | 2019-10-11 | 175269 | 72 | 0.4130 | 0.834 | 0.73400 | 0.000040 | 0.1130 | -5.948 | 0.3410 | 89.989 | 0.356 | 4 | 6 | 0 | spotify:track:6P3FBaZfUjeWYExU2ShaPZ | 2019 | 2022 | gamma |
| L’amour en Solitaire | Petite Amie (Deluxe) | Juliette Armanet | 2018-02-02 | 175266 | 0 | 0.4040 | 0.797 | 0.50600 | 0.000153 | 0.2550 | -6.774 | 0.0327 | 128.027 | 0.539 | 4 | 5 | 0 | spotify:track:2tn51grfchxArwPXeXkoX5 | 2018 | 2020 | gamma |

### General data

- The dataset contains **3600** rows.  
- **2 duplicate entries** were identified and removed. 
- 9 songs with no name, album and artist = `Various artist`. After verification, it is not an input error.  
  Given their low occurrence, they were removed to maintain data quality without significantly impacting model performance.
- Predicted class are **perfectly balanced** (no bias from class imbalanced)

*__Contingency table__: between user and top_year*
| top_year | alpha | beta | delta | epsilon | gamma | unknown |
|----------|-------|------|-------|---------|-------|---------|
| 2018     | 100   | 100  | 100   | 100     | 100   | 0       |
| 2019     | 100   | 100  | 100   | 100     | 98    | 0       |
| 2020     | 100   | 100  | 100   | 95      | 100   | 0       |
| 2021     | 100   | 100  | 96    | 100     | 100   | 0       |
| 2022     | 100   | 100  | 100   | 100     | 100   | 0       |
| 2023     | 100   | 100  | 100   | 100     | 100   | 0       |
| 2024     | 100   | 100  | 100   | 100     | 100   | 0       |
| unknown  | 0     | 0    | 0     | 0       | 0     | 100     |

### Descriptive Analysis

Each track has distinct musical features that differentiate it from others.  
*Source: https://developer.spotify.com/documentation/web-api/reference/get-audio-features*

How about an real example to illustrate it?

🎵 Don't Stop Me Now (Remastered 2011) - Queen 🎵  
[Listen on Spotify](https://open.spotify.com/track/5T8EDUDqKcs6OSOwEsfqG7)  

| **Song**                    | **Album**            | **Artist** | **Release Date** | **Length**     | **Popularity** | **Acousticness** | **Danceability** | **Energy** | **Instrumentalness** | **Liveness** | **Loudness** | **Speechiness** | **Tempo** | **Valence** | **Key**  | **Top Year** | **User** |                                            |
|-----------------------------|----------------------|------------|------------------|----------------|----------------|------------------|------------------|------------|----------------------|--------------|--------------|-----------------|-----------|-------------|----------|--------------|----------|--------------------------------------------------|
| Don't Stop Me Now - Remastered 2011 | Jazz (2011 Remaster) | Queen      | 1978-11-10       | 3m 29s         | 85%            | 4.75%            | 55.9%            | 86.8%      | 0.02%                | 77.6%        | -5.28 dB     | 17%             | 156 BPM   | 60.9%       | F Major | 2020         | Alpha

| **Feature**         | **Value**     | **Description**                                  |
|---------------------|---------------|--------------------------------------------------|
| **Acousticness**     | 4.75%         | Mostly electric, with little acoustic presence   |
| **Danceability**     | 56%           | Energetic, catchy rhythm, dance-friendly         |
| **Energy**           | 86.8%         | High energy, powerful, perfect for a lively vibe |
| **Instrumentalness** | 0.02%         | Vocal-driven with minimal instrumental sections  |
| **Liveness**         | 77.6%         | Studio recording with live performance energy    |
| **Loudness**         | -5.28 dB      | Loud and dynamic, fitting for rock anthems       |
| **Speechiness**      | 17%           | Some spoken moments, mainly singing             |
| **Tempo**            | 156 BPM       | Fast-paced, adding to the upbeat vibe            |
| **Valence**          | 60.9%         | Positive, fun, and uplifting                    |
| **Key**              | F Major       | Bright and open, matching the energetic tone    |

Here is the boxplot for each column. *The complete analysis is in the notebook*

*__Boxplot__ of the differents musical features*
![Alt text](https://github.com/RobertChanData/spotify_project/blob/main/Screenshot/Spotify_1.PNG?raw=true)

### Correlation Analysis:

*__Correlation Matrix__ of the differents musical features*
![Alt text](https://github.com/RobertChanData/spotify_project/blob/main/Screenshot/Spotify_2.PNG?raw=true)

We are checking for multicollinearity.
- `loudness` and `energy` are highly correlated (0.828). There may be multicollinearity bias here
- `acousticness` is strongly negatively correlated with `loudness` (-0.677) and `energy` (-0.786)
- `danceability` and `valence` are positively correlated (0.584)

`loudness`, `energy`, `danceability` and `valence` seems to all be correlated together to some extend.  
We should keep that in mind during the next step.

**Action**: Also, removing either `loudness` or `energy` should be done to avoid multicollinearity bias.


### Feature-Target Relationship Analysis (User):

In this section, we compare the distribution of differents features song regarding their using to identify differentiating pattern
*(Full visualisation and explication are in the notebook)*

![Alt text](https://github.com/RobertChanData/spotify_project/blob/main/Screenshot/Spotify_3.PNG?raw=true)

Reminder: `unknown` is not a separate group but a missing value from one of the 5 users.

- **`gamma`**: Most distinct taste in music. Prefers **energetic**, **danceable**, and **loud** tracks. Also, likes **low instrumentalness**, **high speechiness**, and popular songs. *(Pop/Electronic/House?)*
- **`epsilon`**: Similar to `gamma`, but with **lower valence**, **lower speechiness**, **higher instrumentalness**, and prefers **less-known artists**. *(Experimental/Indie?)*
- **`alpha`**: Notable for **lower acousticness**, **higher energy**, and **higher valence** compared to `beta` and `delta`.
- **`beta`**: Notable for **outliers in song duration**, **wide spread in acousticness** and **energy**, and listens mainly to **mode = 0** songs. *(Tendency for melancholic sound?)*
- **`delta`**: Has **low instrumentalness** compared to `alpha` and `beta`.

Overall, **gamma** is the most distinct, followed by **epsilon**. **Alpha**, **beta**, and **delta** are similar, with **beta** standing out due to some unique characteristics.

### Feature-Target Relationship Analysis (Year):

Same for Year

Since it combines user tastes, the top songs of the year become somewhat irrelevant. There are noticeable differences, but nothing truly distinguishable based on song features.
It might be better to split the analysis by user and top year, or explore metadata like artist or group instead.


<h1 align="center">Data Preparation</h1>

This step deal mainly with data preprocessing. Below are the step implemented in the notebook.

Step implemented:
- remove duplicate (2 rows)
- Drop missing values (9 rows)
- Convert `time_signature` and `mode` to categorical value
- Convert time from millisecond to second
- Perform scaling of data (Z-score normalization)
- We separate unknow from known in the dataset. **We do not want the unknown value (which we will need to predict) to be part of any train or test dataset**

About data standardization:
- **Transform the data** to have a mean = 0 and std = 1 to ensure numeric feature have the same scal
- **Improve model convergence** (Gradient descent for Logistic Regression)
- **Equal weight for feature**: avoid bias introduced by different scale (here millisecond) compare to song feature (ranging from 0 to 1)
- **Distanced-base algorithm**: avoid biasing kNN algorithm

<h1 align="center">Model for User</h1>

We decide to resolve the User prediction by treating it like a **multi-classification** problem.  
There is 5 class to predict, and we would like to mix some **songs features** with **past listening behaviour** to get accurate prediction.  
The idea is that a specific user is most likely to listen to songs that belong to an artist it has listened to the past.

### General rule

- Data is standardized (Z-score) to avoid bias introduced by difference in scale
- Categorical label are handled with One-Hot Encoding
- The class are already balanced (no need for SMOTE, random under-sampling or use Ensemble method)
- Train / Test split (70% - 30%). No validation set for this one (an error but I was still learning)
- Simple model are prefer for better explainability (personal preference)
- Metric is F1-score. In our case, accuracy can also be use (as class are perfectly balance, it is the same)
  All error have the same cost so no emphasise on precision or recall
- Data leakage is carefully check and verified (target leakage, data split leakage, temporal leakage)   

### Feature Engineering
for the feature addition, we implemented:  
Number of time a user has listened to an artist in the past  
This feature capture the **user’s past behavior toward certain artists** (and songs), which was lost due to data deletion.  
In the code, it is calculated by counting the number of time the song was listened by all the user

#### 🚨 Important Note 🚨
This is **not** a standard supervised learning problem where we predict an unknown future user.

Instead, we're **recovering missing user data** that originally existed but was later deleted.  
The goal is to **restore** this information using past listening patterns that were present before deletion.

Because of this, using **target-related features** (like how many times a user has listened to an artist) may **<span style="color:red;">not be data leakage in this specific case</span>** it's leveraging information that was already there.

In a **classic prediction setup**, we **<span style="color:red;">would never</span>** use these metrics to predict future users.  
But here, they help **reconstruct lost data**, making them **valid for our task**.

By implementing this new feature, we improve the most basic model from **40%** to over **85%**

### OneVsRestClassfier (Logistic Regression) and why this choice

Initially, we try various model from One Vs REst Classifier (OvR), Support Vector Classifier (SVC), Random Forest Classifier (RFC) or OvR with 3 degree polynomial but performance keep plateauing at **40%**.

Since hyperparameter tuning is unlikely to significantly improve an underfitting model, the primary issue lies in the model's inability to capture certain patterns. Initially, focusing only on song characteristics for predictions was not ideal because users may listen to a wide range of music, which dilutes the data for individual songs. 

Additionally, it makes sense that users have favorite artists and tend to listen to music from those artists frequently. To address this, we incorporated this information into the model by creating a new feature.

Then OneVsRestClassfier (polynomial = 1) achieve over 85% accuracy. Main advantages:
- Simple and effective model at multi-class classification
- Model interpretation is easier than other model
- Work well with linear decision boundaries that we know have some thanks to initial EDA

**Issue:** The model become too reliant on the new feature, potentially overshadowing other important features.

**Solution:** To counteract this, we apply L2 Regularization (Ridge Regression). This technique penalizes large coefficients, discouraging the model from relying too heavily on any one feature. As a result, other features can maintain their importance and contribute more equally to the model.

To verify the effectivness of L2 Regularization, we extracted the coefficient on the table below:

*_Table of Coefficient_: One Vs Rest Classifier*

| intercept | length | popularity | acousticness | danceability | instrumentalness | liveness | loudness | speechiness | tempo | valence | time_signature | key | mode | release_year | alpha | beta | delta | epsilon | gamma |
|-----------|--------|------------|--------------|--------------|------------------|----------|----------|-------------|-------|--------|----------------|-----|------|--------------|-------|------|-------|---------|-------|
| -2.326080 | 0.0    | 0.017825   | 0.178422     | -0.057884    | -0.139130        | 0.276361 | 0.031776 | 0.019353    | -0.061624 | -0.073768 | 0.460503      | -0.059257 | -0.209634 | -0.054195 | 0.009297 | 3.593210 | -0.952516 | -1.373505 | -1.338615 | -2.272582 |
| -3.328219 | 0.0    | 0.144096   | -0.123597    | 0.310103     | -0.202903        | -0.686798 | -0.132983 | -0.532335   | -0.518969 | 0.033733  | 0.183406      | 0.124217  | 0.134725  | 0.426631   | 0.081367 | -0.664133 | 2.001571 | -3.966839 | -0.683531 | -0.826687 |
| -2.013668 | 0.0    | -0.153045  | -0.051944    | 0.415223     | -0.049618        | -0.421776 | 0.034577 | -0.171955   | -0.260469 | 0.045557  | -0.098376     | 0.012364  | -0.014058 | -0.135649 | -0.326588 | -1.745097 | -1.395176 | 5.213645  | -0.890202 | -1.494953 |
| -4.154240 | 0.0    | -0.163426  | -0.238793    | -0.411020    | -0.176140        | 0.387888 | 0.090226 | 0.425217    | 0.182506  | 0.006468  | -0.167992     | 0.102740  | 0.126372  | -0.153452 | 0.398615 | -4.089585 | -3.550788 | -3.313403 | 7.836781 | -2.507520 |
| -2.070468 | 0.0    | 0.226194   | 0.216133     | -0.073415    | 0.556250         | -0.056953 | -0.112513 | 0.485042    | 0.111461  | 0.078832  | -0.195338     | -0.107733 | -0.086649 | 0.106091  | 0.029783 | -1.585514 | -1.715078 | -0.038873 | -0.492014 | 4.935589  |

### Evaluation of the performance

Lastly, we need to evaluate the performance of the model on both train and test dataset

*Training Data Classification Report*
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.85      | 0.79   | 0.82     | 476     |
| 1     | 0.85      | 0.86   | 0.85     | 483     |
| 2     | 0.87      | 0.86   | 0.86     | 478     |
| 3     | 0.89      | 0.93   | 0.91     | 507     |
| 4     | 0.91      | 0.92   | 0.91     | 498     |
| **Accuracy** |  |  | **0.87** | 2442 |
| **Macro avg** | 0.87 | 0.87 | 0.87 | 2442 |
| **Weighted avg** | 0.87 | 0.87 | 0.87 | 2442 |

The model performs well with an overall accuracy of **87%**. Here are the key metrics:

- **Precision**: 0.85-0.91 across classes.
- **Recall**: 0.79_0.93, with class 3 having the highest recall (0.93).
- **F1-Score**: 0.82-0.91, showing a good balance.
- **Macro & Weighted Averages**: Both around 0.87, indicating consistent performance.

*Test Data Classification Report*
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.85      | 0.78   | 0.82     | 224     |
| 1     | 0.89      | 0.86   | 0.87     | 217     |
| 2     | 0.87      | 0.86   | 0.86     | 218     |
| 3     | 0.83      | 0.95   | 0.89     | 188     |
| 4     | 0.90      | 0.89   | 0.89     | 200     |
| **Accuracy** |  |  | **0.87** | 1047 |
| **Macro avg** | 0.87 | 0.87 | 0.87 | 1047 |
| **Weighted avg** | 0.87 | 0.87 | 0.87 | 1047 |

The model performs similarly well on the test set with an accuracy of **87%**. Key metrics are as follows:

- **Precision**: 0.83-0.90 across classes.
- **Recall**: 0.78-0.95, with class 3 having the highest recall (0.95).
- **F1-Score**: 0.82-0.89, indicating good balance.
- **Macro & Weighted Averages**: Both around 0.87, showing consistent performance across all classes.

Overall, the model did generalize quite well and is effective at **recovering** the missing user

<h1 align="center">Model for Top-Year</h1>

Similar with user we decide to resolve the Top_Year prediction by treating it like a **multi-classification** problem.  
There is 7 class to predict, and we would like to mix some **songs features** with **past album popularity** to get accurate prediction.  
The idea is that a specific album was top of the year then  most likely the songs that belong to the album have more chance to be top of this year.

### Feature Engineering
for the feature addition, we implemented:  
The number of times a song has appeared in a given year, based on its album's appearance in the top year.  
The idea is that if an album was popular in a specific year, the tracks from it are likely to be liked as well.  
This feature capture the **trends in how an album** (and its song) is popular over the year, data that was lost due to data deletion.  
It is calcuated by counting the top year of the album for every year. An album may appear multiples times if 2 songs are listened from the same album or if two differents users listen to a track from this album. Since there is no duplicate, there is no issue counting twice.

### k-NN (k-Nearest Neighbors)

First model tested with the new feature addition was k-NN.  
Since it is a distance-based algorith, the data should be scaled (ideally, Min-Max to keep the shape unchanged)  
We use hyperparameter tuning with GridSearch and achieve a 66% accuracy on the test dataset which is correct but we would like to explore a better model

*Test Data Classification Report:*
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.75      | 0.83   | 0.79     | 163     |
| 1     | 0.71      | 0.72   | 0.72     | 150     |
| 2     | 0.66      | 0.70   | 0.68     | 148     |
| 3     | 0.65      | 0.66   | 0.66     | 153     |
| 4     | 0.52      | 0.55   | 0.53     | 130     |
| 5     | 0.57      | 0.49   | 0.53     | 152     |
| 6     | 0.71      | 0.62   | 0.66     | 151     |
| **Accuracy**     | **0.66**  |        |          | **1047** |
| **Macro avg**    | **0.65**  | **0.65**| **0.65** | **1047** |
| **Weighted avg** | **0.66**  | **0.66**| **0.66** | **1047** |

### Decision Tree Classifier

Decision Tree have the particularity to be build to overfit. So the goal here is to prune the tree before it overfit to the training data set.  
We use again hyperparameter tuning with GridSearch.

![Alt text](https://github.com/RobertChanData/spotify_project/blob/main/Screenshot/Spotify_4.PNG?raw=true)

### Evaluation of the performance

Training Data Classification Report:*
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.86      | 0.83   | 0.85     | 337     |
| 1     | 0.68      | 0.94   | 0.79     | 348     |
| 2     | 0.80      | 0.73   | 0.77     | 347     |
| 3     | 0.87      | 0.80   | 0.84     | 343     |
| 4     | 0.73      | 0.81   | 0.77     | 370     |
| 5     | 0.75      | 0.72   | 0.73     | 348     |
| 6     | 0.99      | 0.72   | 0.83     | 349     |
| **Accuracy**     | **0.79**  |        |          | **2442** |
| **Macro avg**    | **0.81**  | **0.79**| **0.79** | **2442** |
| **Weighted avg** | **0.81**  | **0.79**| **0.79** | **2442** |

**Training Data Classification Report:**
- **Overall Accuracy**: 79%
- **Macro Average**: Precision: 0.81, Recall: 0.79, F1-Score: 0.79
- **Weighted Average**: Precision: 0.81, Recall: 0.79, F1-Score: 0.79


*Test Data Classification Report:*
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.84      | 0.78   | 0.81     | 163     |
| 1     | 0.64      | 0.91   | 0.75     | 150     |
| 2     | 0.78      | 0.76   | 0.77     | 148     |
| 3     | 0.87      | 0.79   | 0.83     | 153     |
| 4     | 0.63      | 0.78   | 0.70     | 130     |
| 5     | 0.71      | 0.62   | 0.66     | 152     |
| 6     | 0.97      | 0.69   | 0.81     | 151     |
| **Accuracy**     | **0.76**  |        |          | **1047** |
| **Macro avg**    | **0.78**  | **0.76**| **0.76** | **1047** |
| **Weighted avg** | **0.78**  | **0.76**| **0.76** | **1047** |

**Test Data Classification Report:**
- **Overall Accuracy**: 76%
- **Macro Average**: Precision: 0.78, Recall: 0.76, F1-Score: 0.76
- **Weighted Average**: Precision: 0.78, Recall: 0.76, F1-Score: 0.76

**Key Insights:**
- **Class 6** shows excellent precision (0.97) but a slightly lower recall (0.69).
- **Class 1** has very high recall (0.91) but lower precision (0.64), suggesting a class imbalance.

**Conclusion**: The model performs reasonably well, but recall and precision are not consistently high across all classes.  
However the model performance is acceptable so we will use it to predict the `top_year` of a song.


<h1 align="center">Reconstruction of the playlist</h1>

"The goal of this project was to **impute the missing `user` and `top_year` values** using Machine Learning techniques.  

While the exploratory data analysis (EDA) and modeling were important steps, the **real business value lies in the following code**, which apply the most efficient ML model founded to give prediction.

### Make prediction

Based on the model obtained, we can now confidently use the appropriate ML model to reconstruct the playlist: 
- Predict unknown users using Logistic Regression (OvR)
- Predict unknown top_year using Decision Tree

### Reshape the initial CSV

With some data cleaning and stucturing, we retrieve the initial dataset and attach the predicted value to its corresponding index

### Output the result + Double check

We create a short snippet of code that create a csv for every user and top_year and double-check the coherence of the result (correct number of csv file = 35 (7 different years x 5 different users) while checking that all top_year and user have been filed.

<h1 align="center">Conclusion</h1>

This project demonstrates an approach to reconstructing playlists based on various song characteristics and user preferences. While the model performs reasonably well, there is still potential for improvement by refining the features or tuning the model further.
