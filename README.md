# Music-recommendation-system
The Music Recommendation System is a mini project in the field of data science implemented using Python. The aim of this project is to develop an intelligent system that can analyze user preferences and recommend music tracks based on their listening history and preferences
import pandas as pd
tracks = pd.read_csv('/content/sample_data/music.csv.zip')
tracks.head(5)

tracks.shape

import matplotlib.pyplot as plt
tracks.dropna(inplace = True)
tracks.isnull().sum().plot.bar()
plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data = {
    'music': np.random.randint(1, 101, 50),
    'len': np.random.randint(1, 6, 50),
    'release_year': np.random.randint(2000, 2023, 50)
}
df = pd.DataFrame(data)
plt.scatter(df['music'], df['len'], c=df['release_year'], cmap='viridis', alpha=0.7)
plt.xlabel('music')
plt.ylabel('len')
plt.title('music vs len')
plt.colorbar(label='Release Year')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

import matplotlib.pyplot as plt
topic= ["sadness", "romantic", "obscene", "night/time"]
len= [95, 51, 24, 54]
plt.bar(topic,len, color='Blue')
plt.xlabel('topic')
plt.ylabel('len')
plt.title('topic and len  Histogram')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
ratings = np.random.randint(1, 6, 100)
plt.hist(ratings, bins=np.arange(1, 7) - 0.5, edgecolor='black', alpha=0.7)
plt.xlabel('music')
plt.ylabel('len')
plt.title('Distribution of music for len')
plt.xticks(np.arange(1, 6))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
popularity = np.random.randint(1, 101, 50)
user_ratings = np.random.randint(1, 6, 50)
plt.scatter(popularity, user_ratings, color='blue', alpha=0.7)
plt.xlabel('music')
plt.ylabel('len')
plt.title('music vs len')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data = {
    'lyrics': np.random.randint(1, 101, 50),
    'artist_name': np.random.randint(1, 6, 50),
    'release_date': np.random.randint(2000, 2023, 50)
}
df = pd.DataFrame(data)
plt.scatter(df['lyrics'], df['artist_name'], c=df['release_date'], cmap='viridis', alpha=0.7)
plt.xlabel('lyrics')
plt.ylabel('artist_name')
plt.title('Scatter Plot of Popularity vs. User Ratings for Songs')
plt.colorbar(label='Release Year')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

import matplotlib.pyplot as plt
import pandas as pd
data = {
    'len': [95, 51, 24, 54, 48],
    'age': [1, 1, 1, 1, 1],
    'release_date': [1951, 1952, 1953, 1954, 1955]
}
df = pd.DataFrame(data)
for col in ['len', 'age']:
    if df[col].dtype == 'float':
        plt.scatter(df['release_date'], df[col], label=col)
    else:
        plt.scatter(df['release_date'], df[col], label=col, s=50)

plt.xlabel('Release Year')
plt.ylabel('Release Year')
plt.title(' Release Yearand User Release Year')
plt.legend()
plt.show()

def process_lyrics(lyrics_text):
    processed_lyrics = lyrics_text.upper()
    return processed_lyrics
lyrics_text = 'care moment hold fast press lips dream heaven speak share glow grow pass meet break speak share glow grow pass meet'
processed_lyrics = process_lyrics(lyrics_text)
print(processed_lyrics)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the music dataset from a zip file
music_data = pd.read_csv('/content/sample_data/music.csv.zip', compression='zip')

# Assuming 'danceability' and 'energy' are numerical features in your dataset
x_feature = 'len'
y_feature = 'age'

# Create a contour plot
plt.figure(figsize=(7, 5))
sns.kdeplot(data=music_data, x=x_feature, y=y_feature, fill=True, cmap="viridis")

# Set labels for the axes
plt.xlabel('len')
plt.ylabel('age')

# Set the title of the plot
plt.title('Contour Plot of len vs. age')

# Show the plot
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
music_data = pd.read_csv('/content/music.csv (1).zip')
X = music_data[['danceability']]
y = music_data['len']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
plt.figure(figsize=(5, 3))
plt.scatter(X_test, y_test, c='blue', label='Actual')
plt.plot(X_test, y_pred, c='red', label='Linear Regression')
plt.xlabel('danceability')
plt.ylabel('len')
plt.title('Linear Regression: danceability vs. len')
plt.legend()
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
music_data = pd.read_csv('/content/sample_data/music.csv.zip')
x_feature = 'danceability'
y_feature = 'energy'
plt.figure(figsize=(5, 3))
sns.kdeplot(data=music_data, x=x_feature, y=y_feature, fill=True, cmap="viridis")
sns.kdeplot(data=music_data, x=x_feature, y=y_feature, levels=10, cmap="viridis", linewidths=1)
plt.xlabel('Danceability')
plt.ylabel('Energy')
plt.title('Joint Density and Contour Plot: Danceability vs. Energy')
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
music_data = pd.read_csv('/content/sample_data/music.csv.zip')
feature_to_plot = 'danceability'
plt.figure(figsize=(8, 6))
sns.histplot(music_data[feature_to_plot], kde=True, color='blue', bins=20)
plt.xlabel(feature_to_plot)
plt.ylabel('Frequency')
plt.title(f'Normal Distribution Curve for {feature_to_plot}')
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
music_data = pd.read_csv('/content/sample_data/music.csv.zip')
x_feature = 'len'
y_feature = 'dating'
plt.figure(figsize=(5, 3))
sns.scatterplot(data=music_data, x=x_feature, y=y_feature, color='blue', alpha=0.7)
plt.xlabel('len')
plt.ylabel('dating')
plt.title('Scatter Plot: len vs. dating')
plt.show()
correlation_coefficient = music_data[x_feature].corr(music_data[y_feature])
print(f'Correlation Coefficient between {x_feature} and {y_feature}: {correlation_coefficient}')


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
a = pd.read_csv('/content/music.csv (1).zip')
numeric_columns = a.select_dtypes(include=[np.number])
model = TSNE(n_components=2, random_state=0)
tsne_data = model.fit_transform(numeric_columns.head(500))
plt.figure(figsize=(7, 7))
plt.scatter(tsne_data[:, 0], tsne_data[:, 1])
plt.title('t-SNE Visualization')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
file_path = '/content/music.csv (1).zip'
data = pd.read_csv(file_path)
numeric_columns = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_columns.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()
