import nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

VERB_CODES = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
lemmatizer = WordNetLemmatizer()


# this is generated from previous stage called datapipeline
df = pd.read_csv('movies.csv')


# for getting top 10 movie producing countries
top_countries = df['Country'].value_counts().head(10)
def get_top_n_countries_polt(df,n=10):
  plt.figure(figsize=(12, 6))  # Set the figure size

  # Create the count plot using seaborn
  sns.set(style="whitegrid")
  plot = sns.barplot(x=top_countries.index, y=top_countries.values, palette="viridis")

  # Set plot titles and labels
  plt.title("Top 10 Movie Countries")
  plt.xlabel("Countries")
  plt.ylabel("Number of Movies")

  # Customize the x-axis labels
  plot.set_xticklabels(plot.get_xticklabels(), rotation=45, ha="right")

  # Show the plot
  plt.tight_layout()
  plt.show()


# filter only USA movies
df_usa = df[df.Country.str.contains('United States') | df.Country.str.contains('USA')]
df_usa.reset_index(drop=True, inplace=True)

# convert USA year to check movie production data
df_usa["Year"] = pd.to_numeric(df_usa["Year"], errors="coerce")

# Filter out rows with NaN values in the "Year" column
df_usa = df_usa[~df_usa["Year"].isnull()]

# Convert the "Year" column to integer type
df_usa["Year"] = df_usa["Year"].astype(int)



df_usa['decade'] = (df_usa['Year'] // 10) * 10
decade_order = sorted(df_usa['decade'].unique())

def get_movie_plot_by_decade(movie_df):
  plt.figure(figsize=(10, 6))  # Set the figure size

  # Create the count plot using seaborn
  sns.set(style="whitegrid")  # Set the style to whitegrid
  plot = sns.countplot(data=movie_df, x='decade', palette="Set2")

  # Set plot titles and labels
  plt.title("Number of Movies Released per Decade in USA")
  plt.xlabel("Decade")
  plt.ylabel("Number of Movies")

  # Customize the x-axis labels
  xtick_labels = [f"{decade}-{decade+9}" for decade in movie_df['decade'].unique()]
  plot.set_xticklabels(xtick_labels, rotation=45)

  # Show the plot
  plt.show()


# filter out movies after 1990
df_usa_latest = df_usa[df_usa["Year"] >= 1990].reset_index(drop=True)
df_usa_latest.drop(['Unnamed: 0', '_id'], axis=1, inplace=True)

# check top generes
top_genres = df_usa_latest['Genre'].value_counts().head(10)

def get_top_n_genere_polt(df,n=10):
  plt.figure(figsize=(12, 6))  # Set the figure size

  # Create the count plot using seaborn
  sns.set(style="whitegrid")
  plot = sns.barplot(x=top_genres.index, y=top_genres.values, palette="viridis")

  # Set plot titles and labels
  plt.title("Top 10 Movie Genres")
  plt.xlabel("Genre")
  plt.ylabel("Number of Movies")

  # Customize the x-axis labels
  plot.set_xticklabels(plot.get_xticklabels(), rotation=45, ha="right")

  # Show the plot
  plt.tight_layout()
  plt.show()

# check max plot length
df_usa_latest['plot_length'] = df_usa_latest['Plot'].str.len()
max_plot_length = df_usa_latest['plot_length'].max()
min_plot_length = df_usa_latest['plot_length'].min()
print("Maximum plot length:", max_plot_length)
print("Minimum plot length:", min_plot_length)




# remove data with n/a value
df_usa_latest = df_usa_latest.dropna(subset=['Plot','Genre','Actors','imdbRating','imdbVotes','Director'])
df_usa_latest.reset_index(drop=True, inplace=True)

# correcting imdb rating
df_usa_latest["imdbRating"] = pd.to_numeric(df_usa_latest["imdbRating"], errors="coerce")

# Filter out rows with NaN values in the "Year" column
df_usa_latest = df_usa_latest[~df_usa_latest["imdbRating"].isnull()]

# Convert the "Year" column to integer type
df_usa_latest["imdbRating"] = df_usa_latest["imdbRating"].astype(int)


# check popular rating only
df_usa_popular = df_usa_latest[df_usa_latest["imdbRating"] >= 6].reset_index(drop=True)




# process plot sentences.
def preprocess_sentences(text):
  text = text.lower()
  temp_sent =[]
  words = nltk.word_tokenize(text)
  tags = nltk.pos_tag(words)
  for i, word in enumerate(words):
      if tags[i][1] in VERB_CODES:
          lemmatized = lemmatizer.lemmatize(word, 'v')
      else:
          lemmatized = lemmatizer.lemmatize(word)
      if lemmatized not in stop_words and lemmatized.isalpha():
          temp_sent.append(lemmatized)
  plot = ' '.join(temp_sent)
  plot = plot.replace("n't", " not")
  plot = plot.replace("'m", " am")
  plot = plot.replace("'s", " is")
  plot = plot.replace("'re", " are")
  plot = plot.replace("'ll", " will")
  plot = plot.replace("'ve", " have")
  plot = plot.replace("'d", " would")
  return plot

# preprocess plot
df_usa_popular["Plot"]= df_usa_popular["Plot"].apply(preprocess_sentences)

# define plot vector
df_usa_popular['plot_vector'] =   df_usa_popular['Genre'] + " " + df_usa_popular['Genre'] +  " " + df_usa_popular['Actors'] + " " + df_usa_popular['Director'] + " " + df_usa_popular['Plot']

df_usa_popular.dropna(subset=['plot_vector'], inplace=True)

#reduced plot
def reduced_text(text):
    words = text.lower().split()[:300]  # trime 256 words
    return ' '.join(words)

df_usa_popular["reduced_vector"] = df_usa_popular["plot_vector"].apply(reduced_text)

# convert plot to vector
tfidf = TfidfVectorizer()
movieid= tfidf.fit_transform((df_usa_popular["reduced_vector"].tolist()))

similarity = cosine_similarity(movieid,movieid)

# find top n
n = 5
top_n_similar_indices = similarity.argsort(axis=1)[:, -n-1:-1]
similar_ids = [df_usa_popular.iloc[i]['imdbID'] for i in top_n_similar_indices]


similar_ids_imdb = [[x.iloc[0],x.iloc[1],x.iloc[2],x.iloc[3],x.iloc[4]] for x in similar_ids]

df_usa_popular['similar_movies'] = similar_ids_imdb

selected_columns = ['Title','Poster','imdbID','similar_movies']



df_usa_popular.to_csv('plot_recsys.csv', index=False)