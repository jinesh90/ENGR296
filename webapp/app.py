from flask import Flask, render_template, request
import random
import pandas as pd
import logging
import ast


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('recommendation-app')

app = Flask(__name__)

df = pd.read_csv("recsys.csv")


def convert_str_to_list(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []

df['similar_movies'] = df['similar_movies'].apply(convert_str_to_list)

@app.route('/')
def home ():
    # Randomly select 100 movies from the DataFrame
    state = random.randint(0,100)
    random_movies = df.sample(n=100, random_state=state)  # Adjust the random_state as needed
    movies = random_movies.values.tolist()
    return render_template('home.html', movies=movies)


@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        search_query = request.form.get('search_query')

        # Filter the DataFrame based on the search query
        filtered_movies = df[df['Title'].str.contains(search_query, case=False)]

        # Convert the filtered DataFrame to a list of dictionaries
        search_results = filtered_movies.to_dict('records')

        return render_template('search.html', search_results=search_results)
    return render_template('search.html', search_results=None)


@app.route('/show_similar/<imdb_id>')
def show_similar(imdb_id):
    selected_movie = df[df['imdbID'].isin([imdb_id])]
    if not selected_movie.empty:
        selected_movie = selected_movie.iloc[0]
        similar_movie_ids = selected_movie['similar_movies']
        original_movie_title = selected_movie['Title']
        if similar_movie_ids:
            logging.info("Similar movie ids for :{} is :{}".format(selected_movie, similar_movie_ids))

            similar_movies = df[df['imdbID'].isin(similar_movie_ids)]


            similar_movies = similar_movies.to_dict('records')
            logging.info("Records: {}".format(similar_movies))
        else:
            similar_movies = []

        return render_template('similar_movies.html', movie=selected_movie, similar_movies=similar_movies, original_movie_title=original_movie_title)
    else:
        return "Movie not found"



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')