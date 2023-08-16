import pandas as pd


def filter_movie(file):
    """
    filter movie from latest imdb drop
    :param file:
    :return:
    """
    df = pd.read_csv(file, sep='\t', low_memory=False)
    movie_df = df[df['titleType'] == 'movie'].reset_index()
    with open('movieid.txt', 'w+') as fout:
        for mid in movie_df['tconst']:
            fout.write(mid + '\n')
    fout.close()


filter_movie('movies.tsv')
