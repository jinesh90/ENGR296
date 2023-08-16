
"""
This dag is doing following steps

1) download data file from imdb.
2) unzip data into tsv file, load tsv to pandas dataframe. get type = movie, export imdb movie ids.
3) get data from tvdb by requesting id. insert data to mongodb.
#http://www.omdbapi.com/?i=tt3896198&apikey=5c317172
"""

import json
import pathlib
import pandas as pd
import requests



from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


s = requests.Session()


import requests
from datetime import datetime

import requests.exceptions as rex

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


retries = Retry(total=5,
                backoff_factor=0.1,
                status_forcelist=[500, 502, 503, 504, 400])

s.mount('http://', HTTPAdapter(max_retries=retries))

API_KEYS = "5c317172"


dag = DAG(
    dag_id='movie_data',
    start_date=datetime(2021, 12, 25),
    schedule_interval=None
)


get_movies_from_imdb = BashOperator(
    task_id="get_data_from_imdb",
    bash_command="curl -o /tmp/movies.gz -L 'https://datasets.imdbws.com/title.basics.tsv.gz'",
    dag=dag
)

wait_for_sometime = BashOperator(task_id="wait_for_some_time",
                                 bash_command="sleep 360",
                                 dag=dag)


extract_movies_from_tz = BashOperator(
    task_id="unzip_movie_tsv_file",
    #bash_command="pwd;ls -al; zcat /tmp/movies.gz > /tmp/data/movies.tsv; exit 99",
    bash_command="ls -al",
    dag=dag
)

def _get_movie_ids():
    df = pd.read_csv("/tmp/data/movies.tsv", sep='\t', low_memory=False)
    movie_df = df[df['titleType'] == 'movie']
    with open('/tmp/data/movieid.txt', 'w+') as fout:
        for mid in movie_df['tconst']:
            fout.write(mid + '\n')
    fout.close()

get_movie_ids = PythonOperator(
    task_id="get_movie_id",
    python_callable=_get_movie_ids,
    dag=dag
)

def _get_movie_data_from_imdb_ids():
    fw = open("/tmp/data/movies_data.txt", "w")
    with open("/tmp/data/movieid.txt", "r") as fr:
        try:
            for mid in fr.readlines():
                movie_id_ = mid.rstrip()
                movie_url = "http://www.omdbapi.com/?i={}&apikey=5c317172".format(movie_id_)
                m_res = s.get(movie_url, verify=False)
                if m_res.status_code == 200:
                    mdata = json.loads(m_res.content)
                    if "Error" in mdata.keys():
                        print("Movie data is not found for this id: {}".format(mid))
                    else:
                        print(mdata)
                        fw.write(str(mdata)+'\n')
                else:
                    print("Status code is not 200. status code is :{}".format(m_res.status_code))
        except Exception as e:
            print("Exception: {}".format(e))
            pass

        #mydb = mongo_client["movie"]
        #mycol = mydb["movie"]


insert_movie_data = PythonOperator(
    task_id="insert_movie_data",
    python_callable=_get_movie_data_from_imdb_ids,
    dag=dag
)


get_movies_from_imdb >> extract_movies_from_tz >> get_movie_ids >> insert_movie_data
