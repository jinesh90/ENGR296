import requests
import multiprocessing
import json
import itertools
import pymongo
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

s = requests.Session()


myclient = pymongo.MongoClient("mongodb://localhost:27017/")
db = myclient.imdb
collection = db.movie

def insert_to_database(movie_id):
    movie_url = "http://www.omdbapi.com/?i={}&apikey=5c317172".format(movie_id.rstrip())
    m_res = s.get(movie_url, verify=False)
    if m_res.status_code == 200:
        try:
            mdata = json.loads(m_res.content)
            if "Error" in mdata.keys():
                print("Movie data is not found for this id: {}".format(movie_id))
            else:
                pass
                #print(mdata)
                collection.insert_one(mdata)
        except:
            pass
    else:
        print("Status code is not 200. status code is :{}".format(m_res.status_code))
    with counter_lock:
        counter_value = next(line_counter)
        if counter_value % 100 == 0:
            print(f"{counter_value} lines processed.")

with open("restid3.txt", "r") as fr:
    lines = fr.readlines()
    line_counter = itertools.count(start=1)
    counter_lock = multiprocessing.Lock()
    pool = multiprocessing.Pool()
    pool.map(insert_to_database, lines)
    pool.close()
    pool.join()



