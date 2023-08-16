#!/usr/bin/env bash
curl -o movies.gz -L 'https://datasets.imdbws.com/title.basics.tsv.gz'
gunzip movies.gz > movies.tsv