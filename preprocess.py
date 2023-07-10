import os
import glob
import json
import numpy as np
import pandas as pd

pattern = "./movie_dataset_public_final/scores/tagdl.csv"

#Read data and create list of dfs of the following format: [['tag', 'item_id', 'score'],['tag', 'item_id', 'score']]
with open(pattern) as f:
    tagdl = pd.read_csv(f, delimiter = ",", header = 0)
    tagdl_grouped_by_tag = [d for _, d in tagdl.groupby(['tag'])]

#Iterate through list of tags to create list of dfs with the following format: [['item_id', 'action'],['item_id', 'adventure'], etc]
for tag in tagdl_grouped_by_tag:
  tagValue = str(tag['tag'].iloc[0])
  tag.drop(columns=tag.columns[0], axis=1, inplace=True)
  tag.columns = ['item_id', tagValue]

#Merge all the dfs from the prior step to create single df with the following format: ['item_id','action','adventure', etc]
MovieLens2021MoviesAllTags = tagdl_grouped_by_tag[0]
for tag in tagdl_grouped_by_tag[1:]:
  MovieLens2021MoviesAllTags = pd.merge(MovieLens2021MoviesAllTags, tag, on='item_id', how='outer')
#MovieLens2021MoviesAllTags.to_csv('MovieLens2021MovieIDAllTags.csv', sep='\t')


patterns = ["./movie_dataset_public_final/raw/metadata_updated.json"] #,"./movie_dataset_public_final/raw/ratings.json","./movie_dataset_public_final/raw/tags.json","./movie_dataset_public_final/raw/tag_count.json"]
#metadata, ratings, tags, tag_count = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
#data = [metadata, ratings, tags, tag_count]
for i in range(0,len(patterns)):
  pattern = patterns[i]
  with open(pattern) as f:
    currFile = [json.loads(line) for line in f]
    MovieLens2021Metadata = pd.DataFrame.from_dict(currFile, orient='columns')

#Merge metadata file with all tags
MovieLens2021Metadata = pd.merge(MovieLens2021Metadata, MovieLens2021MoviesAllTags, on='item_id', how='outer')
MovieLens2021Metadata.to_csv('MovieLens2021MovieMetaDataAllTags.csv', sep='\t')

pattern = "./ml-25m/movies.csv"
#Read ML-25M movies.csv file which is of the format: ['movieID','title','genres'] Drop genres column
with open(pattern, encoding='utf8') as f:
    ML25M = pd.read_csv(f, delimiter = ",", header = 0)
    ML25M.drop(columns=ML25M.columns[2], axis=1, inplace=True)
    ML25M.columns = ['item_id','title']
ML25Mtitles = ML25M['title'].to_numpy()
MovieLens2021titles = MovieLens2021Metadata['title'].to_numpy()
print("Movies in ML-25M not in MovieLens2021 Dataset: " + str(set(ML25Mtitles) - set(MovieLens2021titles)))
print("Total number of movies in ML-25M dataset: " + str(len(ML25Mtitles)))
print("Total number of movies in MovieLens2021 dataset: " + str(len(MovieLens2021titles)))

#Merge ML-25M movieIDs list with MovieLens2021 metadata and tags df
mergedData = pd.merge(ML25M, MovieLens2021Metadata, on=['title','item_id'], how='inner')
mergedData.to_csv('MergedMovieIDMetaDataAllTags.csv', sep='\t')
