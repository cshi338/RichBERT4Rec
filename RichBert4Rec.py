import random
import os
import pandas as pd
from pandas.core.arrays import boolean
import pytorch_lightning as pl
import torch
import copy
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import Recommender
from data_processing import get_context, pad_list, map_column, MASK, PAD, SEP
from multiprocessing import Process, Manager
from collections import Counter
import gzip

def map_directors(directors, total):
    directors = ','.join(directors)
    directors = directors.replace(', ',',')
    directors = list(set(directors.split(",")))
    directors.sort()
    for x in range(len(directors)):
      directors[x] = directors[x].lstrip()
    directors = list(set(directors))
    return dict(zip(directors, range(total, len(directors) + total + 1)))

def map_actors(actors, total):
    actors = ','.join(actors)
    actors = actors.replace(', ',',')
    actors = list(set(actors.split(",")))
    actors.sort()
    for x in range(len(actors)):
      actors[x] = actors[x].lstrip()
    actors = list(set(actors))
    return dict(zip(actors, range(total, len(actors) + total + 1)))

def map_tags(tags, total):
    tags.sort()
    return dict(zip(tags, range(total, len(tags) + total + 1)))

def mask_list(l1, p=0.8):
    l1 = [a if random.random() < p else MASK for a in l1]
    return l1

def mask_last_elements_list(l1, val_context_size: int = 1):
    #everything except last 5 indices + masked last 5 indices
    l1 = l1[:-val_context_size] + mask_list(l1[-val_context_size:], p=0.0)
    return l1

def augment_data(numDir, numCast, numTags, inverse_mapping, directorsMap, actorsMap, tagsMap, augmented):
    mappedData = {}
    for selectedID in tqdm(inverse_mapping):
        entry = [selectedID]
        row = augmented.loc[augmented['item_id'] == inverse_mapping.get(selectedID)]
        if numDir > 0:
            directors = row.iloc[0]['directedBy']
            if ',' in directors:
              if ', ' in directors:
                directors = directors.split(', ')
              else:
                directors = directors.split(',')
              for x in range(len(directors)):
                directors[x] = directors[x].lstrip()
            else:
              directors = directors.lstrip()

            if isinstance(directors, str):
              entry.append(directorsMap.get(directors, PAD))
            else:
              padded = numDir - len(directors)
              if padded > 0:
                for director in directors:
                  entry.append(directorsMap.get(director, PAD))
                for pad in range(padded):
                  entry.append(0) #append PAD
              else:
                for i in range(numDir):
                  entry.append(directorsMap.get(directors[i], PAD))
        if numCast > 0:
            actors = ((row.iloc[0]['starring']).replace(', ',',')).split(",")
            for x in range(len(actors)):
              actors[x] = actors[x].lstrip()

            padded = numCast - len(actors)
            if padded > 0:
              for actor in actors:
                entry.append(actorsMap.get(actor, PAD))
              for pad in range(padded):
                entry.append(0) #append PAD
            else:
              for i in range(numCast):
                entry.append(actorsMap.get(actors[i], PAD))
        if numTags > 0:
            tags = row.iloc[:, 8:]
            tags = tags.T
            if tags.iloc[0,0] != 0.0:
              tags[list(tags.columns)[0]] = tags[list(tags.columns)[0]].astype(float)
              tags = tags.nlargest(numTags, list(tags.columns)[0])
              tags = list(tags.index)
              for tag in tags:
                entry.append(tagsMap.get(tag, PAD))
            else:
              for i in range(numTags):
                entry.append(0) #append PAD
        mappedData[selectedID] = entry
    return mappedData

def augmented_mask_list(l1, p, mappedData):
    selectedIDs = l1
    src_items = []
    trg_items = []
    for selectedID in selectedIDs:
      if random.random() < p:
        entry = mappedData.get(selectedID)
        src_items = src_items + entry
        trg_items.append(selectedID)
      else:
        trg_items.append(selectedID)
        for i in range(len(mappedData.get(selectedID))):
            src_items.append(1)
    return src_items, trg_items

def augmented_mask_last_list(l1, mappedData, val_context_size: int = 1):
    selectedIDs = l1
    src_items = []
    trg_items = []
    for i in range(len(selectedIDs)):
      selectedID = selectedIDs[i]
      if i < len(selectedIDs) - val_context_size:
        entry = mappedData.get(selectedID)
        src_items = src_items + entry
        trg_items.append(selectedID)
      else:
        trg_items.append(selectedID)
        for i in range(len(mappedData.get(selectedID))):
            src_items.append(1)
    return src_items, trg_items

def createData(groups, grp_by, split, p, augment, entry_size, mappedData, numAugments, popular_items):
    all_src_items = []
    y_true = []
    neg_samples = []
    print("Creating " + split + " data:")
    for i in tqdm(range(len(groups))):
        group = groups[i]
        neg_sample = generate_negative_samples(group, grp_by, popular_items)

        df = grp_by.get_group(group)
        if split == "train":
            num_dupes = 10
            for _ in range(num_dupes):
                context = get_context(df, split=split, context_size=entry_size)
                selectedIDs = context["item_id_mapped"].tolist()
                if augment:
                  src_items, trg_items = augmented_mask_list(selectedIDs, p, mappedData)
                  pad_mode = "left" if random.random() < 0.5 else "right"
                  temp = []
                  for j in range(numAugments):
                      src_item = src_items[j::numAugments]
                      src_item = pad_list(src_item, history_size=entry_size, mode=pad_mode)
                      temp = temp + src_item
                  src_items = temp
                else:
                  trg_items = selectedIDs
                  src_items = mask_list(trg_items)
                  pad_mode = "left" if random.random() < 0.5 else "right"
                  src_items = pad_list(src_items, history_size=entry_size, mode=pad_mode)
                trg_items = pad_list(trg_items, history_size=entry_size, mode=pad_mode)
                all_src_items.append(np.array(src_items))
                y_true.append(np.array(trg_items))
                neg_samples.append(neg_sample)
        else:
            context = get_context(df, split=split, context_size=entry_size)
            selectedIDs = context["item_id_mapped"].tolist()
            if augment:
                src_items, trg_items = augmented_mask_last_list(selectedIDs, mappedData)
                pad_mode = "left" if random.random() < 0.5 else "right"
                temp = []
                for j in range(numAugments):
                    src_item = src_items[j::numAugments]
                    src_item = pad_list(src_item, history_size=entry_size, mode=pad_mode)
                    temp = temp + src_item
                src_items = temp
            else:
                trg_items = selectedIDs
                src_items = mask_last_elements_list(trg_items)
                pad_mode = "left" if random.random() < 0.5 else "right"
                src_items = pad_list(src_items, history_size=entry_size, mode=pad_mode)
            trg_items = pad_list(trg_items, history_size=entry_size, mode=pad_mode)
            all_src_items.append(np.array(src_items))
            y_true.append(np.array(trg_items))
            neg_samples.append(neg_sample)
    return np.array(all_src_items), np.array(y_true), np.array(neg_samples)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.split = split
        if self.split == "train":
            with gzip.GzipFile('srctraining.npy.gz', "rb") as f:
                self.src = np.load(f, allow_pickle=True)
            with gzip.GzipFile('ytruetraining.npy.gz', "rb") as f:
                self.ytrue = np.load(f, allow_pickle=True)
        elif self.split == "val":
            with gzip.GzipFile('srcvalidation.npy.gz', "rb") as f:
                self.src = np.load(f, allow_pickle=True)
            with gzip.GzipFile('ytruevalidation.npy.gz', "rb") as f:
                self.ytrue = np.load(f, allow_pickle=True)
        else:
            with gzip.GzipFile('srctest.npy.gz', "rb") as f:
                self.src = np.load(f, allow_pickle=True)
            with gzip.GzipFile('ytruetest.npy.gz', "rb") as f:
                self.ytrue = np.load(f, allow_pickle=True)
            with gzip.GzipFile('negsamtest.npy.gz', "rb") as f:
                self.neg_sam = np.load(f, allow_pickle=True)
    def __len__(self):
        return len(self.ytrue)
    def __getitem__(self, idx):
        src_items = self.src[idx]
        trg_items = self.ytrue[idx]
        src_items = torch.tensor(src_items, dtype=torch.long)
        trg_items = torch.tensor(trg_items, dtype=torch.long)
        if self.split == "test":
            neg_samples = self.neg_sam[idx]
            neg_samples = torch.tensor(neg_samples, dtype=torch.long)
            return src_items, trg_items, neg_samples
        return src_items, trg_items

def train(
    grp_by_train,
    trainGroups,
    movie_title_size,
    directors_size,
    actors_size,
    tags_size,
    entry_size,
    mappedData,
    numDir,
    numCast,
    numTags,
    augment: boolean = False,
    fromCheckpoint: boolean = False,
    checkpointPath: str = "",
    log_dir: str = "recommender_logs",
    model_dir: str = "recommender_models",
    batch_size: int = 256,
    epochs: int = 2000,
):

    #Get movies ranked by popularity
    popular_items = items_by_popularity(trainGroups, grp_by_train)

    srctrainingData, ytruetrainingData, negsamtrainingData = createData(
        groups=trainGroups,
        grp_by=grp_by_train,
        split="train",
        p = 0.8,
        augment = augment,
        entry_size = entry_size,
        mappedData = mappedData,
        numAugments = 1 + numDir + numCast + numTags,
        popular_items = popular_items,
    )
    srcvalidationData, ytruevalidationData, negsamvalidationData = createData(
        groups=trainGroups,
        grp_by=grp_by_train,
        split="val",
        p = 0.8,
        augment = augment,
        entry_size = entry_size,
        mappedData = mappedData,
        numAugments = 1 + numDir + numCast + numTags,
        popular_items = popular_items,
    )

    print("Compressing training data: ")
    f = gzip.GzipFile("srctraining.npy.gz", "w")
    np.save(file=f, arr=srctrainingData)
    f.close()
    f = gzip.GzipFile("ytruetraining.npy.gz", "w")
    np.save(file=f, arr=ytruetrainingData)
    f.close()
    f = gzip.GzipFile("negsamtraining.npy.gz", "w")
    np.save(file=f, arr=negsamtrainingData)
    f.close()
    print("Compressing validation data: ")
    f = gzip.GzipFile("srcvalidation.npy.gz", "w")
    np.save(file=f, arr=srcvalidationData)
    f.close()
    f = gzip.GzipFile("ytruevalidation.npy.gz", "w")
    np.save(file=f, arr=ytruevalidationData)
    f.close()
    f = gzip.GzipFile("negsamvalidation.npy.gz", "w")
    np.save(file=f, arr=negsamvalidationData)
    f.close()
    print("Data compressed!")


    train_data = Dataset(
        split="train",
    )

    val_data = Dataset(
        split="val",
    )

    print("len(train_data)", len(train_data))
    print("len(val_data)", len(val_data))

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
        pin_memory = True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
        pin_memory = True,
    )

    model = Recommender(
        numDir = numDir,
        numCast = numCast,
        numTags = numTags,
        movie_title_size = movie_title_size,
        directors_size = directors_size,
        actors_size = actors_size,
        tags_size = tags_size,
        lr=1e-3,
        dropout=0.1,
    )

    logger = TensorBoardLogger(
        save_dir=log_dir,
    )
    if augment:
      checkpoint_callback = ModelCheckpoint(
          monitor="valid_nDCG10",
          mode="max",
          dirpath=model_dir,
          filename="recommenderAugmented",
      )
    else:
      checkpoint_callback = ModelCheckpoint(
          monitor="valid_nDCG10",
          mode="max",
          dirpath=model_dir,
          filename="recommenderDefault",
      )
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator = 'gpu',
        devices = 1,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    if fromCheckpoint:
        # automatically restores model, epoch, step, LR schedulers, apex, etc...
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpointPath)
    else:
        trainer.fit(model, train_loader, val_loader)

    result_val = trainer.test(test_dataloaders=val_loader)

    output_json = {
        "val_loss": result_val[0]["test_loss"],
        "best_model_path": checkpoint_callback.best_model_path,
    }
    print(output_json)
    return output_json

def items_by_popularity(groups, grp_by):
    #Iterate through all users to rank movies by popularity
    popularity = Counter()
    print("Sorting movies by popularity: ")
    for group in tqdm(groups):
        popularity.update(grp_by.get_group(group)['item_id_mapped'])
    popular_items = sorted(popularity, key=popularity.get, reverse=True)
    return popular_items

def generate_negative_samples(group, grp_by, popular_items):
    seen = set(grp_by.get_group(group)['item_id_mapped'])
    #return random.sample(list(set(popular_items) - seen), 100)

    samples = []
    for item in popular_items:
        if len(samples) == 100:
            break
        if item in seen:
            continue
        samples.append(item)
    return samples


def test(
    grp_by_test,
    testGroups,
    movie_title_size,
    directors_size,
    actors_size,
    tags_size,
    entry_size,
    mappedData,
    batch_size,
    numDir,
    numCast,
    numTags,
    augment: boolean = False,
    checkpointPath: str = "",
    log_dir: str = "recommender_logs",
):

    #Get movies ranked by popularity
    popular_items = items_by_popularity(testGroups, grp_by_test)

    srctestData, ytruetestData, negsamtestData = createData(
        groups=testGroups,
        grp_by=grp_by_test,
        split="test",
        p = 0.8,
        augment = augment,
        entry_size = entry_size,
        mappedData = mappedData,
        numAugments = 1 + numDir + numCast + numTags,
        popular_items = popular_items,
    )

    print("Compressing test data: ")
    f = gzip.GzipFile("srctest.npy.gz", "w")
    np.save(file=f, arr=srctestData)
    f.close()
    f = gzip.GzipFile("ytruetest.npy.gz", "w")
    np.save(file=f, arr=ytruetestData)
    f.close()
    f = gzip.GzipFile("negsamtest.npy.gz", "w")
    np.save(file=f, arr=negsamtestData)
    f.close()
    print("Data compressed!")

    test_data = Dataset(
        split="test",
    )

    print("len(test_data)", len(test_data))

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
        pin_memory = True,
    )

    model = Recommender(
        numDir = numDir,
        numCast = numCast,
        numTags = numTags,
        movie_title_size = movie_title_size,
        directors_size = directors_size,
        actors_size = actors_size,
        tags_size = tags_size,
        lr=1e-3,
        dropout=0.1,
    )

    #model = model.to(device='cuda')
    model.load_state_dict(torch.load(checkpointPath)["state_dict"])
    model.eval()

    logger = TensorBoardLogger(
        save_dir=log_dir,
    )

    trainer = pl.Trainer(
        accelerator = 'gpu',
        devices = 1,
        logger=logger,
    )

    trainer.test(model, dataloaders = test_loader)

if __name__ == "__main__":
    #Parameters
    mode = "eval"
    numDir = 0
    numCast = 0
    numTags = 3
    entry_size = 100
    augment = True
    batch_size = 128
    epochs = 200

    # Confirm that the GPU is detected
    print("Is CUDA available? ", end = '')
    print(torch.cuda.is_available())
    # Get the GPU device name.
    device_name = torch.cuda.get_device_name()
    n_gpu = torch.cuda.device_count()
    print(f"Found device: {device_name}, n_gpu: {n_gpu}")
    torch.cuda.empty_cache()

    #Load ratings.csv and augmented data
    data = pd.read_csv('./ml-25m/ratings.csv')
    data.columns = ['userId','item_id','rating','timestamp']
    augmented = pd.read_csv('./MergedMovieIDMetaDataAllTags.csv', sep='\t')

    augmented.directedBy = augmented.directedBy.fillna('None')
    augmented.starring = augmented.starring.fillna('None')
    augmented.iloc[:, 8:] = augmented.iloc[:, 8:].fillna(0)

    #Delete all rows in the ratings file that contain a movieId not in the augmented data file
    data = data[~data['item_id'].isin(list(set(data['item_id'].tolist()) - set(augmented['item_id'].tolist())))]

    data.sort_values(by="timestamp", inplace=True)

    vocab_size = 0 #vocab_size number of items in vocab
    #Map movieIds
    data, mapping, inverse_mapping = map_column(data, col_name='item_id')
    vocab_size = len(mapping) + 3
    print("Mapped movieIds from " + str(3) + " to " + str(vocab_size))
    #Map directors
    directorsMap = map_directors(augmented['directedBy'].tolist(), vocab_size)
    vocab_size = vocab_size + len(directorsMap)
    print("Mapped directors from " + str(vocab_size - len(directorsMap)) + " to " + str(vocab_size))
    #Map actors
    actorsMap = map_actors(augmented['starring'].tolist(), vocab_size)
    vocab_size = vocab_size + len(actorsMap)
    print("Mapped actors from " + str(vocab_size - len(actorsMap)) + " to " + str(vocab_size))
    #Map tags
    tagsMap = map_tags(list(augmented.columns)[augmented.columns.get_loc('007'):], vocab_size)
    vocab_size = vocab_size + len(tagsMap)
    print("Mapped tags from " + str(vocab_size - len(tagsMap)) + " to " + str(vocab_size))
    vocab_size = len(mapping) + len(directorsMap) + len(actorsMap) + len(tagsMap)

    print("Total number of users: " + str(len(set(data['userId']))))
    print("Total number of movies: " + str(len(mapping)))
    print("Total number of directors: " + str(len(directorsMap)))
    print("Total number of actors: " + str(len(actorsMap)))
    print("Total number of tags: " + str(len(tagsMap)))

    #Map each movie to its entry vector
    print("Mapping data to correct input format: ")
    mappedData = augment_data(numDir, numCast, numTags, inverse_mapping, directorsMap, actorsMap, tagsMap, augmented)

    #Select ten percent of the groups randomly to be used as the test set
    random.seed(9001)
    grp_by = data.groupby(by="userId")
    groups = list(grp_by.groups)

    #Define vocab size depending on if data is augmented or not
    if not augment:
        vocab_size = len(mapping)

    movie_title_size = max(mapping.values()) + 1
    directors_size = max(directorsMap.values()) + 1
    actors_size = max(actorsMap.values()) + 1
    tags_size = max(tagsMap.values()) + 1



    #########################################

    #Free memory
    #del mapping
    #del directorsMap
    #del actorsMap
    #del tagsMap
    #del augmented
    #del data

    if mode == "train":
        train(
            grp_by_train = grp_by,
            trainGroups = groups,
            movie_title_size = movie_title_size,
            directors_size = directors_size,
            actors_size = actors_size,
            tags_size = tags_size,
            entry_size = entry_size,
            mappedData = mappedData,
            batch_size = batch_size,
            numDir = numDir,
            numCast = numCast,
            numTags = numTags,
            #fromCheckpoint = True,
            #checkpointPath = "./recommender_models/recommenderAugmented.ckpt",
            epochs = epochs,
            augment = augment,
        )
    elif mode == 'test':
        test(
            grp_by_test = grp_by,
            testGroups = groups,
            movie_title_size = movie_title_size,
            directors_size = directors_size,
            actors_size = actors_size,
            tags_size = tags_size,
            entry_size = entry_size,
            mappedData = mappedData,
            batch_size = batch_size,
            numDir = numDir,
            numCast = numCast,
            numTags = numTags,
            augment = augment,
            checkpointPath = "./recommender_models/tags3.ckpt"
        )
    elif mode == 'eval':
        #########################################
        #               EVAL                    #
        #########################################
        titletoID = augmented.groupby('title')['item_id'].agg(list).to_dict()
        IDtoTitle = augmented.groupby('item_id')['title'].agg(list).to_dict()

        allMovies = augmented['title'].tolist()
        #allMovies = list(map(str.lower,allMovies))
        currMovie = ""
        inputMovies = []
        inputIds = []
        count = 1
        while currMovie != "done":
            inputMovie = []
            currMovie = str(input("Enter a Movie Title else type \"done\": "))#.lower()
            if currMovie != "done":
                result = list(filter(lambda x: currMovie in x, allMovies))
                if len(result) == 0:
                    print("Movie not found.")
                else:
                    inputMovie.append(1)
                    inputMovie.append((titletoID[result[0]])[0])
                    inputIds.append((titletoID[result[0]])[0])
                    inputMovie.append(0)
                    inputMovie.append(count)
                    count += 1
                    inputMovies.append(inputMovie)
        inputMovies.append([1,1,0.0,count])

        #predictData = pd.DataFrame([[1,1,0.0,1],[1,13,0.0,2],[1,107,0.0,3],[1,596,0.0,4],[1,4896,0.0,5],[1,1,0.0,6]],columns = ['userId','item_id','rating','timestamp'])
        predictData = pd.DataFrame(inputMovies,columns = ['userId','item_id','rating','timestamp'])
        predictData['item_id_mapped'] = predictData['item_id'].apply(lambda x: mapping[x])
        inputMovies = (predictData['item_id'].tolist())[:-1]

        print("Input Movies: ", end='')
        for movie in inputMovies:
            print(" "+str(IDtoTitle[movie]), end=',')
        print()
        #Select ten percent of the groups randomly to be used as the test set
        random.seed(9001)
        grp_by_test = predictData.groupby(by="userId")
        testGroups = list(grp_by_test.groups)
        #Get movies ranked by popularity
        popular_items = items_by_popularity(testGroups, grp_by_test)

        srctestData, ytruetestData, negsamtestData = createData(
            groups=testGroups,
            grp_by=grp_by_test,
            split="test",
            p = 0.8,
            augment = augment,
            entry_size = entry_size,
            mappedData = mappedData,
            numAugments = 1 + numDir + numCast + numTags,
            popular_items = popular_items,
        )

        src_items = torch.tensor(srctestData, dtype=torch.long)
        trg_items = torch.tensor(ytruetestData, dtype=torch.long)

        model = Recommender(
            numDir = numDir,
            numCast = numCast,
            numTags = numTags,
            movie_title_size = movie_title_size,
            directors_size = directors_size,
            actors_size = actors_size,
            tags_size = tags_size,
            lr=1e-3,
            dropout=0.1,
        )

        #model = model.to(device='cuda')
        model.load_state_dict(torch.load("./recommender_models/tags3.ckpt")["state_dict"])
        model.eval()
        with torch.no_grad():
            out_data = model(src_items)
        scores = out_data[0,99]
        scores = scores.tolist()
        df = pd.DataFrame({'col':scores})
        df.sort_values(by = ['col'],inplace = True, ascending=False)
        count = 0
        for index, row in df.iterrows():
            if inverse_mapping[index] in inputIds:
                continue
            if count == 20:
                break
            print(str(IDtoTitle[inverse_mapping[index]]) + ", Prediction Value: "+str(row['col']))
            count += 1

        quit()
        recs = df.index.values.tolist()
        for rec in recs[0:20]:
            print(IDtoTitle[inverse_mapping[rec]])
