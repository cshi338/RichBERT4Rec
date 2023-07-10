from typing import Optional
from tqdm import tqdm
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn import functional as F
import random

def recall(scores, labels, k):
    scores = scores
    labels = labels
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / torch.min(torch.Tensor([k]).to(hit.device), (labels.sum(1).float() + 1e-10))).mean().cpu().item()

def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in labels.sum(1)])
    ndcg = dcg / (idcg + 1e-10)
    return ndcg.mean()

def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}

    scores = scores
    labels = labels
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    print()
    print("Calculating recall and ndcg for k = " + str(ks) + ": ")
    for k in tqdm(sorted(ks, reverse=True)):
       cut = cut[:, :k]
       hits = labels_float.gather(1, cut)
       metrics['Recall@%d' % k] = \
           (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())).mean().cpu().item()

       position = torch.arange(2, 2+k)
       weights = 1 / torch.log2(position.float())
       dcg = (hits * weights.to(hits.device)).sum(1)
       idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
       ndcg = (dcg / idcg).mean()
       metrics['NDCG@%d' % k] = ndcg.cpu().item()
    return metrics

def masked_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor):

    _, predicted = torch.max(y_pred, 1)

    y_true = torch.masked_select(y_true, mask)
    predicted = torch.masked_select(predicted, mask)

    acc = (y_true == predicted).double().mean()

    return acc


def masked_ce(y_pred, y_true, mask):

    loss = F.cross_entropy(y_pred, y_true, reduction="none")

    loss = loss * mask

    return loss.sum() / (mask.sum() + 1e-8)

def negative_sample(y_pred, y_true, mask, negative_samples):
    negative_samples = torch.repeat_interleave(negative_samples, int((mask.size(0)/negative_samples.size(0))), dim = 0)
    #Mask the rows that are not originally masked
    data = (y_pred.T * mask).T
    y_trues = y_true * mask
    negative_samples = (negative_samples.T * mask).T
    #Extract all non-masked rows
    masked = data[(torch.abs(data)).sum(dim=1) != 0]
    targets = y_trues[torch.nonzero(y_trues)]
    negative_samples = negative_samples[(torch.abs(negative_samples)).sum(dim=1) != 0]
    #Concatenate the negative samples with the target values s.t. the first 100 indices are the negative samples and the last index is the target
    indices = torch.cat((negative_samples, targets), 1)
    #Gather the scores for the negative samples and the target
    scores = masked.gather(dim = 1, index = indices)
    #Create tensor that holds the index of the target score, which in this case will always be index 100
    labels = [100] * scores.size(0)
    labels = torch.tensor(labels)
    labels = labels.to(device='cuda')
    return scores, labels

class Recommender(pl.LightningModule):
    def __init__(
        self,
        numDir,
        numCast,
        numTags,
        movie_title_size,
        directors_size,
        actors_size,
        tags_size,
        channels=64,
        cap=0,
        mask=1,
        dropout=0.5,
        lr=1e-4,
    ):
        super().__init__()
        self.valScores = []
        self.valLabels = []
        self.cap = cap
        self.mask = mask

        self.lr = lr
        self.dropout = dropout

        self.movie_title_size = movie_title_size
        self.directors_size = directors_size
        self.actors_size = actors_size
        self.tags_size = tags_size

        self.numDir = numDir
        self.numCast = numCast
        self.numTags = numTags

        self.movie_title_embeddings = torch.nn.Embedding(
            self.movie_title_size, embedding_dim=channels
        )
        self.directors_embeddings = torch.nn.Embedding(
            self.directors_size, embedding_dim=channels
        )
        self.actors_embeddings = torch.nn.Embedding(
            self.actors_size, embedding_dim=channels
        )
        self.tags_embeddings = torch.nn.Embedding(
            self.tags_size, embedding_dim=channels
        )
        self.input_pos_embedding = torch.nn.Embedding(512, embedding_dim=channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, dim_feedforward = 256, activation = "gelu", layer_norm_eps = 1e-12,  nhead=4, dropout=self.dropout
        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.linear_out = Linear(channels, self.movie_title_size)

        self.do = nn.Dropout(p=self.dropout)

    def encode_src(self, src_items):
        data = list(torch.split(src_items, int(src_items.size(1)/(1 + self.numDir + self.numCast + self.numTags)), dim = 1))
        #Embed movie IDs
        movie_titles = data.pop(0)
        src_items = self.movie_title_embeddings(movie_titles)
        #Embed Positions
        batch_size, in_sequence_len = src_items.size(0), src_items.size(1)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src_items.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_encoder = self.input_pos_embedding(pos_encoder)
        src_items += pos_encoder
        #Embed director IDs
        for i in range(self.numDir):
            directors = data.pop(0)
            directors = self.directors_embeddings(directors)
            src_items += directors
        #Embed actor IDs
        for i in range(self.numCast):
            actors = data.pop(0)
            actors = self.actors_embeddings(actors)
            src_items += actors
        #Embed tag IDs
        for i in range(self.numTags):
            tags = data.pop(0)
            tags = self.tags_embeddings(tags)
            src_items += tags

        src = src_items.permute(1, 0, 2)
        src = self.encoder(src)
        return src.permute(1, 0, 2)

    def forward(self, src_items):
        src = self.encode_src(src_items)
        out = self.linear_out(src)
        return out

    def training_step(self, batch, batch_idx):
        src_items, y_true = batch
        y_pred = self(src_items)

        src_items = src_items[:,0:int(src_items.size(1)/(1 + self.numDir + self.numCast + self.numTags))]

        y_pred = y_pred.view(-1, y_pred.size(2))
        y_true = y_true.view(-1)

        masked_items = src_items.contiguous().view(-1)
        mask = masked_items == self.mask

        loss = masked_ce(y_pred=y_pred, y_true=y_true, mask=mask)
        accuracy = masked_accuracy(y_pred=y_pred, y_true=y_true, mask=mask)

        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        src_items, y_true = batch
        y_pred = self(src_items)

        src_items = src_items[:,0:int(src_items.size(1)/(1 + self.numDir + self.numCast + self.numTags))]

        y_pred = y_pred.view(-1, y_pred.size(2))
        y_true = y_true.view(-1)

        masked_items = src_items.contiguous().view(-1)
        mask = masked_items == self.mask

        loss = masked_ce(y_pred=y_pred, y_true=y_true, mask=mask)
        accuracy = masked_accuracy(y_pred=y_pred, y_true=y_true, mask=mask)

        #Calculate nDCG@10
        scores = (y_pred.T * mask).T
        labels = y_true * mask
        #Extract all non-masked rows
        scores = scores[(torch.abs(scores)).sum(dim=1) != 0]
        labels = labels[torch.nonzero(labels)]
        labels = torch.flatten(labels)
        labels = F.one_hot(labels, num_classes = scores.size(1))
        rank = (-scores).argsort(dim=1)
        cut = rank[:, :10]
        labels = labels.gather(1, cut)
        scores = scores.gather(1, cut)
        self.valScores.append(scores)
        self.valLabels.append(labels)

        self.log("valid_loss", loss)
        self.log("valid_accuracy", accuracy)

        return loss

    def validation_epoch_end(self, outputs):
        #Calculate the nDCG@10 for the current validation epoch
        scores = torch.cat(self.valScores, dim = 0)
        labels = torch.cat(self.valLabels, dim = 0)
        nDCG = ndcg(scores, labels, 10)
        self.valScores = []
        self.valLabels = []
        self.log("valid_nDCG10", nDCG)

    def test_step(self, batch, batch_idx):
        src_items, y_true, neg_sam = batch
        y_pred = self(src_items)

        src_items = src_items[:,0:int(src_items.size(1)/(1 + self.numDir + self.numCast + self.numTags))]

        y_pred = y_pred.view(-1, y_pred.size(2))
        y_true = y_true.view(-1)

        masked_items = src_items.contiguous().view(-1)
        mask = masked_items == self.mask

        y_pred, y_true = negative_sample(y_pred = y_pred, y_true = y_true, mask = mask, negative_samples = neg_sam)

        return y_pred, y_true

    def test_epoch_end(self, outputs):
        scores = []
        labels = []
        for dataloader_outputs in outputs:
            y_pred = dataloader_outputs[0]
            scores.append(y_pred)
            y_true = dataloader_outputs[1]
            labels.append(F.one_hot(y_true))
        scores = torch.cat(scores, dim = 0)
        labels = torch.cat(labels, dim = 0)

        ks = [1,5,10,20,50,100]
        self.log("final_metric", recalls_and_ndcgs_for_ks(scores, labels, ks))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size = 25, gamma = 1.0
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid_loss",
        }
