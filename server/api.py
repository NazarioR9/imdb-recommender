import pickle
import random
from itertools import permutations, combinations

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel as BaseObject
from simpletransformers.custom_models.models import RobertaForMultiLabelSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from transformers import RobertaConfig, RobertaTokenizer

PATH = 'server/assets/model/'
MODEL_NAME = 'roberta-base'


class Item(BaseObject):
    plot: str


class Recommender:
    def __init__(self):
        self.label_mapper = None
        self.predictor = None
        self.embeddings = None
        self.db = None
        self.__load()

    def __load(self):
        with open(f'{PATH}label_mapper.pkl', 'rb') as f:
            self.label_mapper = pickle.load(f)
            self.label_mapper = {idx: label for label, idx in self.label_mapper.items()}

        self.embeddings = {}
        with open('server/assets/embeddings/embeddings.pkl', 'rb') as f:
            self.embeddings = pickle.load(f)

        self.predictor = IMDBModel()
        self.predictor.eval()

        self.db = pd.read_csv('server/assets/db/data.csv')

    def __infer(self, data):
        x, embeddings = self.predictor(data)

        x = x.detach().cpu().numpy()[0]
        labels = np.argwhere(x >= 0.5).squeeze(axis=1)
        if len(labels) == 0:
            labels = [np.argmax(x)]
        labels = [self.label_mapper[label] for label in labels]

        return x, labels, embeddings.detach().cpu().numpy()

    def __find_embedding(self, labels: list):
        combos = list(self.embeddings.keys())
        found = False
        embs = []

        for combo in permutations(labels):
            key = '|'.join(combo)
            c = self.embeddings.get(key, None)
            if c is not None:
                found = True
                embs += [c]
                break

        if not found:  # find combinations with 2 of the labels
            for ls in combinations(labels, 2):
                for combo in permutations(ls):
                    key = '|'.join(combo)
                    c = self.embeddings.get(key, None)
                    if c is not None:
                        found = True
                        embs += [c]

        if not found:  # find combinations with all the labels
            _combos = [c for c in combos if all([label in c for label in labels])]
            if len(_combos) != 0:
                found = True
                embs = [self.embeddings.get(combo) for combo in _combos]

        if not found:  # gather embeddings with at least one of the labels (then randomly sample 3 of them)
            _combos = [c for c in combos if any([label in c for label in labels])]
            _combos = random.sample(_combos, 3)
            embs = [self.embeddings.get(combo) for combo in _combos]

        embeddings, index = [], []
        for emb in embs:
            embeddings += emb['embedding']
            index += emb['index']

        return {'embedding': embeddings, 'index': index}

    def __recommend(self, embeddings, target_embeddings, topk = 5):
        cosine = cosine_similarity(target_embeddings['embedding'], embeddings).squeeze(axis=1)
        cosine_topk_argmax = cosine.argsort()[-topk:][::-1]
        index = np.array(target_embeddings['index'])[cosine_topk_argmax]

        films = self.db.loc[index]
        films = [
            {'title': row['title'], 'plot': row['plot']}
            for _, row in films.iterrows()
        ]

        return films

    def __call__(self, item):
        data = InferDataset(item).get()
        probas, labels, embeddings = self.__infer(data)
        target_embeddings = self.__find_embedding(labels)
        films = self.__recommend(embeddings, target_embeddings)

        return {'genres': labels, 'films': films, 'proba': dict(zip(self.label_mapper.values(), probas.tolist()))}


class InferDataset:
    tokenizer = RobertaTokenizer.from_pretrained(PATH)

    def __init__(self, item: Item):
        self.transformer = None
        self.encoding = None
        self.item = item

    def __transform(self, text):
        if self.encoding is None:
            enc = self.tokenizer(
                [text],
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors="pt")
            self.encoding = enc['input_ids']
        return self.encoding

    def get(self):
        return self.__transform(self.item.plot)


class IMDBModel(nn.Module):
    config = RobertaConfig.from_pretrained(PATH)

    def __init__(self):
        super(IMDBModel, self).__init__()

        self.transformer = RobertaForMultiLabelSequenceClassification.from_pretrained(PATH, config=self.config)

    def embeddings(self, features):
        x = features[:, 0, :]
        x = self.transformer.classifier.dropout(x)
        x = self.transformer.classifier.dense(x)

        return x

    def features(self, x):
        outputs = self.transformer.roberta(x)
        sequence_output = outputs[0]
        embeddings = self.embeddings(sequence_output)

        return embeddings

    def forward(self, x):
        embeddings = self.features(x)
        x = torch.tanh(embeddings)
        x = self.transformer.classifier.dropout(x)
        x = self.transformer.classifier.out_proj(x)
        x = torch.sigmoid(x)

        return x, embeddings
