import os
import pickle


def save(obj, fname):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)


def load(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj
