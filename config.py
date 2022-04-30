from importlib.metadata import metadata
import argparse
from importlib.metadata import metadata

args = argparse.Namespace(
    lr = 0.001,
    epoch = 100,
    bs = 32,
    train_size = 0.8,
    path = "./data",
    metadata = "./data/cancer_reg.csv",
)
