import torch
import torch.nn.functional as F
import numpy as np
import logging
import csv

from logging import handlers
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import BaseDataset
from model import Model

def init_logger(filename, when='D', backCount=3,
                fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
    logger = logging.getLogger(filename)
    format_str = logging.Formatter(fmt)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
    th.setFormatter(format_str)
    logger.addHandler(sh)
    logger.addHandler(th)

    return logger


TRAIN_DATA = 'data/train.csv'
DEV_DATA = 'data/eval.csv'
TRAIN_DATASET = 'data/train.pt'
DEV_DATASET = 'data/eval.pt'
RESULT_DATA = 'out/result.csv'
LOG_FILE = 'out/lstm.log'
MAX_LEN = 100
MIN_OCCURANCE = 2

BATCH_SIZE = 32
NUM_EPOCH = 8

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_CLASS = 2
DROP_OUT = 0.2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = init_logger(filename=LOG_FILE)

# get dataset and dataloader
# dataset = BaseDataset(TRAIN_DATA, MAX_LEN, MIN_OCCURANCE)
# torch.save(dataset, TRAIN_DATASET)
dataset = torch.load(TRAIN_DATASET)

train_size = int(0.85 * len(dataset))
train_data, dev_data = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
dev_dataloader = DataLoader(dev_data, batch_size = BATCH_SIZE, shuffle=True)

# define model
model = Model(len(dataset.word2idx), EMBEDDING_DIM, HIDDEN_DIM, 1, NUM_CLASS, DROP_OUT)
model.to(DEVICE)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# evaluating
def evaluate(dataloader):
    losses, accs = [], []
    for batch in dataloader:
        idx, sent, label = map(lambda x: x.to(DEVICE), batch)

        logits = model(sent)

        loss = F.nll_loss(logits, label)
        acc = (torch.argmax(logits, dim=-1) == label).sum().float() / label.shape[0]

        losses.append(loss.item())
        accs.append(acc.item())
    return (losses, accs)

# trainning
model.train()
for epoch in range(NUM_EPOCH):
    logger.info('=' * 100)
    losses, accs = [], []
    pbar = tqdm(total=len(train_dataloader))
    for batch in train_dataloader:
        idx, sent, label = map(lambda x: x.to(DEVICE), batch)

        logits = model(sent)

        loss = F.nll_loss(logits, label)
        acc = (torch.argmax(logits, dim=-1) == label).sum().float() / label.shape[0]

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accs.append(acc.item())

        pbar.set_description('Epoch: %2d | Loss: %.3f | ACCURACY: %.3f' % (epoch, loss.item(), acc.item()))
        pbar.update(1)

    pbar.close()

    # logger.info log
    dev_loss, dev_acc = evaluate(dev_dataloader)
    logger.info('Training:\t Accuracy: %.3f | Loss: %.3f' % (np.mean(accs), np.mean(losses)))
    logger.info('Evaluating:\t Accuracy: %.3f | Loss: %.3f' % (np.mean(dev_acc), np.mean(dev_loss)))
    logger.info('')

# writing result
result = open(RESULT_DATA, 'w', newline='')
result_writer = csv.writer(result)
result_writer.writerow(['id', 'target'])

# test_dataset = BaseDataset(DEV_DATA, MAX_LEN, MIN_OCCURANCE, word2idx=dataset.word2idx)
# torch.save(test_dataset, DEV_DATASET)
test_dataset = torch.load(DEV_DATASET)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

with torch.no_grad():
    for batch in tqdm(test_dataloader):
        idx, sent, label = map(lambda x: x.to(DEVICE), batch)
        pred = model(sent)

        pred_id = torch.argmax(pred, dim=-1).cpu().numpy().tolist()
        for i, label in zip(idx, pred_id):
            result_writer.writerow([idx, label])
