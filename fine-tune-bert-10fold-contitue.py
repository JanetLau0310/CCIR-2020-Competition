
import numpy as np 
import pandas as pd 
import os
import torch
from torch import nn
import torch.optim as optim
import time
import gc
import random
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
from keras.preprocessing import text, sequence
from keras.utils import to_categorical
from torch.utils import data
from torch.nn import functional as F
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from nltk.tokenize.treebank import TreebankWordTokenizer
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel


def calculate_F1(y_test, test_preds):
    y_test_hat = np.argmax(test_preds, 1)
    f1_score = metrics.f1_score(y_test, y_test_hat, average='macro')  
    return f1_score

class CustomModel(nn.Module):
    
    def __init__(self, bert_path):
        super().__init__()
        # 加载并冻结bert模型参数
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.output = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, 3)
        )
    def forward(self, seqs):
        _, pooled = self.bert(seqs, output_all_encoded_layers=False)
#         concat = torch.cat([pooled, features], dim=1)
#         logits = self.output(concat)
        logits = self.output(pooled)
        return logits
    

def convert_bert_token(data):
    i = 0
    des_list = list()
    for sen in data:
        i += 1
        tokens = tokenizer.tokenize(sen)
        tokens = ["[CLS]"] + tokens
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        if len(input_ids) < MAX_LEN:    
            input_ids = input_ids + [0]*(MAX_LEN-len(input_ids))
        else:
            input_ids = input_ids[0:MAX_LEN]
        des_list.append(input_ids)
    return des_list


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_model(model, train, val, y_val, test, loss_fn, kfold, output_dim=3, lr=0.000005,
                batch_size=32, n_epochs=3,
                enable_checkpoint_ensemble=True):
    param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
    optimizer = torch.optim.Adam(param_lrs, lr=lr)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    max_f1_score = 0

    for epoch in range(n_epochs):
        start_time = time.time()

        optimizer.step()

        model.train()
        avg_loss = 0.

        for data in tqdm(train_loader, disable=False):
            x_batch = data[:-1]
            y_batch = data[-1]
            y_pred = model(x_batch[0])
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            avg_loss += loss.item() / len(train_loader)

        model.eval()
        val_preds = np.zeros((len(val), output_dim))
        test_preds = np.zeros((len(test), output_dim))

        for i, x_batch in enumerate(val_loader):
            y_pred = sigmoid(model(x_batch[0]).detach().cpu().numpy())

            val_preds[i * batch_size:(i + 1) * batch_size, :] = y_pred


        for i, x_batch in enumerate(test_loader):
            y_pred = sigmoid(model(x_batch[0]).detach().cpu().numpy())

            test_preds[i * batch_size:(i + 1) * batch_size, :] = y_pred

        f1_score = calculate_F1(y_val, val_preds)
        elapsed_time = time.time() - start_time
        print('Kfold {} \t Epoch {}/{} \t loss={:.4f} \t f1={:.4f} \t time={:.2f}s'.format(
            kfold, epoch, n_epochs, avg_loss, f1_score, elapsed_time))
        if f1_score > max_f1_score:
            max_f1_score = f1_score
            torch.save(model, str(kfold) + "net.pkl")  # 保存整个网络


    return val_preds, test_preds

seed_everything(1234)
BERT_FP = '/kaggle/input/bertbasechinese/'
tokenizer = BertTokenizer(vocab_file='/kaggle/input/bertbasechinese/vocab.txt')
MAX_LEN = 180
MODEL_NAME = 'my_model'

train_f = open("/kaggle/input/ncovcov/nCoV_100k_train.labled.csv", encoding="gbk",errors='ignore')
test_f = open("/kaggle/input/ncovcov/nCov_10k_test.csv", encoding="gbk",errors='ignore')

train = pd.read_csv(train_f)
test = pd.read_csv(test_f)
train = train.fillna('') #将缺失值设为空串
test = test.fillna('') #将缺失值设为空串

y_data = list()
x_data = list()
for i in range(100000):
    if train['情感倾向'][i] in ['-1', '0', '1']: # 情感倾向有些异常值，需要去掉这些句子
        x_data.append(train['微博中文内容'][i])
        y_data.append(int(train['情感倾向'][i]))

x_data = convert_bert_token(x_data)
x_test = convert_bert_token(test['微博中文内容'])
x_data = np.array(x_data)
y_data = np.array(y_data)
skf = StratifiedKFold(n_splits=10, random_state=42)
for train_index, val_index in skf.split(x_data, y_data):
    x_train, x_test = x_data[train_index], x_data[val_index]
    y_train, y_test = y_data[train_index], y_data[val_index]
    break
kfold = 0
x_trainset = x_train
y_trainset = y_train
for train_index, val_index in skf.split(x_train, y_train):
    if kfold < 6: #继续训练没训练完的折
        kfold += 1
        continue
    x_train, x_val = x_trainset[train_index], x_trainset[val_index]
    y_train, y_val = y_trainset[train_index], y_trainset[val_index]    

    for i in range(y_train.size):
        y_train[i] += 1 #类别原本为 -1，0，1转换为 0，1，2，因为后面转成onehot的函数输入不能有负数

    for i in range(y_val.size):
        y_val[i] += 1 #方便评测计算
    y_train = to_categorical(y_train)
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
    x_val = sequence.pad_sequences(x_val, maxlen=MAX_LEN)
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

    x_train_torch = torch.tensor(x_train, dtype=torch.long).cuda()
    x_val_torch = torch.tensor(x_val, dtype=torch.long).cuda()
    x_test_torch = torch.tensor(x_test, dtype=torch.long).cuda()
    y_train_torch = torch.tensor(y_train, dtype=torch.float32).cuda()


    train_dataset = data.TensorDataset(x_train_torch, y_train_torch)
    val_dataset = data.TensorDataset(x_val_torch)
    test_dataset = data.TensorDataset(x_test_torch)
    model = CustomModel(BERT_FP)
    model.cuda() # GPU的内存13G,CPU的内存有16G，内存不够时可用CPU
    print("start training")
    val_preds, test_preds = train_model(model, train_dataset, val_dataset, y_val, test_dataset, nn.BCEWithLogitsLoss(reduction='mean'), kfold)
    kfold += 1