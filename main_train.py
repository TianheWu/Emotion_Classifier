from pickle import GLOBAL
from typing import Sequence, Text
from torchtext.vocab import Vectors, GloVe
from torchtext.legacy import data, datasets, vocab
from __Parameters import *
from __Model import TextCNN, train_epoch, evaluate
import torch
import time
import copy


customize_tokensize = lambda x: x.split()

# sequential: 输入是否为序列化的
# include_length: 返回本序列的长度
# use_vocab: 是否使用Vocab，否则Field的对象是数字类型的
# pad_token: 用于填充文本的关键字
# unk_token: 用于填充不在词汇表中的关键字
TEXT = data.Field(sequential=True, tokenize=customize_tokensize, include_lengths=True, use_vocab=True, batch_first=True, fix_length=200)
LABEL = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)
train_fileds = [("text", TEXT), ("label", LABEL)]
train_data = data.TabularDataset(
    path=r"./imdb_data.csv",
    format='csv',
    skip_header=True,
    fields=train_fileds
)
train_data_real, val_data_real = train_data.split(split_ratio=0.7)
vec = Vectors("glove.6B.100d.txt", "./Emotion")
# 将训练集转换为词向量
TEXT.build_vocab(train_data_real, max_size=20000, vectors=vec)
LABEL.build_vocab(train_data_real)
# print(TEXT.vocab.freqs.most_common(n=10))
# print("类别标签情况: ", LABEL.vocab.freqs)
# print("词典个数: ", len(TEXT.vocab.itos))

# 定义加载器
train_iter = data.BucketIterator(train_data_real, batch_size=BATCH_SIZE)
val_iter = data.BucketIterator(val_data_real, batch_size=BATCH_SIZE)

INPUT_DIM = len(TEXT.vocab) # 词典数量
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = TextCNN(INPUT_DIM, EMBEDDING_DIM, N_FILITERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

# 将导入的词向量作为embedding.weight的初值
pretrained_embedding = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embedding)
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.BCEWithLogitsLoss()

best_val_loss = float("inf")
best_acc = float(0)
for epoch in range(EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train_epoch(model, train_iter, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_iter, criterion)
    end_time = time.time()
    print("Epoch:", epoch + 1, "|", "Epoch Time: ", end_time - start_time, "s")
    print("Train Loss:", train_loss, "|", "Train Acc: ", train_acc)
    print("Val Loss: ", val_loss, "|", "Val Acc: ", val_acc)
    if (val_loss < best_val_loss) & (val_acc > best_acc):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_val_loss = val_loss
        best_acc = val_acc

model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), SAVE_MODEL_PARAMETERS)







