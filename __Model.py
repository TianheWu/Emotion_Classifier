import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):

    # vocab_size: 词典大小
    # embedding_dim: 词向量维度
    # n_fliters: 卷积核的个数
    # filter_sizes: 卷积核的尺寸
    # output_dim: 输出的维度
    # pad_idx: 填充0的索引
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)


def train_epoch(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    train_corrects = 0
    train_num = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        pre = model(batch.text[0]).squeeze(1)
        # 用 pre(预测出的值) 和真实值比较
        loss = criterion(pre, batch.label.type(torch.FloatTensor))
        # 四舍五入，利用sigmod来判定为0或1
        pre_lab = torch.round(torch.sigmoid(pre))
        train_corrects += torch.sum(pre_lab.long() == batch.label)
        train_num += len(batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    # 所有样本的平均损失
    epoch_loss = epoch_loss / train_num
    # 所有样本的精度
    epoch_acc = train_corrects.double().item() / train_num
    return epoch_loss, epoch_acc


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    train_corrects = 0
    train_num = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            pre = model(batch.text[0]).squeeze(1)
            # 用 pre(预测出的值) 和真实值比较
            loss = criterion(pre, batch.label.type(torch.FloatTensor))
            # 四舍五入，利用sigmod来判定为0或1
            pre_lab = torch.round(torch.sigmoid(pre))
            train_corrects += torch.sum(pre_lab.long() == batch.label)
            train_num += len(batch.label)
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / train_num
        epoch_acc = train_corrects.double().item() / train_num
    return epoch_loss, epoch_acc

