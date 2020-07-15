import torch
from torch.autograd import  variable
w1=variable(torch.tensor([1.0,2.0,3.0]),requires_grad=True)#需要更新梯度
w2=variable(torch.tensor([1.0,2.0,3.0]),requires_grad=True)

d=torch.mean(w1)
d.backward()
w1.grad
d.backward()
w1.grad

print(w1.grad)
print(w2.grad)


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.is_training = True
        self.dropout_rate = config.dropout_rate
        self.num_class = config.num_class
        self.use_element = config.use_element
        self.config = config

        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                      embedding_dim=config.embedding_size)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=config.embedding_size,
                                    out_channels=config.feature_size,
                                    kernel_size=h),
                          #                              nn.BatchNorm1d(num_features=config.feature_size),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=config.max_text_len - h + 1))
            for h in config.window_sizes
        ])
        self.fc = nn.Linear(in_features=config.feature_size * len(config.window_sizes),
                            out_features=config.num_class)
        if os.path.exists(config.embedding_path) and config.is_training and config.is_pretrain:
            print("Loading pretrain embedding...")
            self.embedding.weight.data.copy_(torch.from_numpy(np.load(config.embedding_path)))

    def forward(self, x):
        embed_x = self.embedding(x)

        # print('embed size 1',embed_x.size())  # 32*35*256
        # batch_size x text_len x embedding_size  -> batch_size x embedding_size x text_len
        embed_x = embed_x.permute(0, 2, 1)
        # print('embed size 2',embed_x.size())  # 32*256*35
        out = [conv(embed_x) for conv in self.convs]  # out[i]:batch_size x feature_size*1
        # for o in out:
        #    print('o',o.size())  # 32*100*1
        out = torch.cat(out, dim=1)  # 对应第二个维度（行）拼接起来，比如说5*2*1,5*3*1的拼接变成5*5*1
        # print(out.size(1)) # 32*400*1
        out = out.view(-1, out.size(1))
        # print(out.size())  # 32*400
        if not self.use_element:
            out = F.dropout(input=out, p=self.dropout_rate)
            out = self.fc(out)
        return out

