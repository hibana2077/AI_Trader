import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):#input_size should be (batch_size, seq_len, input_size)
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        
        # 分離最後一層以生成特定格式的輸出
        self.fc2a = nn.Linear(hidden_size, 3)  # 用於生成 [x, x, x]
        self.fc2b = nn.Linear(hidden_size, 1)  # 用於生成單一的 x
        self.fc2c = nn.Linear(hidden_size, 2)  # 用於生成 [x, x]
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.dropout(out)
        
        out = F.relu(self.fc1(out[:, -1, :]))  # 注意這裡取了 LSTM 輸出的最後一個時間點
        
        # 使用不同的全連接層來生成不同部分的輸出
        out_a = F.softmax(self.fc2a(out), dim=1) # 生成 [x, x, x] 由於前面的是要用於分類，所以使用 softmax，決定[買][賣][持有]
        out_b = self.fc2b(out) # 決定(買 or 賣)的數量
        out_c = F.softmax(self.fc2c(out), dim=1) # 決定是否要反向交易
        
        # 將兩個輸出合併為所需的格式
        final_out = [out_a, out_b.squeeze(-1), out_c]
        
        return final_out
