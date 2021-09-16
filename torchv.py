'''
函数说明: 
Author: hongqing
Date: 2021-09-03 11:31:51
LastEditTime: 2021-09-16 17:10:03
'''
import numpy as np
from matplotlib.pyplot import imshow
from numpy.core.fromnumeric import mean, std
%matplotlib inline
np.random.seed(1)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader,Dataset
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
from faker import Faker
from babel.dates import format_date
from td_utils import *

x = graph_spectrogram("audio_examples/example_train.wav")
_, data = wavfile.read("audio_examples/example_train.wav")
print("Time steps in audio recording before spectrogram", data[:,0].shape)
print("Time steps in input after spectrogram", x.shape)
Tx = 5511 # 时间步
n_freq = 101 # 每个时间步有多少个频率值
Ty = 1375 # 神经网络的输出时间步数量
# 使用pydub模块加载语音片段
activates, negatives, backgrounds = load_raw_audio()
print("background len: " + str(len(backgrounds[0])))# 背景音是10秒语音，所以pydub将其转换成立100000个数值
print("activate[0] len: " + str(len(activates[0])))# 因为一秒内就可以说完"activate"这个单词了，所以是721个数值
print("activate[1] len: " + str(len(activates[1])))# 因为每个人说完"activate"这个单词的时间不一样，所以这个是731
def get_random_time_segment(segment_ms):
    """
    随机从10秒背景中选取一个长度为segment_ms毫秒的片段
    
    返回值:
    segment_time -- 返回片段的开始时间和结束时间
    """
    
    segment_start = np.random.randint(low=0, high=10000-segment_ms) 
    segment_end = segment_start + segment_ms - 1
    
    return (segment_start, segment_end)

def is_overlapping(segment_time, previous_segments):
    """
    用于检测当前要插入的词的时间是否与之前词的位置重叠了
    
    参数:
    segment_time -- 当前要插入的词的起止时间(segment_start, segment_end)
    previous_segments -- 这是一个list列表，包含了之前已经插入的所有词的起止时间(segment_start, segment_end) 
    
    返回值:
    如果重叠了，那么返回true，否则返回false
    """
    
    segment_start, segment_end = segment_time
    
    overlap = False
    
    for previous_start, previous_end in previous_segments:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True

    return overlap

  # 往背景音中合成一个词语音

def insert_audio_clip(background, audio_clip, previous_segments):
    """    
    参数:
    background -- 10秒长的背景音 
    audio_clip -- 准备要合成进入的词语音
    previous_segments -- 这是一个list列表，包含了之前已经插入的所有词的起止时间(segment_start, segment_end) 
    
    返回值:
    new_background -- 合成了词语音后的10秒语音判断，包含了背景音和词语音
    """
    
    # 获取词语音的长度
    segment_ms = len(audio_clip)
    
    # 根据词语音的长度从10秒背景音片段中随机选取一个时间段，用来插入词语音
    segment_time = get_random_time_segment(segment_ms)
    
    # 检测上面随机选取的时间段是否与之前插入的词语音的位置重叠了，如果重叠了，那么重新选取一个时间段
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)

    previous_segments.append(segment_time)
    
    # 将词语音合成到背景音中
    new_background = background.overlay(audio_clip, position = segment_time[0])
    
    return new_background, segment_time

def insert_ones(y, segment_end_ms):
    """  
    参数:
    y -- y标签，维度是 (1, Ty)
    segment_end_ms -- 插入的词语音在10秒背景音中的结束位置
    
    返回值:
    y -- 更新后的y标签
    """
    
    # 将词语音的结束时间的单位从pydub单位转换成输出时间步的单位，也就是从10000的单位转换成1375的单位
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    
    
    for i in range(segment_end_y + 1, segment_end_y + 51):# 将后面的50个数值都设置为1
        if i < Ty:
            y[0, i] = 1
    
    return y

def create_training_example(background, activates, negatives):
    """
    用背景音，唤醒词语音和其它词语音合成一个训练样本
    
    返回值:
    background -- 10秒背景音
    activates -- 这里面包含了所有唤醒词语音，不是一个哦
    negatives -- 这里面包含了所有其它词语音，不是一个哦
    
    返回值:
    x -- 训练样本的频谱，也就是5511个数值
    y -- 这个训练样本对应的y标签
    """
    
    np.random.seed(18)
    
    # 将背景音的音量调小一点
    background = background - 20

    y = np.zeros((1, Ty))

    previous_segments = []
    
    # 从唤醒词语音列表中随机选取几个出来
    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]
    
    ### 将随机选取的唤醒词语音一个个插入到背景音中
    for random_activate in random_activates:
        # 插入唤醒词语音
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        segment_start, segment_end = segment_time
        # 修改y标签
        y = insert_ones(y, segment_end_ms=segment_end)

    # 从其它语音列表中随机选取几个出来
    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

     ### 将随机选取的其它词语音一个个插入到背景音中
    for random_negative in random_negatives:
        background, _ = insert_audio_clip(background, random_negative, previous_segments)
    
    background = match_target_amplitude(background, -20.0)

    # 将这个训练样本保存到文件中
    # file_handle = background.export("train" + ".wav", format="wav")
    # print("File (train.wav) was saved in your directory.")
    
    # 将训练样本的语音转换成频谱
    x = graph_spectrogram("train.wav")
    
    return x, y


# 加载预先处理过了的训练集
X = np.load("./XY_train/X.npy")
Y = np.load("./XY_train/Y.npy")
X_dev = np.load("./XY_dev/X_dev.npy")
Y_dev = np.load("./XY_dev/Y_dev.npy")
class traindata(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.X = np.load("./XY_train/X.npy")
        self.Y = np.load("./XY_train/Y.npy")
        
    def __getitem__(self, index):
        return self.X[index],self.Y[index]

    def __len__(self):
        return self.X.shape[0]

def my_collate(batch):
    data = [item[0] for item in batch]
    # data = torch.tensor(data,dtype=torch.float16)
    data = torch.FloatTensor(data)
    data = torch.transpose(data,1,2).half()

    target = [item[1] for item in batch]
    # target = torch.tensor(target,dtype=torch.float16)
    target = torch.FloatTensor(target)
    return [data, target] 

Xdata = traindata()
datas = dataloader.DataLoader(Xdata,5,True,collate_fn=my_collate)



class RNN(nn.Module):
    def __init__(self,):
        super(RNN, self).__init__()
        self.drop = nn.Dropout(0.8)
        self.relu = nn.ReLU()
        #torch.Size([26, 101, 5511])
        self.conv1d = nn.Conv1d(101,196,15,4)
        self.bn1 = nn.BatchNorm1d(1375)
        #torch.Size([26, 196, 1375])  to torch.Size([26, 1375,196])
        self.gru1 = nn.GRU(196, 128,1,batch_first=True)
        self.bn2 = nn.BatchNorm1d(1375)
        self.gru2 = nn.GRU(128, 128,1,batch_first=True)
        self.dense = nn.Linear(128,2)

    def forward(self, input):
        x = self.conv1d(input)
        # torch.Size([26, 196, 1375])  to torch.Size([26, 1375,196])
        x = torch.transpose(x,1,2)
        x = self.drop(self.relu(self.bn1(x)))
        x,_ = self.gru1(x)
        x = self.bn2(self.relu(x))
        x,_ = self.gru2(x)
        x = self.drop(self.bn2(self.drop(x)))
        output = self.dense(x)
        return output

def calacc(output,Y_dev):
    a = np.sum(output.detach().numpy().round()==Y_dev.astype(np.long))
    b = Y_dev.shape[0]*Y_dev.shape[1]
    return a*100/b
def calac(output,Y_dev):
    output = torch.unsqueeze(output.argmax(2),2)
    a = np.sum(output.detach().numpy()==Y_dev.astype(np.long))
    b = Y_dev.shape[0]*Y_dev.shape[1]
    return a*100/b

net = RNN()
optimizer = torch.optim.Adam(net.parameters(),lr=0.00001)
criterion = nn.CrossEntropyLoss()
totalloss=0

for i in range(10):
    net.train() 
    for _,(x,y) in enumerate(datas):
        output = net(x)
        loss = criterion(output.reshape(-1,2), y.reshape(-1).long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totalloss+=(loss.item())
    print('episode{} finished,loss = {}'.format(i,totalloss))
    totalloss = 0
    # 测试精度
    net.eval()
    with torch.no_grad():
        output = net(torch.transpose(torch.FloatTensor(X),1,2))
        print('train accuracy = {}%'.format(calac(output,Y)))
        output = net(torch.transpose(torch.FloatTensor(X_dev),1,2))
        print('test accuracy = {}%'.format(calac(output,Y_dev)))


