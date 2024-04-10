#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('nvidia-smi')


# In[2]:


import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import math
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import random 
import seaborn as sns
import matplotlib
import pickle
from layers import GraphConvolution
from sklearn.model_selection import train_test_split
import GearyC
import LISA

torch.cuda.get_device_name()


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

torch.set_printoptions(edgeitems=100)
def accuracy(vector_x, vector_y):

  # torch.Size([283200])
  new_v = vector_x - vector_y
  new_v = torch.abs(new_v)
  new_v = torch.sum(new_v).data.cpu().numpy()

  return new_v/len(vector_x)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm



# In[4]:


from torch.utils.data import Dataset

class DM_Dataset(Dataset):
    def __init__(self, X_input, Y_input, adj_input):
        self.X_input = Variable(torch.Tensor(X_input).float())
        self.Y_input = Variable(torch.Tensor(Y_input).float())
        self.adj_input = Variable(torch.Tensor(adj_input).float())

    def __len__(self):
        return len(self.X_input)

    def __getitem__(self, idx):
        return self.X_input[idx], self.Y_input[idx], self.adj_input[idx]

class DM_Dataset_test(Dataset):
    def __init__(self, X_input, Y_input, adj_input, total_xy):
        self.X_input = Variable(torch.Tensor(X_input).float())
        self.Y_input = Variable(torch.Tensor(Y_input).float())
        self.adj_input = Variable(torch.Tensor(adj_input).float())
        self.total_xy = Variable(torch.Tensor(total_xy).float())
        
    def __len__(self):
        return len(self.X_input)

    def __getitem__(self, idx):
        return self.X_input[idx], self.Y_input[idx], self.adj_input[idx], self.total_xy[idx]
    
X = np.load(open('X_Iowa.npy', 'rb'))
Y = np.load(open('Y_Iowa.npy', 'rb'))

file_to_read = open("adj_dict_pc_weekly.json", "rb")
adj_dict = pickle.load(file_to_read)

mask = np.load(open('mask_128_64.npy', 'rb'))


x_len = X.shape[2]
y_len = X.shape[3]

import seaborn as sns
import matplotlib

# In[5]:


def coor_matrix_filter(coors_matrix):
    #x_range = [42,74]
    #y_range = [10,42]
    final = []
    for i in range(len(coors_matrix)):
        out = []
        for each in coors_matrix[i]:
            if each[0] < x_range[0] or each[0] >= x_range[1] or each[1] < y_range[0] or each[1] >= y_range[1]:
                continue
            out.append(each)
        final.append(out)
    final = np.array(final, dtype=object)
    return final


# In[6]:



S = 3

def adj_generator(temp_x, temp_y, cell_x, cell_y, window_size):
    
    win_x = temp_x - cell_x
    win_y = temp_y - cell_y
    
    out = []
    
    for x in range(window_size):
        for y in range(window_size):
            try:
                value = adj_dict[str(temp_x)+"-"+str(temp_y)][str(win_x+x)+"-"+str(win_y+y)]
            except:
                value = 0
            out.append(value)
            
    return out


def generate_small_cell(x1, y1, x2, y2, S, limitx, limity):
    if x1 > S*2 or y1 > S*2:
        print("S error")
    if x2 >= limitx or y2 >= limity:
        print("Limit error")
    origin = [x2 - S, y2 - S]
    x1 = x1 + origin[0]
    y1 = y1 + origin[1]
    if 0 > x1 or x1 >= limitx:
        x1 = -1
    if 0 > y1 or y1 >= limity:
        y1 = -1
    return x1, y1

#temp = generate_small_cell(6, 6, 31, 31, S, 32)
#print(temp)

def generate_samples_by_cells(coors_matrix, X, Y):
    batch, period, len_x, len_y, features = X.shape

    coors = coors_matrix
    num_coor = len(coors)
    total_model = []
    
    adj_matrix_total = []
    
    for each in coors:
        x, y = each[0], each[1]
        window_size = 2*S + 1
        single = np.zeros((batch, period, window_size, window_size, features))
        adj_matrix = np.zeros((batch, window_size**2, window_size**2))
        
        for cell_x in range(2*S + 1):
            for cell_y in range(2*S + 1):

                temp_x, temp_y = generate_small_cell(cell_x, cell_y, x, y, S, 128, 64)
                adj_matrix[:, cell_x*window_size + cell_y] = adj_generator(temp_x, temp_y, cell_x, cell_y, window_size)
                
                if temp_x < 0 or temp_y < 0:
                    continue

                single[:,:,cell_x,cell_y] = X[:,:,temp_x,temp_y]
                
        total_model.append(single)
        adj_matrix_total.append(adj_matrix)
                                  
    total_model = np.array(total_model)
    adj_matrix_total = np.array(adj_matrix_total)
    total_model = total_model.reshape(num_coor*batch, period, window_size, window_size, features)
    adj_matrix_total = adj_matrix_total.reshape(num_coor*batch, window_size**2, window_size**2)
    
    new_X = total_model

    batch, period, len_x, len_y = Y.shape

    coors = coors_matrix
    num_coor = len(coors)
    total_model = []
    for each in coors:
        x, y = each[0], each[1]
        single = np.zeros((batch, period))

        single[:,:] = Y[:,:,x,y]

        total_model.append(single)
    total_model = np.array(total_model)
    total_model = total_model.reshape(num_coor*batch, period)
    new_Y = total_model

    return new_X, new_Y, adj_matrix_total

def generate_samples_by_cells_test(coors_matrix, X, Y):
    batch, period, len_x, len_y, features = X.shape

    coors = coors_matrix
    num_coor = len(coors)
    total_model = []
    
    adj_matrix_total = []
    
    for each in coors:
        x, y = each[0], each[1]
        window_size = 2*S + 1
        single = np.zeros((batch, period, window_size, window_size, features))
        adj_matrix = np.zeros((batch, window_size**2, window_size**2))
        
        for cell_x in range(2*S + 1):
            for cell_y in range(2*S + 1):

                temp_x, temp_y = generate_small_cell(cell_x, cell_y, x, y, S, 128, 64)
                adj_matrix[:, cell_x*window_size + cell_y] = adj_generator(temp_x, temp_y, cell_x, cell_y, window_size)
                
                if temp_x < 0 or temp_y < 0:
                    continue

                single[:,:,cell_x,cell_y] = X[:,:,temp_x,temp_y]
                
        total_model.append(single)
        adj_matrix_total.append(adj_matrix)
                                  
    total_model = np.array(total_model)
    adj_matrix_total = np.array(adj_matrix_total)
    total_model = total_model.reshape(num_coor*batch, period, window_size, window_size, features)
    adj_matrix_total = adj_matrix_total.reshape(num_coor*batch, window_size**2, window_size**2)
    
    new_X = total_model

    batch, period, len_x, len_y = Y.shape

    coors = coors_matrix
    num_coor = len(coors)
    total_model = []
    total_xy = []
    
    for each in coors:
        x, y = each[0], each[1]
        single = np.zeros((batch, period))
        single_xy = np.zeros((batch, period, 2))
        
        single[:,:] = Y[:,:,x,y]
        single_xy[:,:, 0] = x
        single_xy[:,:, 1] = y
        
        total_model.append(single)
        total_xy.append(single_xy)
        
    total_model = np.array(total_model)
    total_xy = np.array(total_xy)
    total_model = total_model.reshape(num_coor*batch, period)
    total_xy = total_xy.reshape(num_coor*batch, period, 2)
    
    new_Y = total_model

    return new_X, new_Y, adj_matrix_total, total_xy

def generate_samples_by_cells2(coors_matrix, X, Y):
    batch, period, len_x, len_y, features = X.shape

    coors = coors_matrix
    num_coor = len(coors)
    total_model = []
    
    adj_matrix_total = []
    
    for each in coors:
        x, y = each[0], each[1]
        window_size = 2*S + 1
        single = np.zeros((batch, period, window_size, window_size, features))
        adj_matrix = np.zeros((batch, window_size**2, window_size**2))
        
        for cell_x in range(2*S + 1):
            for cell_y in range(2*S + 1):

                temp_x, temp_y = generate_small_cell(cell_x, cell_y, x, y, S, x_len, y_len)
                adj_matrix[:, cell_x*window_size + cell_y] = adj_generator(temp_x, temp_y, cell_x, cell_y, window_size)
                
                if temp_x < 0 or temp_y < 0:
                    continue

                single[:,:,cell_x,cell_y] = X[:,:,temp_x,temp_y]
                
        total_model.append(single)
        adj_matrix_total.append(adj_matrix)
                                  
    total_model = np.array(total_model)
    adj_matrix_total = np.array(adj_matrix_total)
    total_model = total_model.reshape(num_coor*batch, period, window_size, window_size, features)
    adj_matrix_total = adj_matrix_total.reshape(num_coor*batch, window_size**2, window_size**2)
    
    new_X = total_model

    batch, period, len_x, len_y = Y.shape

    coors = coors_matrix
    num_coor = len(coors)
    total_model = []
    total_xy = []
    
    for each in coors:
        x, y = each[0], each[1]
        single = np.zeros((batch, period))
        single_xy = np.zeros((batch, period, 2))
        
        single[:,:] = Y[:,:,x,y]
        single_xy[:,:, 0] = x
        single_xy[:,:, 1] = y
        
        total_model.append(single)
        total_xy.append(single_xy)
        
    total_model = np.array(total_model)
    total_xy = np.array(total_xy)
    total_model = total_model.reshape(num_coor*batch, period)
    total_xy = total_xy.reshape(num_coor*batch, period, 2)
    
    new_Y = total_model

    return new_X, new_Y, adj_matrix_total, total_xy


# In[7]:


#S2 = 3
class NeuralNet(nn.Module):
    def __init__(self, num_temporal, num_spatial, num_spatial_tempoal, map_size, input_len):
        super(NeuralNet, self).__init__()
        self.num_temporal = num_temporal
        self.num_spatial = num_spatial
        self.num_spatial_tempoal = num_spatial_tempoal
        self.map_size = map_size
        self.input_len = input_len

        cnn_list = []
        #cnn_list2 = []
        #cnn_list3 = []

        for i in range(0,7):
            cnn_list.append(GraphConvolution(num_spatial_tempoal, num_spatial_tempoal))
            #cnn_list2.append(GraphConvolution(num_spatial_tempoal, num_spatial_tempoal))
            #cnn_list3.append(GraphConvolution(num_spatial_tempoal, num_spatial_tempoal))

        self.cnn3d = nn.ModuleList(cnn_list)
        #self.cnn3d_2 = nn.ModuleList(cnn_list2)
        #self.cnn3d_3 = nn.ModuleList(cnn_list3)

        #self.bn1 = nn.BatchNorm2d(15)
        #self.bn2 = nn.BatchNorm2d(15)
        
        #self.three_d_cnn = nn.Conv3d(in_channels=num_spatial_tempoal, out_channels=num_spatial_tempoal, kernel_size = (1, S2, S2), padding="same")
        #self.two_d_cnn = nn.Conv2d(in_channels=num_spatial, out_channels=num_spatial, kernel_size = (S2, S2), padding="same")
        self.two_d_cnn = GraphConvolution(num_spatial, num_spatial)
        #self.two_d_cnn2 = GraphConvolution(num_spatial, num_spatial)

        self.LSTM = nn.LSTM((num_temporal+num_spatial_tempoal), hidden_size = num_temporal+num_spatial_tempoal,batch_first=True)

        self.FC= nn.Linear(map_size*map_size*num_spatial_tempoal*input_len, num_spatial_tempoal*input_len)
        self.FC2= nn.Linear(map_size*map_size*num_spatial, num_spatial)
        self.FC3= nn.Linear(num_temporal+num_spatial+num_spatial_tempoal, 1)

    def forward(self, x, adj, future_seq=0):
        
        temporal_view = x[:,:,0,0,0:self.num_temporal]
        spatial_view = x[:,0,:,:,self.num_temporal:self.num_temporal+self.num_spatial]
        spatial_tempoal_view = x[:,:,:,:,self.num_temporal+self.num_spatial:self.num_temporal+self.num_spatial+self.num_spatial_tempoal]

        #spatial_tempoal_view = spatial_tempoal_view.permute(0, 4, 1, 2, 3)
        #spatial_tempoal_view = self.three_d_cnn(spatial_tempoal_view).

        for i in range(7):
            #spatial_tempoal_view[:, i] = self.cnn3d[i](spatial_tempoal_view[:, i], torch.zeros((x.shape[0], 25, 25), dtype=torch.float32, device=torch.device('cuda:0')))
            spatial_tempoal_view[:, i] = self.cnn3d[i](spatial_tempoal_view[:, i], adj)
            #spatial_tempoal_view[:, i] = F.relu(spatial_tempoal_view[:, i])
            #spatial_tempoal_view[:, i] = self.bn1(spatial_tempoal_view[:, i])
            #spatial_tempoal_view[:, i] = self.cnn3d_2[i](spatial_tempoal_view[:, i], adj)
            #spatial_tempoal_view[:, i] = F.relu(spatial_tempoal_view[:, i])
            #spatial_tempoal_view[:, i] = self.cnn3d_3[i](spatial_tempoal_view[:, i], adj)
            #spatial_tempoal_view[:, i] = F.relu(spatial_tempoal_view[:, i])
            #spatial_tempoal_view[:, i] = self.bn2(spatial_tempoal_view[:, i])
            
        spatial_tempoal_view = F.relu(spatial_tempoal_view)

        #spatial_tempoal_view = spatial_tempoal_view.permute(0, 2, 3, 4, 1)
        
        #spatial_view = spatial_view.permute(0, 3, 1, 2)
        #spatial_view = self.two_d_cnn(spatial_view, torch.zeros((x.shape[0], 25, 25), dtype=torch.float32, device=torch.device('cuda:0')))
        spatial_view = self.two_d_cnn(spatial_view, adj)
        spatial_view = F.relu(spatial_view)
        #spatial_view = self.two_d_cnn2(spatial_view, adj)
        #spatial_view = F.relu(spatial_view)

        #spatial_view = spatial_view.permute(0, 2, 3, 1)

        spatial_tempoal_view = self.FC(spatial_tempoal_view.flatten(1))
        spatial_tempoal_view = torch.reshape(spatial_tempoal_view, (len(x), self.input_len, self.num_spatial_tempoal))

        merged_two_view = torch.cat((spatial_tempoal_view, temporal_view), 2)

        current, (h_n, c_n) = self.LSTM(merged_two_view)

        merged_two_view = h_n.permute(1, 0, 2).flatten(1)
        spatial_view = self.FC2(spatial_view.flatten(1))

        final_view = torch.cat((merged_two_view, spatial_view), 1)
        final_view = self.FC3(final_view)
        final_view = F.relu(final_view)
        
        return final_view


# In[8]:


def plot_loss(train_loss_arr, valid_loss_arr):
    fig, ax1 = plt.subplots(figsize=(20, 10))

    ax1.plot(train_loss_arr, 'k', label='training loss')
    ax1.plot(valid_loss_arr, 'g', label='validation loss')
    ax1.legend(loc=1)
    ax2 = ax1.twinx()
    #ax2.plot(train_mape_arr, 'r--', label='train_mape_arr')
    #ax2.plot(v_mape_arr, 'b--', label='v_mape_arr')

    ax2.legend(loc=2)
    plt.show()
    plt.clf()


# In[9]:


def MSE_torch(prediction, true_value):
    prediction = prediction.flatten(0)
    true_value = true_value.flatten(0)

    #prediction = torch.round(prediction)

    mse = torch.sum(torch.square(prediction - true_value)/len(prediction))

    return mse

def RMSE_torch(prediction, true_value):
    prediction = prediction.flatten()
    true_value = true_value.flatten()

    #prediction = np.round(prediction)

    rmse = torch.sqrt(torch.sum(torch.square(prediction - true_value)/len(prediction)))

    return rmse

def MAE_torch(prediction, true_value):
    prediction = prediction.flatten()
    true_value = true_value.flatten()
    mae = torch.abs(prediction-true_value)
    return torch.sum(mae)/len(prediction)

def accuracy(vector_x, vector_y):
    vector_x = vector_x.flatten()
    vector_y = vector_y.flatten()
    new_v = vector_x - vector_y
    new_v = np.abs(new_v)
    new_v = np.sum(new_v)
    return 1 - new_v/len(vector_x)

def MSE_np(prediction, true_value):
    prediction = prediction.flatten()
    true_value = true_value.flatten()

    #prediction = np.round(prediction)

    mse = np.sum(np.square(prediction - true_value))/len(prediction)

    return mse

def MAE_np(prediction, true_value):
    prediction = prediction.flatten()
    true_value = true_value.flatten()
    mae = np.abs(prediction-true_value)
    return np.sum(mae)/len(prediction)

def RMSE_np(prediction, true_value):
    prediction = prediction.flatten()
    true_value = true_value.flatten()

    #prediction = np.round(prediction)

    rmse = np.sqrt(np.sum(np.square(prediction - true_value)/len(prediction)))

    return rmse
    
def precision(predicted, true_y):
    TruePositives = 0
    FalsePositives = 0
    for i in range(true_y.shape[0]):
      for t in range(true_y.shape[1]):
          true = true_y[i][t]
          pred = predicted[i][t]
          for x in range(len(true)):
              for y in range(len(true[0])):
                  if pred[x][y] == 0 and true[x][y] == 0:
                      TruePositives += 1
                  elif pred[x][y] == 0 and true[x][y] == 1:
                      FalsePositives += 1
    precision = TruePositives/(TruePositives + FalsePositives)
    return precision

def recall(predicted, true_y):
    # Recall Recall = TruePositives / (TruePositives + FalseNegatives)
    TruePositives = 0
    FalseNegatives = 0
    for i in range(true_y.shape[0]):
      for t in range(true_y.shape[1]):
          true = true_y[i][t]
          pred = predicted[i][t]
          for x in range(len(true)):
              for y in range(len(true[0])):
                  if pred[x][y] == 0 and true[x][y] == 0:
                      TruePositives += 1
                  elif pred[x][y] == 1 and true[x][y] == 0:
                      FalseNegatives += 1
    recall = TruePositives/(TruePositives + FalseNegatives)
    return recall


# In[10]:


def remove_from_arr_to_arr(a, b):
    indices = np.argwhere(np.isin(a,b))
    a = np.delete(a,indices)
    return a


def data_split(x_arrange, num_X_bag, X, Y):
    if len(x_arrange) < num_X_bag:
        return np.NaN, X[x_arrange], Y[x_arrange]

    idx = np.random.choice(x_arrange, num_X_bag, replace=False)
    x_sample, y_sample = X[idx], Y[idx]
    x_arrange = remove_from_arr_to_arr(x_arrange, idx)

    return x_arrange, x_sample, y_sample

def data_split_no_random(x_arrange, num_X_bag, X, Y):
    if len(x_arrange) < num_X_bag:
        return np.NaN, X[x_arrange], Y[x_arrange]
    
    idx = x_arrange[:num_X_bag]
    x_sample, y_sample = X[idx], Y[idx]
    x_arrange = remove_from_arr_to_arr(x_arrange, idx)

    return x_arrange, x_sample, y_sample
  


# In[11]:



#model = NeuralNet()
#model.cuda()
def save_model(model, path, rd):
    PATH = "entire_model"+str(rd) +"_"+str(path)+".pt"
    torch.save(model.state_dict(), PATH)
    
def load_model(model, path, rd):
    PATH = "entire_model"+str(rd) +"_"+str(path)+".pt"
    model.load_state_dict(torch.load(PATH, map_location="cuda:0"))
    model.to(device)  
    return model

def speak(text):
    from win32com.client import Dispatch

    speak = Dispatch("SAPI.SpVoice").Speak

    speak(text)


# In[12]:



    
def to_coor_list(lables, level):
    temp = []
    for x in range(128):
        for y in range(64):
            if lables[x][y] == level:
                temp.append([x, y])
    return np.array(temp)


def expanding(coor_list, mask, record_map):
    temp = np.zeros(mask.shape)
    for coor in coor_list:
        x, y = coor[0], coor[1]
        temp[x][y] = 1

    new_list = []
    for coor in coor_list:
        x, y = coor[0], coor[1]

        for new_c in [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]]:

            try:
                new_x = x + new_c[0]
                new_y = y + new_c[1]
                if mask[new_x][new_y] == 1 and temp[new_x][new_y] == 0 and record_map[new_x][new_y] == 1:
                    new_list.append([new_x, new_y])
            except:
                pass

    new_list = np.array(new_list)

    if len(new_list) == 0:
        return None,None,None

    full_list = np.concatenate((coor_list, new_list), axis=0)
    return new_list, coor_list, full_list


def list_to_map(coor_list, mask):
    out = np.zeros(mask.shape)
    for coor in coor_list:
        x, y = coor[0], coor[1]
        out[x][y] = 1
    return out


def expanding_map(current_map, range_map, mask):

    coor_list = to_coor_list(current_map, 1)
    new_list, coor_list, full_list = expanding(coor_list, mask, range_map)

    if new_list is None:
        return None, None

    example_map = list_to_map(new_list, mask)
    expanded_map = list_to_map(full_list, mask)
    example_map = example_map + expanded_map

    return expanded_map, example_map


def top_percentile(ratio, moran_i_map, mask):
    if np.sum(mask) == 0:
        return None
    distribution = np.where(mask == 1, moran_i_map, np.inf).flatten()
    distribution = distribution[distribution != np.inf]
    thresh = np.percentile(distribution, ratio)
    out = np.where((moran_i_map >= thresh) & (mask == 1), 1, 0)
    return out


def map_minus(range_map, expanded_map):
    out = range_map - expanded_map
    for x in out:
        for y in x:
            if y != 0 and y != 1:
                print("!empty map_minus!")
                return None
    return out

def map_plus(range_map, expanded_map):
    out = range_map + expanded_map
    out = np.where(out==0, 0, 1)
    return out

def plot_moran_i_map(moran_i_map):
    out = moran_i_map
    fig, ax = plt.subplots(figsize=(8, 4))
    ax = sns.heatmap(out.transpose(), robust=False, annot=False, cbar=True, cmap='coolwarm', norm=colors.CenteredNorm())
    ax.invert_yaxis()
    plt.axis("off")
    plt.show()


def plot_map(map, vmax=None, vmin=None):
    out = map
    fig, ax = plt.subplots(figsize=(8, 4))
    ax = sns.heatmap(out.transpose(), robust=False, annot=False, cbar=True, vmax=vmax, vmin=vmin)
    ax.invert_yaxis()
    plt.axis("off")
    plt.show()


# In[13]:


def assign_scale(num):
    if num < 150:
        return 2
    elif num <300:
        return 4
    elif num <500:
        return 8
    elif num <1000:
        return 16
    else:
        return 32

def flip_num_matrix(target):

    arr = np.unique(target)
    arr = np.sort(arr)
    out = np.zeros(target.shape)

    temp_dict = {}

    for i in range(len(arr)):
        temp_dict[arr[i]] = arr[len(arr) - i - 1]

    for x in range(target.shape[0]):
        for y in range(target.shape[1]):
            out[x][y] = temp_dict[target[x][y]]

    return out

def split_by_step(hot_and_sig_map, range_map, moran_i_map, step_size):
    
    remaining_map = map_minus(range_map, hot_and_sig_map)
    
    if np.sum(remaining_map) == 0:
        return None, None
    
    if np.sum(remaining_map) > step_size:
        distribution = np.where(remaining_map == 1, moran_i_map, np.inf).flatten()
        distribution = distribution[distribution != np.inf]
        thresh = np.sort(distribution)[::-1][step_size-1]
        expanded_map = np.where((moran_i_map >= thresh) & (remaining_map == 1), 1, 0)
    else:
        expanded_map = remaining_map     
    
    expanded_map = expanded_map + hot_and_sig_map
    return expanded_map, None
"""
def G_norm(sig_list, sig_model):
    
    pred_map = np.full(Yv.shape, 0.0)
    ctr_map = np.zeros(Yv.shape)

    pred_map, ctr_map = test_model(sig_list, sig_model, pred_map, ctr_map)

    temp_mask = ctr_map.copy()
    
    ctr_map = np.where(ctr_map==0, 1, ctr_map)
    final_pred = pred_map/ctr_map
    
    mean_map = np.zeros((Yv.shape[2], Yv.shape[3])) * 1.0
    sd_map = np.zeros((Yv.shape[2], Yv.shape[3])) * 1.0
    for x in range(Yv.shape[2]):
        for y in range(Yv.shape[3]):
            arr = []
            for t in range(Yv.shape[0]):
                arr.append(Yv[t][0][x][y])
            mean_map[x][y] = np.mean(arr)
            sd_map[x][y] = np.std(arr)
                
    z_map = np.zeros(Yv.shape) * 1.0
    for t in range(Yv.shape[0]):
        z_map[t][0] = np.nan_to_num((final_pred[t][0] - mean_map)/sd_map)

    temp_Yv = np.where(temp_mask == 0, 0, Yv)
    z_map = np.where(temp_mask == 0, 0, z_map)
        
    loss_norm = MSE_np(z_map, temp_Yv)*(Yv.shape[2]*Yv.shape[3])/len(sig_list)

    return loss_norm
"""
"""
for sca in scale:
    temp = []
    for x in range(128):
        for y in range(64):
            if lables[x][y] == sca and mask_table[x][y] == 1:
                temp.append([x, y])
    coors_matrix.append(temp)
coors_matrix = np.array(coors_matrix, dtype=object)

#coors_matrix = coor_matrix_filter(coors_matrix)

inner_iteration = np.zeros(len(scale), dtype=int)

for i in range(len(scale)):
    print(len(coors_matrix[i]))

    inner_iteration[i] = assign_scale(len(coors_matrix[i]))
    
print(inner_iteration)
"""


# In[14]:



def train_model(coors_matrix, k, transfer_model=None):

    iterations = assign_scale(len(coors_matrix))
    batch_size = 1024
    device = torch.device('cuda' if torch .cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    model = NeuralNet(5, 29, 13, S*2+1, 7).to(device)
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_arr = []
    valid_loss_arr =[]

    if transfer_model != None:
        model = load_model(model, transfer_model, k)

    all_results = {}
    best_result = math.inf
    best_ctr = 0
    best_model = model
    
    for echo in range(100):

        avg_train_loss = []
        avg_valid_loss = []
        ctr = 0
        ctrv = 0      
        x_arrange = np.arange(len(X))
        xv_arrange = np.arange(len(Xv))

        num_X_bag = int(len(X)/iterations)
        num_Xv_bag = int(len(Xv)/iterations)

        for i in range(iterations):

            if i == iterations - 1:
                num_Xv_bag = np.inf
                num_X_bag = np.inf

            x_arrange, x_sample, y_sample = data_split(x_arrange, num_X_bag, X, Y)
            xv_arrange, xv_sample, yv_sample = data_split(xv_arrange, num_Xv_bag, Xv, Yv)

            cur_X, cur_Y, cur_adj = generate_samples_by_cells(coors_matrix, x_sample, y_sample)
            cur_Xv, cur_Yv, cur_adjv = generate_samples_by_cells(coors_matrix, xv_sample, yv_sample)

            train_dataset = DM_Dataset(cur_X, cur_Y, cur_adj)
            training_generator = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            validation_dataset = DM_Dataset(cur_Xv, cur_Yv, cur_adjv)
            validation_generator = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

            model.train()

            for local_batch, local_labels, local_adj in training_generator:
                local_batch, local_labels, local_adj = local_batch.to(device), local_labels.to(device), local_adj.to(device)

                outputs = model(local_batch, local_adj, 1)

                outputs = torch.flatten(outputs)
                true_y = torch.flatten(local_labels)
                train_loss = criterion(outputs, true_y)


                avg_train_loss.append(train_loss.cpu().data*len(local_batch))
                ctr += len(local_batch)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            model.eval()

            with torch.no_grad():
                for local_batch, local_labels, local_adj in validation_generator:
                    local_batch, local_labels, local_adj = local_batch.to(device), local_labels.to(device), local_adj.to(device)

                    Voutputs = model(local_batch, local_adj, 1)
                    Voutputs = torch.flatten(Voutputs)

                    V_true_y = torch.flatten(local_labels)
                    v_loss = criterion(Voutputs, V_true_y)

                    avg_valid_loss.append(v_loss.cpu().data*len(local_batch))
                    ctrv += len(local_batch)
        train_loss_arr.append(sum(avg_train_loss) / ctr)
        valid_loss_arr.append(sum(avg_valid_loss) / ctrv)

        if best_result <= float(valid_loss_arr[-1].item()):
            best_ctr += 1
        else:
            best_ctr = 0
            # save_model(model, out_path, k)
            best_model = model
            #print("epochs: " + str(echo))
            #print(float(valid_loss_arr[-1].item()))

        #print("echo: " + str(echo))
        #print("train_loss: "+str(train_loss_arr[-1].item())  + "||" + "v_loss: " + str(valid_loss_arr[-1].item()))

        best_result = min(best_result, valid_loss_arr[-1].item())

        if best_ctr > 5:
           #print("early stop")
           #plot_loss(train_loss_arr, valid_loss_arr)
           #print("best mse: " + str(best_result))
           break

        #if best_result < best_arr[num]:
        #    break
    return best_result, best_model


# In[15]:


def test_model(coors_matrix, origin_list, model):
    
    aaaa = {}
    for x, y in origin_list:
        aaaa[int(x)] = {}
    for x, y in origin_list:
        aaaa[int(x)][int(y)] = 1
    
    iterations = assign_scale(len(coors_matrix))
    batch_size = 1024
    device = torch.device('cuda' if torch .cuda.is_available() else 'cpu')
    criterion = nn.MSELoss(size_average=False, reduce=False, reduction=False)
    # model = EncoderDecoderConvLSTM(32, 47).to(device)

    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    x_arrange = np.arange(len(X))

    valid_loss_arr =[]
    ctr = 0
    avg_valid_loss = []

    y_out_total = []
    x_out_total = []
    xy_out_total = []
    
    loss_total = []
    
    x_arrange = np.arange(len(X))
    num_X_bag = int(len(X)/iterations)

    for i in range(iterations):

        if i == iterations - 1:
            num_Xv_bag = np.inf

        x_arrange, x_sample, y_sample = data_split(x_arrange, num_X_bag, X, Y)
        cur_X, cur_Y, cur_adj, total_xy = generate_samples_by_cells2(coors_matrix, x_sample, y_sample)

        train_dataset = DM_Dataset_test(cur_X, cur_Y, cur_adj, total_xy)
        training_generator = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        model.eval()

        with torch.no_grad():
            for local_batch, local_labels, local_adj, total_xy in training_generator:
                local_batch, local_labels, local_adj = local_batch.to(device), local_labels.to(device), local_adj.to(device)

                Voutputs = model(local_batch, local_adj)
                Voutputs = torch.flatten(Voutputs)

                y_out_total.append(Voutputs.cpu().detach().numpy())
                x_out_total.append(local_batch[:, :, :, :, :5].cpu().detach().numpy())
                xy_out_total.append(total_xy)
                
                V_true_y = torch.flatten(local_labels)
                v_loss = criterion(Voutputs, V_true_y)
                ctr += len(local_batch)
                loss_total.append(v_loss)
                valid_loss_arr.append(sum(avg_valid_loss) / ctr)
    #print("layer: " + str(path))
    #print("test loss: " + str(valid_loss_arr[-1].item()))

    
    loss_total = torch.cat(loss_total)
    xy_out_total = torch.cat(xy_out_total).squeeze()
    
    print(loss_total.shape)
    print(xy_out_total.shape)
    
    tmp = {}
    cctr = {}
    
    for ls, coor in zip(loss_total, xy_out_total):
        
        if tmp.get(coor) is None:
            tmp[coor] = 1
            cctr[coor] = 1
        else:
            tmp[coor] += ls
            cctr[coor] += 1

    key_list = []
    avg_loss_list = []
        
    for key in tmp.keys():
        key_list.append(key[None, :])
        avg_loss_list.append(tmp[key]/cctr[key])
    
    key_list = torch.cat(key_list, dim=0)
    avg_loss_list = torch.tensor(avg_loss_list)
    
    key_list = key_list.cpu().detach().numpy()
    avg_loss_list = avg_loss_list.cpu().detach().numpy()
    
    top_k = 35
    idx  = np.argsort(avg_loss_list)[:top_k]#[::-1][:top_k]

    key_list = key_list[idx]
    
    # print(key_list.shape) # (1, 2)
    # print(key_list) # [[122.  30.]]
    
    target_x, target_y = key_list[0]
    target_x, target_y = int(target_x), int(target_y)
    
    print("target")
    print(target_x, target_y)
    new_expandede = [[target_x, target_y]]
    
    new_x, new_y = target_x - 1, target_y - 1

    for x in range(3):
        for y in range(3):
            try:
                if mask[new_x+x][new_y+y] == 1 and aaaa[int(x)][int(y)] != 1:
                    new_expandede.append([x, y])
            except:
                print("reach bound")

    return new_expandede

def convert_to_pred(y_out_total, x_out_total, xy_out_total, pred_map, ctr_map):
    
    max_vecotr = [2018, 12, 31, 7, 1]
    min_vector = [2016, 1,  1,  1, 0]
    max_vecotr = np.array(max_vecotr)
    min_vector = np.array(min_vector)
    minus_vector = max_vecotr-min_vector

    final_dict = {}

    for j in range(len(y_out_total)):
        temp_y = y_out_total[j]
        temp_x = x_out_total[j]
        temp_xy = xy_out_total[j]

        for fk in range(len(temp_y)):

            calendar = temp_x[fk, 0, 3, 3, :5]
            
            calendar = calendar*(minus_vector) + min_vector
            calendar = np.rint(calendar)
            year = int(calendar[0])
            month = int(calendar[1])
            day = int(calendar[2])

            key = str(year) + "-" + str(month) + "-" + str(day) +"-" + str(int(temp_xy[fk][0][0])) + "-"+str(int(temp_xy[fk][0][1]))
            

            if final_dict.get(key) is not None:
                print("dup detect!")
                print(key)

            final_dict[key] = temp_y[fk]
            
    for t in range(len(Xv)):
        calendar = Xv[t][0][0][0][:5]
        calendar = calendar*(minus_vector) + min_vector

        year = int(calendar[0])
        month = int(calendar[1])
        day = int(calendar[2])    


        for x in range(x_len):
            for y in range(y_len):

                key = str(year) + "-" + str(month) + "-" + str(day) +"-" + str(x) + "-"+str(y)
                if final_dict.get(key) is not None:
                    pred_map[t][0][x][y] += final_dict[key]
                    ctr_map[t][0][x][y] += 1
    
    return pred_map, ctr_map


# In[16]:


def find_least_lost_spot(expanded_map, remaining_map, sig_model):
    
    print(np.sum(expanded_map))
    print(np.sum(remaining_map))

    remaining_list = to_coor_list(remaining_map, 1)
    origin_map = map_minus(expanded_map, remaining_map)
    origin_list = to_coor_list(origin_map, 1)

    remaining_list = test_model(remaining_list, origin_list, sig_model) 
    print(len(remaining_list))
    expanded_list = []
    
    for x, y in remaining_list:
        expanded_list.append([int(x), int(y)])
    print(len(expanded_list))

    for x, y in origin_list:
        expanded_list.append([int(x), int(y)])
    print(len(expanded_list))
    # print(expanded_list)
    
    bbb = []

    for x, y in remaining_list:
        bbb.append([int(x), int(y)])    
    
    return bbb, expanded_list
    

def expanding_map_manhattan(hot_and_sig_map, range_map, mask, radius):
    x_len = mask.shape[0]
    y_len = mask.shape[1]
    
    expanded_map = np.zeros(mask.shape)
    
    for x1 in range(x_len):
        for y1 in range(y_len):
            if mask[x1][y1] != 1 or hot_and_sig_map[x1][y1] != 1:
                continue
            
            for x2 in range(x_len):
                for y2 in range(y_len):
                    
                    if mask[x2][y2] != 1 or range_map[x2][y2] != 1:
                        continue
                    if abs(x1 - x2) + abs(y1 - y2) <= radius:
                    #if ((x1 - x2)**2 + (y1 - y2)**2)**0.5 <= radius:
                        expanded_map[x2][y2] = 1
    
    example_map =  expanded_map + hot_and_sig_map
    remaining_map = map_minus(expanded_map, hot_and_sig_map)
    if np.sum(remaining_map) == 0:
        return None, None
    
    return expanded_map, example_map

for model_num in range(12, 14, 1):
    print("================= model num:" + str(model_num) + "=====================")

    X = np.load(open('X_Iowa.npy', 'rb'))
    Y = np.load(open('Y_Iowa.npy', 'rb'))
    X, Xv, Y, Yv = train_test_split(X, Y, test_size=0.20, random_state=0)

    acc_matrix = np.zeros((Y.shape[2], Y.shape[3]))
    for b in range(len(Y)):
        acc_matrix += Y[b][0]
    
    moran_i_mask = np.load(open('mask_128_64.npy', 'rb'))
    
    max_iteration = 100
    moran_i = 1
    ctr = 0
    ratio = 75
    radius= 4
    
    partition_map = np.ones(moran_i_mask.shape)*(-1)
    
    
    #while moran_i > 0.01:
    while not (np.isnan(moran_i)):
        print("iteration: " + str(ctr))
        LISA_obj = LISA.LISAMatrix(contiguity="queen", alpha=0.05)
        
        moran_i_map, quadrant_map, moran_i_list, moran_i, hot_and_sig_map = LISA_obj.fit(acc_matrix, moran_i_mask)
        print("moran_i: " + str(moran_i))
        LISA_obj.plot_quadrant_map()
        LISA_obj.plot_sig_and_hot()
        LISA_obj.scatter_plot_moran_i()

        range_map = np.where((quadrant_map == 2)|(quadrant_map == 1), 1, 0)
        transfer_model = None if ctr == 0 else str(ctr - 1)
        
        rest_map = map_minus(range_map, hot_and_sig_map)
        step_size = np.sum(top_percentile(ratio, moran_i_map, rest_map))
        sig_list = to_coor_list(hot_and_sig_map, 1)
            
        if len(sig_list) != 0:
            sig_loss, sig_model = train_model(sig_list, model_num, transfer_model)
            save_model(sig_model, str(ctr), model_num)
            
            #sig_loss_norm = G_norm(sig_list, sig_model)

        else:
            sig_loss = np.inf
            #sig_loss_norm = np.inf
            sig_model = None
            
        tolerence = 0
        best_model = sig_model
        best_HH = hot_and_sig_map

        for i in range(max_iteration):
            print(str(i) + "th expanded")
            
            # expanded_map, example_map = expanding_map(hot_and_sig_map, range_map, mask)
            # expanded_map, example_map = split_by_step(hot_and_sig_map, range_map, moran_i_map, step_size)
            expanded_map, example_map = expanding_map_manhattan(hot_and_sig_map, range_map, mask, radius=radius)

            if expanded_map is None:
                print("expand reach bound")
                break
            
            current_sig_hot = expanded_map
            current_example_map = example_map

            while np.sum(expanded_map) - np.sum(hot_and_sig_map) < 30:
                print("#############")
                expanded_map, example_map = expanding_map_manhattan(current_sig_hot, range_map, mask, radius=radius)
                if expanded_map is None:
                    break
                else:
                    current_sig_hot = expanded_map
                    current_example_map = example_map
                    
            if expanded_map is None:
                expanded_map = current_sig_hot
                example_map = current_example_map
                            
            remaining_map = map_minus(expanded_map, hot_and_sig_map)
        
            remaining_list, expanded_list = find_least_lost_spot(expanded_map, remaining_map, sig_model)
        
            expanded_loss, exp_model = train_model(expanded_list, model_num, transfer_model)
            
            #exp_loss_norm = G_norm(expanded_list, exp_model)
            expanded_map = list_to_map(expanded_list, mask)
            
            remaining_map = list_to_map(remaining_list, mask)
            
            example_map =  expanded_map + hot_and_sig_map
            # plot_map(example_map)
            
            remaining_loss, remaining_model = train_model(remaining_list, model_num, transfer_model)
            

            
            #remaining_loss_norm = G_norm(remaining_list, remaining_model)
            
            pecent = (len(expanded_list)*expanded_loss - len(remaining_list)*remaining_loss - len(sig_list)*sig_loss)/(len(remaining_list)*remaining_loss + len(sig_list)*sig_loss)
            print("percent： " + str(pecent))
            
            if len(expanded_list)*expanded_loss > len(remaining_list)*remaining_loss + len(sig_list)*sig_loss:
                
            #if len(expanded_list)*exp_loss_norm > len(remaining_list)*remaining_loss_norm + len(sig_list)*sig_loss_norm:
                tolerence += 1
                print("tolerence + 1")
                
            else:
                tolerence = 0
                best_model = sig_model
                best_HH = hot_and_sig_map
                
            if tolerence == 2:
                print("!BROKEN!: expanded > candidate + sig HH")

                print("-------------------------------------------")
                print("expanded: " + str(expanded_loss))
                print("candidate: " + str(remaining_loss))
                print("sig_HH: " + str(sig_loss))
                sig_model = best_model
                hot_and_sig_map = best_HH
                break
                
            print("===========================================")
            print("expanded <= candidate + sig HH")

            print("-------------------------------------------")
            print("expanded: " + str(expanded_loss))
            print("candidate: " + str(remaining_loss))
            print("sig_HH: " + str(sig_loss))
            
            hot_and_sig_map, sig_list, sig_loss, sig_model = expanded_map, expanded_list, expanded_loss, exp_model
            #sig_loss_norm = exp_loss_norm
            #hot_and_sig_map = expanded_map
            save_model(exp_model, str(ctr), model_num)
            
        if np.sum(hot_and_sig_map) != 0:
            save_model(sig_model, str(ctr), model_num)
            moran_i_mask = np.where(hot_and_sig_map == 1, 0, moran_i_mask)
            partition_map = np.where(hot_and_sig_map == 1, ctr, partition_map)
            ctr += 1
        else:
            print("sig area left is zero!")
            break
        # -----------------------------------------------------------------------------

        """
        hot_and_sig_map = np.where((hot_and_sig_map!=1)&(range_map==1), 1, 0)
        
        sig_list = to_coor_list(hot_and_sig_map, 1)
        if len(sig_list) != 0:
            sig_loss, sig_model = train_model(sig_list, model_num, int(ctr-1))
            save_model(sig_model, str(ctr), model_num)

            moran_i_mask = np.where(hot_and_sig_map == 1, 0, moran_i_mask)
            partition_map = np.where(hot_and_sig_map == 1, ctr, partition_map)
            ctr += 1  
        """

        
    coors_matrix = []
    for x in range(128):
        for y in range(64):
            if hot_and_sig_map[x][y] != 1 and moran_i_mask[x][y] == 1:
                coors_matrix.append([x, y])
    coors_matrix = np.array(coors_matrix)
    
    if len(coors_matrix) != 0:
        rest_loss, remaining_model = train_model(coors_matrix, model_num, str(ctr-1))
        save_model(remaining_model, str(ctr), model_num)

        partition_map = np.where((hot_and_sig_map) != 1 & (moran_i_mask == 1), ctr, partition_map)
        test = np.where((hot_and_sig_map) != 1 & (moran_i_mask == 1), 1, 0)
        plot_map(test)
    
    partition_map = np.where(partition_map == -1, partition_map.max()+1, partition_map)
    partition_map = flip_num_matrix(partition_map)

    plot_map(partition_map)
    np.save(str(model_num)+"_map", partition_map)


# In[ ]:





# In[ ]:




