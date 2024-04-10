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
import LISA
from matplotlib import colors
from convlstm import *

torch.cuda.get_device_name()


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.manual_seed(0)
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
    def __init__(self, X_input, Y_input, adj_input, total_xy):
        self.X_input = Variable(torch.Tensor(X_input).float())
        self.Y_input = Variable(torch.Tensor(Y_input).float())
        self.adj_input = Variable(torch.Tensor(adj_input).float())
        self.total_xy = Variable(torch.Tensor(total_xy).float())
        
    def __len__(self):
        return len(self.X_input)

    def __getitem__(self, idx):
        return self.X_input[idx], self.Y_input[idx], self.adj_input[idx], self.total_xy[idx]

file_to_read = open("adj_dict_pc_weekly.json", "rb")
adj_dict = pickle.load(file_to_read)


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

"""
class NeuralNet(nn.Module):
    def __init__(self, num_temporal, num_spatial, num_spatial_tempoal, map_size, input_len):
        super(NeuralNet, self).__init__()
        self.num_temporal = num_temporal
        self.num_spatial = num_spatial
        self.num_spatial_tempoal = num_spatial_tempoal
        self.map_size = map_size
        self.input_len = input_len

        self.LSTM = nn.LSTM(2303, hidden_size = 64, batch_first=True)

        self.FC3= nn.Linear(64, 1)

    def forward(self, x, adj, future_seq=0):
        
        x = x.flatten(2)
        current, (h_n, c_n) = self.LSTM(x)
        h_n = h_n.permute(1, 0, 2).flatten(1)
        x = self.FC3(h_n)
        
        return x
"""


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
learning_rate = 0.0001
    
def load_model(model, path, rd):
    PATH = "entire_model"+str(rd) +"_"+str(path)+".pt"
    model.load_state_dict(torch.load(PATH, map_location="cuda:0"))
    model.to(device)  
    return model

def speak(text):
    from win32com.client import Dispatch

    speak = Dispatch("SAPI.SpVoice").Speak

    speak(text)


# In[ ]:





# In[12]:


def assign_scale(num):
    if num < 150:
        return 1
    elif num <300:
        return 2
    elif num <500:
        return 4
    elif num <1000:
        return 8
    else:
        return 16


# In[13]:



def aa(y_out_total, x_out_total, xy_out_total, Xv, Yv):

    max_vecotr = [2018, 12, 31, 7, 1]
    min_vector = [2016, 1,  1,  1, 0]
    max_vecotr = np.array(max_vecotr)
    min_vector = np.array(min_vector)
    minus_vector = max_vecotr-min_vector

    final_dict = {}

    for i in range(len(inner_iteration)-1, -1, -1):

        for j in range(len(y_out_total[i])):
            temp_y = y_out_total[i][j]
            temp_x = x_out_total[i][j]
            temp_xy = xy_out_total[i][j]

            for fk in range(len(temp_y)):

                calendar = temp_x[fk, 0, 0, 0, :5]
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


    pred_Y = np.full((Yv.shape), -1.0)

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
                    pred_Y[t][0][x][y] = final_dict[key]
        
    return pred_Y


# In[14]:


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
    


# In[15]:


def test_model(coors_matrix, path, pred_map, ctr_map, model_num):
    
    iterations = assign_scale(len(coors_matrix))
    batch_size = 1024
    device = torch.device('cuda' if torch .cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    model = NeuralNet(5, 29, 13, S*2+1, 7).to(device)
    # model = EncoderDecoderConvLSTM(32, 47).to(device)

    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model = load_model(model, path, model_num) #!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    num_Xv_bag = int(len(Xv)/iterations)
    xv_arrange = np.arange(len(Xv))

    valid_loss_arr =[]
    ctr = 0
    avg_valid_loss = []

    y_out_total = []
    x_out_total = []
    xy_out_total = []
    
    for i in range(iterations):

        if i == iterations - 1:
            num_Xv_bag = np.inf

        xv_arrange, xv_sample, yv_sample = data_split_no_random(xv_arrange, num_Xv_bag, Xv, Yv)
        cur_Xv, cur_Yv, cur_adjv, total_xy = generate_samples_by_cells(coors_matrix, xv_sample, yv_sample)

        validation_dataset = DM_Dataset(cur_Xv, cur_Yv, cur_adjv, total_xy)
        validation_generator = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

        model.eval()

        with torch.no_grad():
            for local_batch, local_labels, local_adj, total_xy in validation_generator:
                local_batch, local_labels, local_adj = local_batch.to(device), local_labels.to(device), local_adj.to(device)

                Voutputs = model(local_batch, local_adj)
                Voutputs = torch.flatten(Voutputs)

                y_out_total.append(Voutputs.cpu().detach().numpy())
                x_out_total.append(local_batch[:, :, :, :, :5].cpu().detach().numpy())
                xy_out_total.append(total_xy.cpu().detach().numpy())

                V_true_y = torch.flatten(local_labels)
                v_loss = criterion(Voutputs, V_true_y)

                avg_valid_loss.append(v_loss.cpu().data*len(local_batch))
                ctr += len(local_batch)

                valid_loss_arr.append(sum(avg_valid_loss) / ctr)
    #print("layer: " + str(path))
    #print("test loss: " + str(valid_loss_arr[-1].item()))

    return convert_to_pred(y_out_total, x_out_total, xy_out_total, pred_map, ctr_map)
    
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

def plot_map(out):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax = sns.heatmap(out.transpose(), robust=False, annot=False, cbar=True)
    ax.invert_yaxis()
    plt.axis("off")
    plt.show()
    


# In[16]:



# In[17]:



def plot_map_11(out):
    fig, ax = plt.subplots(figsize=(10, 5))
    # hmap = colors.ListedColormap(['black', 'navy', 'blue', 'deepskyblue', 'azure', 'seashell', 'lightsalmon', 'salmon', 'red', 'maroon'])
    # hmap = colors.ListedColormap(['black', '#470c0c', 'maroon', 'firebrick', 'brown', 'darkred', 'red', 'darkorange', 'orange', 'white'])
    hmap = colors.ListedColormap(['white', "sienna", 'darkred', 'darkgreen', 'navy', 'skyblue', 'navy', 'purple', 'darkorange', "red"])
    ax = sns.heatmap(out.transpose(), robust=False, annot=False, cbar=True, cmap=hmap) # "gist_heat" "seismic"
    cbar = ax.collections[0].colorbar
    #cbar.set_ticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax.invert_yaxis()
    plt.axis("off")
    plt.show()
    
#plot_map_11(partition_map)


# In[20]:



Xv = np.load(open('X_Iowa.npy', 'rb'))
Yv = np.load(open('Y_Iowa.npy', 'rb'))
mask = np.load(open('mask_128_64.npy', 'rb'))

Xv = Xv
Yv = Yv
mask = mask

x_len = Xv.shape[2]
y_len = Xv.shape[3]
    
out = []

for k in range(10, 12, 1):
    
    try:
        partition_map = np.load(str(k) + "_map.npy")
    except:
        continue
        
    pred_map = np.full(Yv.shape, 0.0)
    ctr_map = np.zeros(Yv.shape)
        
    plot_map(partition_map)
    
    partition_map = flip_num_matrix(partition_map)

    for i in np.unique(partition_map):

        if i == partition_map.max():
            continue

        coors_matrix = []
        for x in range(x_len):
            for y in range(y_len):
                if partition_map[x][y] == i:
                    coors_matrix.append([x, y])
        coors_matrix = np.array(coors_matrix)
        pred_map, ctr_map = test_model(coors_matrix, str(int(i)), pred_map, ctr_map, k)

    ctr_map = np.where(ctr_map==0, 1, ctr_map)
    final_pred = pred_map/ctr_map
    Yv = np.where(mask==1, Yv, 0.0)
    
    print("MSE: " + str(MSE_np(final_pred, Yv)))

