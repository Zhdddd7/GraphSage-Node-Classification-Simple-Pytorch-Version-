from datasets import *
from model import Classifier
config = {}
dataset = "cora"
ds = DataCenter(config)
ds.load_dataSet(dataset)
print(type(ds))
train_nodes = getattr(ds,dataset +'_train')
print(train_nodes.shape)
print(type(train_nodes))
input_dim = ds.feat_data.shape[1]
hidden_dim = 128
output_dim = 7
num_neighbors = 10
epochs = 300
learning_rate = 0.01
save_path = './models'
num_layers = 2
# model_test = GraphSage(num_layers, input_dim, hidden_dim)
from utils import *
node_sampling = mul_sampling(list(train_nodes), [10, 25], ds.adj_lists)
print("the len of node_sampling is", len(node_sampling))
print("the sample of node_sampling is",len(node_sampling[2]))
# res = model_test()

feat_array = ds.feat_data
# feat_array(ndarray): [2708, 1433]
print(type(feat_array))
print(feat_array.shape)
print(ds.labels.shape)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
node_feature = [torch.from_numpy(feat_array[idx]).float().to(device) for idx in node_sampling]
# node_feature_list(list): [3, x, 1433]
train_nodes = getattr(ds,dataset +'_train')
print(type(train_nodes))
print(train_nodes.shape)

"""
training part
"""
from train import train
# model_test = Classifier(2, input_dim, hidden_dim, output_dim)
# train(model_test,"cora",  ds, epochs, learning_rate, save_path)

"""
evaluating part
"""
from train import evaluate
model_file = "./models/model_epoch_250.pt"
state_dict= torch.load(model_file)
model_eval = Classifier(2, input_dim, hidden_dim, output_dim)
model_eval.load_state_dict(state_dict)
evaluate(model_eval, 'cora', ds)