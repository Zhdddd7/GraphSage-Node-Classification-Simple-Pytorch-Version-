import torch
import torch.nn.functional as F
from utils import mul_sampling


def train(model, dataset_name,ds, epochs, learning_rate, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("training on:", device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
    feat_array = ds.feat_data
    train_nodes = getattr(ds,dataset_name +'_train')
    train_node_sampling = mul_sampling(list(train_nodes), [10, 25], ds.adj_lists)
    train_node_features = [torch.from_numpy(feat_array[idx]).float().to(device) for idx in train_node_sampling]
    
    val_nodes = getattr(ds,dataset_name +'_val')
    val_node_sampling = mul_sampling(list(val_nodes), [10, 25], ds.adj_lists)
    val_node_features = [torch.from_numpy(feat_array[idx]).float().to(device) for idx in val_node_sampling]

    labels = torch.LongTensor(ds.labels).to(device)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_node_features)
        loss = F.cross_entropy(output, labels[train_nodes])
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

        if epoch % 50 ==0:
            model.eval()
            with torch.no_grad():
                val_output = model(val_node_features)
                val_loss = F.cross_entropy(val_output, labels[val_nodes])
                val_acc = (val_output.argmax(dim = 1) == labels[val_nodes]).float().mean()
                print(f"the validation Loss: {val_loss.item()}, the Accuracy: {val_acc}")
            
            model_file = save_path + f"/model_epoch_{epoch}.pt"
            torch.save(model.state_dict(), model_file)
            print("model saved!")

def evaluate(model, dataset_name, ds):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"evaluate on {device}")
    model.to(device)

    feat_array = ds.feat_data
    test_nodes = getattr(ds, dataset_name +"_test")
    test_node_sampling = mul_sampling(list(test_nodes), [10, 25], ds.adj_lists)
    test_node_features = [torch.from_numpy(feat_array[idx]).float().to(device) for idx in test_node_sampling]
    labels = torch.LongTensor(ds.labels).to(device)

    model.eval()
    with torch.no_grad():
        test_output = model(test_node_features)
        test_loss = F.cross_entropy(test_output, labels[test_nodes])
        test_acc = (test_output.argmax(dim = 1) == labels[test_nodes]).float().mean()
        print(f'Test Loss: {test_loss.item()}, Accuracy: {test_acc.item()}')