from torch.utils.data import TensorDataset, DataLoader
from data_generators import load_dataset
from utils.graph_utils import init_features, graphs_to_tensor
import torch
import random

def graphs_to_dataloader(config, graph_list):

    adjs_tensor = graphs_to_tensor(graph_list, config.data.max_node_num)
    x_tensor = init_features(config.data.init, adjs_tensor, config.data.max_feat_num)

    # la, u = torch.symeig(adjs_tensor, eigenvectors=True)
    # la = torch.diag_embed(la)

    train_ds = TensorDataset(x_tensor, adjs_tensor)
    train_dl = DataLoader(train_ds, batch_size=config.data.batch_size, shuffle=True)
    return train_dl


def graphs_to_dataloader2(config, graph_list):
    # get the adjacency matrix
    adjs_tensor = graphs_to_tensor(graph_list, config.data.max_node_num)
    x_tensor = init_features(config.data.init, adjs_tensor, config.data.max_feat_num)

    # get the eigenvectors and eigenvalues
    # la, u = torch.symeig(adjs_tensor, eigenvectors=True)
    la, u = torch.linalg.eigh(adjs_tensor)
    # la = torch.diag_embed(la)

    train_ds = TensorDataset(x_tensor, adjs_tensor, u, la)
    train_dl = DataLoader(train_ds, batch_size=config.data.batch_size, shuffle=True)
    return train_dl


def dataloader(config, get_graph_list=False):
    graph_list = load_dataset(data_dir=config.data.dir, file_name=config.data.data)
    print("len graph_list:",len(graph_list))

    # This line can be deleted to fix train and test set
    # random.shuffle(graph_list)

    test_size = int(config.data.test_split * len(graph_list))

    train_graph_list, test_graph_list = graph_list[test_size:], graph_list[:test_size]
    if get_graph_list:
        return train_graph_list, test_graph_list

    return graphs_to_dataloader(config, train_graph_list), graphs_to_dataloader(config, test_graph_list)


def dataloader2(config, get_graph_list=False):
    graph_list = load_dataset(data_dir=config.data.dir, file_name=config.data.data)
    test_size = int(config.data.test_split * len(graph_list))
    train_graph_list, test_graph_list = graph_list[test_size:], graph_list[:test_size]
    if get_graph_list:
        return train_graph_list, test_graph_list

    return graphs_to_dataloader2(config, train_graph_list), graphs_to_dataloader(config, test_graph_list)