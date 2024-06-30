from dgl.data import FraudYelpDataset, FraudAmazonDataset
from dgl.data.utils import load_graphs, save_graphs
import dgl
import numpy as np
import torch
import pickle
import random
from torch_scatter import scatter_add
from collections import Counter
from tqdm import tqdm

def feature_noise(features, missing_rate):
    mask = torch.rand(size=features.size())
    mask = mask <= missing_rate
    return mask

def apply_feature_noise(features, mask):
    noise_matrix = torch.FloatTensor(np.random.normal(0, 0.1, size=[features.shape[0], features.shape[1]])).to(features.device)
    features = torch.where(mask==True,features+noise_matrix,features)
    return features


def feature_mask(features, missing_rate):
    mask = torch.rand(size=features.size())
    mask = mask <= missing_rate
    return mask
def apply_feature_mask(features, mask):
    features[mask] = float('nan')
    return features


def Dataset (name='tfinance'):

        # if name == 'tfinance':
        #     graph, label_dict = load_graphs('dataset/tfinance')
        #     graph = graph[0]
        #     graph.ndata['label'] = graph.ndata['label'].argmax(1)
        #
        #     if anomaly_std:
        #         graph, label_dict = load_graphs('dataset/tfinance')
        #         graph = graph[0]
        #         feat = graph.ndata['feature'].numpy()
        #         anomaly_id = graph.ndata['label'][:,1].nonzero().squeeze(1)
        #         feat = (feat-np.average(feat,0)) / np.std(feat,0)
        #         feat[anomaly_id] = anomaly_std * feat[anomaly_id]
        #         graph.ndata['feature'] = torch.tensor(feat)
        #         graph.ndata['label'] = graph.ndata['label'].argmax(1)
        #
        #     if anomaly_alpha:
        #         graph, label_dict = load_graphs('dataset/tfinance')
        #         graph = graph[0]
        #         feat = graph.ndata['feature'].numpy()
        #         anomaly_id = list(graph.ndata['label'][:, 1].nonzero().squeeze(1))
        #         normal_id = list(graph.ndata['label'][:, 0].nonzero().squeeze(1))
        #         label = graph.ndata['label'].argmax(1)
        #         diff = anomaly_alpha * len(label) - len(anomaly_id)
        #         import random
        #         new_id = random.sample(normal_id, int(diff))
        #         # new_id = random.sample(anomaly_id, int(diff))
        #         for idx in new_id:
        #             aid = random.choice(anomaly_id)
        #             # aid = random.choice(normal_id)
        #             feat[idx] = feat[aid]
        #             label[idx] = 1  # 0
        #
        # elif name == 'tsocial':
        #     graph, label_dict = load_graphs('dataset/tsocial')
        #     graph = graph[0]
        #
        # elif name=='elliptic':
        #     dataset = pickle.load(open('dataset/{}.dat'.format(name), 'rb'))
        #     graph = dgl.graph((dataset.edge_index[0], dataset.edge_index[1]))
        #     graph.ndata['feature'] = dataset.x
        #     graph.ndata['label'] = dataset.y
        #     graph.ndata['train_mask'] = dataset.train_mask
        #     graph.ndata['val_mask'] = dataset.val_mask
        #     graph.ndata['test_mask'] = dataset.test_mask
        #
        #     noise = feature_noise(graph.ndata['feature'],0.8)
        #     features = apply_feature_noise(graph.ndata['feature'], noise)
        #
        #     mask = feature_mask(features, 0.8)
        #     apply_feature_mask(features, mask)
        #     graph.ndata['feature'] = features
        #     import random
        #     edge_index = random.choices([i for i in range(graph.num_edges())], k=int(graph.num_edges() * 0.8))
        #     edge_remove = graph.edge_ids(graph.edges()[1][edge_index].numpy(), graph.edges()[0][edge_index].numpy())
        #
        #     edge_index = edge_index + edge_remove.tolist()
        #     graph.remove_edges(edge_index)
        #     graph = dgl.add_self_loop(graph)

        # elif name == 'yelp':
        #     dataset = FraudYelpDataset()
        #     graph = dataset[0]
        #     if homo:
        #         graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        #         graph = dgl.add_self_loop(graph)
        # elif name == 'amazon':
        #     dataset = FraudAmazonDataset()
        #     graph = dataset[0]
        #     if homo:
        #         graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        #         graph = dgl.add_self_loop(graph)
    if name=='Tel-max':
        max_sms_feature_user = np.load("dataset/Tele-max/max_sms_feature_user.npy")
        max_voc_feature_user = np.load("dataset/Tele-max/max_voc_feature_user.npy")
        max_user_personal_feature = np.load("dataset/Tele-max/max_user_personal_feature.npy")
        max_user_label = np.load("dataset/Tele-max/max_user_label.npy")
        max_voc_voc_graph = np.load("dataset/Tele-max/max_voc_voc_graph.npy")
        max_sms_sms_graph = np.load("dataset/Tele-max/max_sms_sms_graph.npy")
        max_voc_sms_graph = np.load("dataset/Tele-max/max_voc_sms_graph.npy")

        max_sms_feature_user=normalize(max_sms_feature_user)
        max_voc_feature_user = normalize(max_voc_feature_user)

        max_user_personal_feature=normalize(max_user_personal_feature)

        max_voc_voc_graph=dgl.graph((torch.tensor(max_voc_voc_graph[:,0]),torch.tensor(max_voc_voc_graph[:,1])),num_nodes=6106)
        max_sms_sms_graph = dgl.graph((torch.tensor(max_sms_sms_graph[:, 0]), torch.tensor(max_sms_sms_graph[:, 1])),num_nodes=6106)
        max_voc_sms_graph = dgl.graph((torch.tensor(max_voc_sms_graph[:, 0]), torch.tensor(max_voc_sms_graph[:, 1])),num_nodes=6106)

        max_voc_voc_graph = dgl.add_self_loop(max_voc_voc_graph)
        max_sms_sms_graph = dgl.add_self_loop(max_sms_sms_graph)
        max_voc_sms_graph = dgl.add_self_loop(max_voc_sms_graph)

        graph=[]
        graph.append(max_voc_voc_graph)
        graph.append(max_sms_sms_graph)
        graph.append(max_voc_sms_graph)
        return graph, max_voc_feature_user, max_sms_feature_user, max_user_label, max_user_personal_feature

    if name == 'Tel-min':
        min_sms_feature_user = np.load("dataset/Tele-min/min_sms_feature_user.npy")
        min_voc_feature_user = np.load("dataset/Tele-min/min_voc_feature_user.npy")
        min_user_personal_feature = np.load("dataset/Tele-min/min_user_personal_feature.npy")
        min_user_label = np.load("dataset/Tele-min/min_user_label.npy")
        min_voc_voc_graph = np.load("dataset/Tele-min/min_voc_voc_graph.npy")
        min_sms_sms_graph = np.load("dataset/Tele-min/min_sms_sms_graph.npy")
        min_voc_sms_graph = np.load("dataset/Tele-min/min_voc_sms_graph.npy")

        min_sms_feature_user = normalize(min_sms_feature_user)
        min_voc_feature_user = normalize(min_voc_feature_user)

        min_user_personal_feature = normalize(min_user_personal_feature)

        min_voc_voc_graph = dgl.graph(
            (torch.tensor(min_voc_voc_graph[:, 0]), torch.tensor(min_voc_voc_graph[:, 1])), num_nodes=2045)
        min_sms_sms_graph = dgl.graph(
            (torch.tensor(min_sms_sms_graph[:, 0]), torch.tensor(min_sms_sms_graph[:, 1])), num_nodes=2045)
        min_voc_sms_graph = dgl.graph(
            (torch.tensor(min_voc_sms_graph[:, 0]), torch.tensor(min_voc_sms_graph[:, 1])), num_nodes=2045)

        min_voc_voc_graph = dgl.add_self_loop(min_voc_voc_graph)
        min_sms_sms_graph = dgl.add_self_loop(min_sms_sms_graph)
        min_voc_sms_graph = dgl.add_self_loop(min_voc_sms_graph)

        graph = []
        graph.append(min_voc_voc_graph)
        graph.append(min_sms_sms_graph)
        graph.append(min_voc_sms_graph)
        return graph, min_voc_feature_user, min_sms_feature_user, min_user_label, min_user_personal_feature

    else:
        print('no such dataset')
        exit(1)


        # graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        # graph.ndata['feature'] = graph.ndata['feature'].float()
        #
        # # noise = feature_noise(graph.ndata['feature'] , 0.8)
        # # graph.ndata['feature']  = apply_feature_noise(graph.ndata['feature'] , noise)
        #
        # mask = feature_mask(graph.ndata['feature'] , mask_ratio)
        # graph.ndata['feature']=apply_feature_mask(graph.ndata['feature'], mask)








def normalize(mx: np.ndarray):
    shape = mx.shape
    mx = mx.reshape((-1, shape[-1])) #[sample*length, dimension]
    for k in range(mx.shape[-1]): # for each dimension
        mx[:, k] = (mx[:, k]-np.mean(mx[:, k]))/np.std(mx[:, k])
    mx = mx.reshape(shape)
    return mx