import torch
import matplotlib.pyplot as plt
import networkx as nx
from parsers.config import get_config
from utils.loader import load_ckpt, load_device, load_model_from_ckpt, load_ema_from_ckpt, load_sampling_fn2, load_seed, load_data
from utils.graph_utils import init_flags2, mask_adjs, quantize, adjs_to_graphs

# --- User: Set these ---
CONFIG_NAME = "Digg"  # e.g., "Digg", "community_small", etc.
CKPT_NAME = "Jun18-06-35-35"  
SEED = 12   # this is the seed for the random number generator

# --- Load config ---
config = get_config(CONFIG_NAME, SEED)
config.ckpt = CKPT_NAME

# --- Load device ---
device = load_device()

# --- Load checkpoint and models ---
ckpt_dict = load_ckpt(config, device)
configt = ckpt_dict['config']
if not hasattr(configt, 'type'):
    configt.type = config.type

# --- Set batch size to 1 for single graph generation ---
config.data.batch_size = 1000
configt.data.batch_size = 1000
config.seed = None
configt.seed = None

# --- Set seed for reproducibility ---
# load_seed(configt.seed)    # don't load: random
train_graph_list, _ = load_data(configt, get_graph_list=True)

# --- Load models ---
model_x = load_model_from_ckpt(ckpt_dict['params_x'], ckpt_dict['x_state_dict'], device)
model_adj = load_model_from_ckpt(ckpt_dict['params_adj'], ckpt_dict['adj_state_dict'], device)

if config.sample.use_ema:
    ema_x = load_ema_from_ckpt(model_x, ckpt_dict['ema_x'], configt.train.ema)
    ema_adj = load_ema_from_ckpt(model_adj, ckpt_dict['ema_adj'], configt.train.ema)
    ema_x.copy_to(model_x.parameters())
    ema_adj.copy_to(model_adj.parameters())

# --- Prepare sampling function ---
sampling_fn2 = load_sampling_fn2(configt, config.sampler, config.sample, device)

# --- Generate one graph ---
# init_flags, train_tensor = init_flags2(train_graph_list, configt, batch_size=1)
init_flags, train_tensor = init_flags2(train_graph_list, configt)
init_flags = init_flags.to(device[0])
train_tensor = train_tensor.to(device[0])

with torch.no_grad():
    x, adj, _ = sampling_fn2(model_x, model_adj, init_flags, train_tensor)
    adj = mask_adjs(adj, init_flags)
    adj = adj.triu(1)
    adj = adj + torch.transpose(adj, -1, -2)
    samples_int = quantize(adj)
    graph_list = adjs_to_graphs(samples_int, True)

# --- Output the generated graph ---
print("generated:  len(graph_list)=", len(graph_list))
graph = graph_list[0]
print(f"Number of nodes: {graph.number_of_nodes()}, Number of edges: {graph.number_of_edges()}")

# --- Visualize the graph ---
plt.figure(figsize=(4,4))
nx.draw(graph, with_labels=True, node_color='skyblue', edge_color='gray')
plt.title("Generated Graph")
plt.savefig("test.png") 