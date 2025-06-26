import utils
from utils.plot import save_graph_list, plot_graphs_list
from utils.loader import load_data
from parsers.parser import Parser
from parsers.config import get_config
import pickle

graph_name = "Digg"
with open(f"data/{graph_name}.pkl", 'rb') as f:
    graph_list = pickle.load(f)
print(len(graph_list))
plot_graphs_list(graphs=graph_list, title=f'{graph_name}: 2-hop ego', max_num=16,
                         save_dir=f"data")