import os
import time
import pickle
import math
import torch

from utils.logger import Logger, set_log, start_log, train_log, sample_log, check_log
from utils.loader import load_ckpt, load_data, load_seed, load_device, load_model_from_ckpt, \
    load_ema_from_ckpt, load_sampling_fn, load_eval_settings, load_sampling_fn2
from utils.graph_utils import adjs_to_graphs, init_flags, quantize, init_flags2, mask_adjs, init_flags3
from utils.plot import save_graph_list, plot_graphs_list
from evaluation.stats import eval_graph_list
from utils.mol_utils import gen_mol, mols_to_smiles, load_smiles, canonicalize_smiles, mols_to_nx
from moses.metrics.metrics import get_all_metrics
import numpy as np
import networkx as nx


# -------- Sampler for generic graph generation tasks --------
class Sampler(object):
    def __init__(self, config):
        super(Sampler, self).__init__()

        self.config = config
        self.device = load_device()

    def sample(self):
        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device)
        self.configt = self.ckpt_dict['config']      # it seems that the configt is the same as self.config
        if not "type" in self.configt:
            print("self.configt has not type setting")
            self.configt.type = self.config.type

        load_seed(self.configt.seed)
        self.train_graph_list, self.test_graph_list = load_data(self.configt, get_graph_list=True)

        self.log_folder_name, self.log_dir, _ = set_log(self.configt, is_train=False)
        self.log_name = f"{self.config.ckpt}-sample"
        logger = Logger(str(os.path.join(self.log_dir, f'{self.log_name}.log')), mode='a')

        if not check_log(self.log_folder_name, self.log_name):
            logger.log(f'{self.log_name}')
            start_log(logger, self.configt)
            train_log(logger, self.configt)
        sample_log(logger, self.config)

        # -------- Load models --------
        self.model_x = load_model_from_ckpt(self.ckpt_dict['params_x'], self.ckpt_dict['x_state_dict'], self.device)
        self.model_adj = load_model_from_ckpt(self.ckpt_dict['params_adj'], self.ckpt_dict['adj_state_dict'],
                                              self.device)

        if self.config.sample.use_ema:
            self.ema_x = load_ema_from_ckpt(self.model_x, self.ckpt_dict['ema_x'], self.configt.train.ema)
            self.ema_adj = load_ema_from_ckpt(self.model_adj, self.ckpt_dict['ema_adj'], self.configt.train.ema)

            self.ema_x.copy_to(self.model_x.parameters())
            self.ema_adj.copy_to(self.model_adj.parameters())

        # self.sampling_fn = load_sampling_fn(self.configt, self.config.sampler, self.config.sample, self.device)
        self.sampling_fn2 = load_sampling_fn2(self.configt, self.config.sampler, self.config.sample, self.device)

        # -------- Generate samples --------
        logger.log(f'GEN SEED: {self.config.sample.seed}')
        load_seed(self.config.sample.seed)

        num_sampling_rounds = math.ceil(len(self.test_graph_list) / self.configt.data.batch_size)
        gen_graph_list = []

        for r in range(num_sampling_rounds):
            t_start = time.time()
            # print("self.configt.data.data:",self.configt.data.data)
            # self.init_flags = init_flags(self.train_graph_list, self.configt).to(self.device[0])
            self.init_flags, train_tensor = init_flags2(self.train_graph_list, self.configt)
            self.init_flags = self.init_flags.to(self.device[0])
            train_tensor = train_tensor.to(self.device[0])
            # x, adj, _ = self.sampling_fn(self.model_x, self.model_adj, self.init_flags)
            x, adj, _ = self.sampling_fn2(self.model_x, self.model_adj, self.init_flags, train_tensor)

            # la, u = torch.symeig(adj, eigenvectors=True)
            # for i in range(adj.shape[0]):
            #     print(adj[i])
            # la = torch.diag_embed(la)
            adj = mask_adjs(adj, self.init_flags)

            logger.log(f"Round {r} : {time.time() - t_start:.2f}s")

            adj_np = adj.cpu().numpy()

            adj = adj.triu(1)
            adj = adj + torch.transpose(adj, -1, -2)

            samples_int = quantize(adj)
            # print("samples_int:", samples_int)
            samples_int_np = samples_int.cpu().numpy()

            if r == 0:
                adj_np_to_save = adj_np
                samples_int_np_to_save = samples_int_np
            else:
                adj_np_to_save = np.vstack([adj_np_to_save, adj_np])
                samples_int_np_to_save = np.vstack([samples_int_np_to_save, samples_int_np])

            gen_graph_list.extend(adjs_to_graphs(samples_int, True))

        gen_graph_list = gen_graph_list[:len(self.test_graph_list)]

        print("len(gen_graph_list):", len(gen_graph_list))
        print("len(test_graph_list):", len(self.test_graph_list))

        # -------- Evaluation --------
        methods, kernels = load_eval_settings(self.config.data.data)
        result_dict = eval_graph_list(self.test_graph_list, gen_graph_list, methods=methods, kernels=kernels)
        logger.log(f'MMD_full {result_dict}', verbose=False)
        logger.log('=' * 100)

        # -------- Save samples --------
        save_dir = save_graph_list(self.log_folder_name, self.log_name, gen_graph_list)
        with open(save_dir, 'rb') as f:
            sample_graph_list = pickle.load(f)
        plot_graphs_list(graphs=sample_graph_list, title=f'{self.config.ckpt}', max_num=16,
                         save_dir=self.log_folder_name)
        
        exit(0)
