import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
import torch
import copy
from sklearn.manifold import TSNE
import os

class ConstStyle2(nn.Module):
    def __init__(self, cfg, eps=1e-6):
        super().__init__()
        self.cfg = cfg
        self.eps = eps
        self.mean = []
        self.std = []
        self.domain_list = []
        self.cluster_means = []
        self.cluster_covs = []
        self.generators = []
        
    def clear_memory(self):
        self.mean = []
        self.std = []
        self.domain_list = []
        
    def get_style(self, x):
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        var = (var + self.eps).sqrt()
        mu, var = mu.detach().squeeze().cpu().numpy(), var.detach().squeeze().cpu().numpy()
        
        return mu, var
    
    def store_style(self, x, domains):
        mu, var = self.get_style(x)
        self.mean.extend(mu)
        self.std.extend(var)
        self.domain_list.extend([i.item() for i in domains])
    
    def cal_mean_std(self, idx, epoch):
        print('Get clusters')
        self.cluster_means = []
        self.cluster_covs = []
        self.generators = []

        mean_list = np.array(self.mean)
        std_list = np.array(self.std)
        stacked_data = np.stack((mean_list, std_list), axis=1)
        reshaped_data = stacked_data.reshape((len(mean_list), -1))
        
        self.bayes_cluster = BayesianGaussianMixture(n_components=self.cfg.NUM_CLUSTERS, covariance_type='full', init_params='k-means++', max_iter=200)
        self.bayes_cluster.fit(reshaped_data)

        for idx in range(self.cfg.NUM_CLUSTERS):
            cluster_mean = torch.from_numpy(self.bayes_cluster.means_[idx]).to('cuda')
            cluster_cov = torch.from_numpy(self.bayes_cluster.covariances_[idx]).to('cuda')
            self.cluster_means.append(cluster_mean)
            self.cluster_covs.append(cluster_cov)
            cluster_generator = torch.distributions.MultivariateNormal(loc=cluster_mean, covariance_matrix=cluster_cov)
            self.generators.append(cluster_generator)
            print(f'Len cluster mean: {len(self.cluster_means)}')

    def plot_style_statistics(self, idx, epoch):
        domain_list = np.array(self.domain_list)
        #clustering
        mean_list = copy.copy(self.mean_after)
        std_list = copy.copy(self.std_after)
        mean_list = np.array(mean_list)
        std_list = np.array(std_list)
        stacked_data = np.stack((mean_list, std_list), axis=1)
        reshaped_data = stacked_data.reshape((len(mean_list), -1))
        
        classes = ['in domain 1', 'in domain 2', 'in domain 3', 'out domain']
        tsne = TSNE(n_components=2, random_state=self.cfg.SEED)
        plot_data = tsne.fit_transform(reshaped_data)
        
        scatter = plt.scatter(plot_data[:, 0], plot_data[:, 1], c=domain_list)
        plt.legend(handles=scatter.legend_elements()[0], labels=classes)
        save_path = os.path.join(f'{self.cfg.OUTPUT_DIR}', f'testing-features_after{idx}_epoch{epoch}.png')
        plt.savefig(save_path, dpi=200)
        plt.close()
        plt.cla()
        plt.clf()

    def forward(self, x, domain, store_feature=False, apply_conststyle=False, is_test=False, cluster_idx=0):
        if store_feature:
            self.store_style(x, domain)
        
        if apply_conststyle:
            mu = x.mean(dim=[2, 3], keepdim=True)
            var = x.var(dim=[2, 3], keepdim=True)
            sig = (var + self.eps).sqrt()
            mu, sig = mu.detach(), sig.detach()
            x_normed = (x - mu) / (sig + self.eps)
            
            if not is_test and np.random.random() > self.cfg.TRAINER.CONSTSTYLE.PROB:
                return x
            
            if is_test:
                const_value = torch.reshape(self.cluster_means[cluster_idx], (2, -1))
                const_mean = const_value[0].float()
                const_std = const_value[1].float()
                # print(f'Const mean of cluster {cluster_idx}:\n{const_mean} | {const_std}')
                const_mean = torch.reshape(const_mean, (1, const_mean.shape[0], 1, 1)).to('cuda')
                const_std = torch.reshape(const_std, (1, const_std.shape[0], 1, 1)).to('cuda')
            else:
                random_idx = torch.randint(0, self.cfg.NUM_CLUSTERS, (x.size(0),))
                
                style_mean = []
                style_std = []
                for i in range(len(x_normed)):
                    style = self.generators[random_idx[i]].sample()
                    style = torch.reshape(style, (2, -1))
                    style_mean.append(style[0])
                    style_std.append(style[1])
                
                const_mean = torch.vstack(style_mean).float()
                const_std = torch.vstack(style_std).float()
                
                const_mean = torch.reshape(const_mean, (const_mean.shape[0], const_mean.shape[1], 1, 1)).to('cuda')
                const_std = torch.reshape(const_std, (const_std.shape[0], const_std.shape[1], 1, 1)).to('cuda')
            
            out = x_normed * const_std + const_mean
            
            return out
        else:
            return x