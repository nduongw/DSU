import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
import torch
import copy
from sklearn.manifold import TSNE
import os

class ConstStyle3(nn.Module):
    def __init__(self, cfg, eps=1e-6):
        super().__init__()
        self.cfg = cfg
        self.eps = eps
        self.alpha = 0.5
        self.mean = []
        self.std = []
        self.const_mean = None
        self.const_cov = None
        self.domain_list = []
        self.bayes_cluster = None
        self.domain_mean_list = []
        self.domain_std_list = []
        # self.beta = torch.distributions.Beta(self.alpha, self.alpha)
    
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
        print(f'Mean length: {len(self.mean)}')
    
    def cal_mean_std(self, idx, epoch):
        #clustering
        mean_list = np.array(self.mean)
        print(f'Len of mean: {len(mean_list)}')
        std_list = np.array(self.std)
        stacked_data = np.stack((mean_list, std_list), axis=1)
        reshaped_data = stacked_data.reshape((len(mean_list), -1))
        
        self.bayes_cluster = BayesianGaussianMixture(n_components=1, covariance_type='full', init_params='k-means++', max_iter=200)
        self.bayes_cluster.fit(reshaped_data)
        
        self.const_mean = torch.from_numpy(self.bayes_cluster.means_[0])
        self.const_cov = torch.from_numpy(self.bayes_cluster.covariances_[0])

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
        
        if (not is_test and np.random.random() > self.cfg.TRAINER.CONSTSTYLE.PROB) or not apply_conststyle:
            return x
        
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig
        
        if is_test:
            const_value = torch.reshape(self.const_mean, (2, -1))
            const_mean = const_value[0].float()
            const_std = const_value[1].float()
            const_mean = torch.reshape(const_mean, (1, const_mean.shape[0], 1, 1)).to('cuda')
            const_std = torch.reshape(const_std, (1, const_std.shape[0], 1, 1)).to('cuda')
            
            beta = const_std * self.alpha + (1 - self.alpha) * sig
            gamma = const_mean * self.alpha + (1 - self.alpha) * mu
            out = x_normed * beta + gamma
        else:
            generator = torch.distributions.MultivariateNormal(loc=self.const_mean, covariance_matrix=self.const_cov)
            style_mean = []
            style_std = []
            for i in range(len(x_normed)):
                style = generator.sample()
                style = torch.reshape(style, (2, -1))
                style_mean.append(style[0])
                style_std.append(style[1])
            
            const_mean = torch.vstack(style_mean).float()
            const_std = torch.vstack(style_std).float()
            
            const_mean = torch.reshape(const_mean, (const_mean.shape[0], const_mean.shape[1], 1, 1)).to('cuda')
            const_std = torch.reshape(const_std, (const_std.shape[0], const_std.shape[1], 1, 1)).to('cuda')

            # factor = self.beta.sample((x.size(0), 1, 1, 1)).to(x.device)
            # beta = const_std * self.alpha + (1 - self.alpha) * sig
            # gamma = const_mean * self.alpha + (1 - self.alpha) * mu
            out = x_normed * const_std + const_mean
        
        return out
