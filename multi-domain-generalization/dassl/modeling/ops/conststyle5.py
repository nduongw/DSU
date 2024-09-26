import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
import torch
import copy
from sklearn.manifold import TSNE
import os
from scipy.linalg import sqrtm

def wasserstein_distance_multivariate(mean1, cov1, mean2, cov2):
    mean_diff = mean1 - mean2
    mean_distance = np.dot(mean_diff, mean_diff)
    sqrt_cov1 = sqrtm(cov1)
    if np.iscomplexobj(sqrt_cov1):
        sqrt_cov1 = sqrt_cov1.real
    # Compute the term involving the covariance matrices
    cov_sqrt_product = sqrtm(sqrt_cov1 @ cov2 @ sqrt_cov1)
    if np.iscomplexobj(cov_sqrt_product):
        cov_sqrt_product = cov_sqrt_product.real

    cov_term = np.trace(cov1 + cov2 - 2 * cov_sqrt_product)
    wasserstein_distance = np.sqrt(mean_distance + cov_term)
    return wasserstein_distance

def mahalanobis_distance(point, mean, cov):
    inv_cov_matrix = torch.linalg.inv(cov)
    diff = point - mean
    distance = torch.sqrt(torch.matmul(torch.matmul(diff, inv_cov_matrix), torch.t(diff)))
    distance = torch.diagonal(distance)
    return distance

def euclid_distance(point, mean):
    diff = point - mean
    square_diff = diff ** 2
    sum_square_diff = torch.sum(square_diff, dim=1)
    distance = torch.sqrt(sum_square_diff)
    return distance

def wasserstein_distance(mu1, sigma1, mu2, sigma2):
    mu2, sigma2 = torch.squeeze(mu2), torch.squeeze(sigma2)
    mean_diff_squared = (mu1 - mu2) ** 2
    std_diff_squared = (sigma1 - sigma2) ** 2
    wasserstein_distance = torch.sqrt(torch.sum(mean_diff_squared, dim=1) + torch.sum(std_diff_squared, dim=1))
    return wasserstein_distance

def sigmoid_increase(x, k=6.0):
    return 1 / (1 + torch.exp(-k * (x - 1)))

class ConstStyle5(nn.Module):
    def __init__(self, idx, cfg, eps=1e-6):
        super().__init__()
        self.idx = idx
        self.cfg = cfg
        self.eps = eps
        self.alpha = cfg.TRAINER.CONSTSTYLE.ALPHA
        self.mean = []
        self.std = []
        self.domain = []
        self.const_mean = None
        self.const_cov = None
        self.bayes_cluster = None
        self.beta = torch.distributions.Beta(0.3, 0.3)
    
    def clear_memory(self):
        self.mean = []
        self.std = []
        self.domain = []
        
    def get_style(self, x):
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        var = (var + self.eps).sqrt()
        mu, var = mu.detach().squeeze().cpu().numpy(), var.detach().squeeze().cpu().numpy()
        return mu, var
    
    def store_style(self, x, domain):
        mu, var = self.get_style(x)
        self.mean.extend(mu)
        self.std.extend(var)
        self.domain.extend(domain.detach().squeeze().cpu().numpy())
    
    def cal_mean_std(self, idx, epoch):
        mean_list = np.array(self.mean)
        std_list = np.array(self.std)
        stacked_data = np.stack((mean_list, std_list), axis=1)
        reshaped_data = stacked_data.reshape((len(mean_list), -1))
        
        print(f'Number of cluster: {self.cfg.NUM_CLUSTERS}')
        self.bayes_cluster = BayesianGaussianMixture(n_components=self.cfg.NUM_CLUSTERS, covariance_type='full', init_params='k-means++', max_iter=200)
        self.bayes_cluster.fit(reshaped_data)
        
        labels = self.bayes_cluster.predict(reshaped_data)
        unique_labels, _ = np.unique(labels, return_counts=True)
        
        cluster_mean = np.mean([self.bayes_cluster.means_[i] for i in range(len(unique_labels))], axis=0)
        cluster_cov = np.mean([self.bayes_cluster.covariances_[i] for i in range(len(unique_labels))], axis=0)
        
        self.const_mean = torch.from_numpy(cluster_mean)
        self.const_cov = torch.from_numpy(cluster_cov)
        self.generator = torch.distributions.MultivariateNormal(loc=self.const_mean, covariance_matrix=self.const_cov)

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
    
    def calculate_domain_distance(self, idx):
        mean_list = np.array(self.mean)
        std_list = np.array(self.std)
        domain_list = np.array(self.domain)
        stacked_data = np.stack((mean_list, std_list), axis=1)
        reshaped_data = stacked_data.reshape((len(mean_list), -1))
        
        unique_domain = np.unique(domain_list)
        print(f'Unique domain: {unique_domain}')
        seen_domain_mean, seen_domain_cov = [], []
        unseen_domain_mean, unseen_domain_cov = [], []
        for val in unique_domain:
            domain_feats = reshaped_data[domain_list == val]
            domain_distribution = BayesianGaussianMixture(n_components=1, covariance_type='full', init_params='k-means++', max_iter=200)
            domain_distribution.fit(domain_feats)
            if val < 10:
                seen_domain_mean.append(domain_distribution.means_[0])
                seen_domain_cov.append(domain_distribution.covariances_[0])
            else:
                unseen_domain_mean.append(domain_distribution.means_[0])
                unseen_domain_cov.append(domain_distribution.covariances_[0])
        
        total_distance = 0.0
        for i in range(len(seen_domain_mean)):
            for j in range(len(unseen_domain_mean)):
                distance = wasserstein_distance_multivariate(unseen_domain_mean[j], unseen_domain_cov[j], seen_domain_mean[i], seen_domain_cov[i])
                total_distance += distance
        
        print(f'Total distance from seen to unseen domain of layer {idx}: {total_distance}')
        # center_to_unseen_dist = wasserstein_distance_multivariate(unseen_domain_mean[0], unseen_domain_cov[0], self.const_mean.numpy(), self.const_cov.numpy())
        # print(f'Distance from center to unseen domain of layer {idx}: {center_to_unseen_dist}\n')

    def forward(self, x, domain, store_feature=False, apply_conststyle=False, is_test=False, apply_rate=0.0):
        if store_feature:
            self.store_style(x, domain)
        
        if (not is_test and np.random.random() > self.cfg.TRAINER.CONSTSTYLE.PROB) or not apply_conststyle:
            return x
        
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        # style_feats = torch.hstack((mu, sig)).squeeze().detach().cpu()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig
        
        if is_test:
            # wass_dist = wasserstein_distance(const_mean, const_std, mu, sig)
            # if self.idx == 0:
            #     print(f'Wass distance max value of conststyle layer {self.idx}: {torch.max(wass_dist)} | min value: {torch.min(wass_dist)}')

            const_value = torch.reshape(self.const_mean, (2, -1))
            const_mean = const_value[0].float().to('cuda')
            const_std = const_value[1].float().to('cuda')
            
            const_mean = torch.reshape(const_mean, (1, const_mean.shape[0], 1, 1))
            const_std = torch.reshape(const_std, (1, const_std.shape[0], 1, 1))
            if not apply_rate:
                beta = const_std * self.alpha + (1 - self.alpha) * sig
                gamma = const_mean * self.alpha + (1 - self.alpha) * mu
            else:
                beta = const_std
                gamma = const_mean
            
        else:
            style_mean = []
            style_std = []
            for i in range(len(x_normed)):
                style = self.generator.sample()
                style = torch.reshape(style, (2, -1))
                style_mean.append(style[0])
                style_std.append(style[1])
            
            const_mean = torch.vstack(style_mean).float()
            const_std = torch.vstack(style_std).float()
            
            const_mean = torch.reshape(const_mean, (const_mean.shape[0], const_mean.shape[1], 1, 1)).to('cuda')
            const_std = torch.reshape(const_std, (const_std.shape[0], const_std.shape[1], 1, 1)).to('cuda')

            beta = const_std
            gamma = const_mean
        out = x_normed * beta + gamma
            
        return out
