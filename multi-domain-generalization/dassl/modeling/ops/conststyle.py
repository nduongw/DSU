import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
import torch
import copy
from sklearn.manifold import TSNE
import os
import ot

class ConstStyle(nn.Module):
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
        #clustering
        mean_list = np.array(self.mean)
        std_list = np.array(self.std)
        stacked_data = np.stack((mean_list, std_list), axis=1)
        reshaped_data = stacked_data.reshape((len(mean_list), -1))
        
        self.bayes_cluster = BayesianGaussianMixture(n_components=self.cfg.NUM_CLUSTERS, covariance_type='full', init_params='k-means++', max_iter=200)
        self.bayes_cluster.fit(reshaped_data)
        
        labels = self.bayes_cluster.predict(reshaped_data)
        unique_labels, _ = np.unique(labels, return_counts=True)
        
        cluster_samples = []
        cluster_samples_idx = []
        cluster_means = []
        cluster_covs = []
        for val in unique_labels:
            print(f'Get samples belong to cluster {val}')
            samples = [reshaped_data[i] for i in range(len(labels)) if labels[i] == val]
            samples_idx = [i for i in range(len(labels)) if labels[i] == val]
            samples = np.stack(samples)
            print(f'Cluster {val} has {len(samples)} samples')
            cluster_samples.append(samples)
            cluster_samples_idx.append(samples_idx)
            
            cluster_means.append(self.bayes_cluster.means_[val])
            cluster_covs.append(self.bayes_cluster.covariances_[val])
            # reshaped_mean = self.bayes_cluster.means_[val].reshape(2, -1)
            # self.domain_mean_list.append(reshaped_mean[0])
            # self.domain_std_list.append(reshaped_mean[1])
        
        cluster_generated_samples = []
        for idx in range(len(unique_labels)):
            cluster_mean, cluster_cov = cluster_means[idx], cluster_covs[idx]
            generated_sample = np.random.multivariate_normal(cluster_mean, cluster_cov, 1000)
            cluster_generated_samples.append(generated_sample)
        
        if self.cfg.CLUSTER == 'barycenter':
            # cluster_means = np.stack([self.bayes_cluster.means_[i] for i in range(len(unique_labels))])
            # cluster_covs = np.stack([self.bayes_cluster.covariances_[i] for i in range(len(unique_labels))])
            # weights = np.ones(len(unique_labels), dtype=np.float64) / len(unique_labels)
            
            # total_mean, total_cov = ot.gaussian.bures_wasserstein_barycenter(cluster_means, cluster_covs, weights)
            total_mean = np.mean([self.bayes_cluster.means_[i] for i in range(len(unique_labels))], axis=0)
            total_cov = np.mean([self.bayes_cluster.covariances_[i] for i in range(len(unique_labels))], axis=0)
            self.const_mean = torch.from_numpy(total_mean)
            self.const_cov = torch.from_numpy(total_cov)
        elif self.cfg.CLUSTER == 'ot':
            ot_score = []
            for i in range(len(cluster_samples_idx)):
                total_cost = 0.0
                cluster_sample_x = cluster_generated_samples[i]
                for j in range(len(cluster_samples_idx)):
                    if i == j:
                        continue
                    else:
                        cluster_sample_y = cluster_generated_samples[j]
                        cluster_sample_x = np.array(cluster_sample_x)
                        cluster_sample_y = np.array(cluster_sample_y)
                        if self.cfg.DISTANCE == 'wass':
                            M = ot.dist(cluster_sample_y, cluster_sample_x)
                            a, b = np.ones(len(cluster_sample_y)) / len(cluster_sample_y), np.ones(len(cluster_sample_x)) / len(cluster_sample_x) 
                            cost = ot.emd2(a, b, M)
                            print(f'Cost to move from cluster {j} to cluster {i} is {cost}')
                        total_cost += cost
                print(f'Total cost of cluster {i}: {total_cost}')
                ot_score.append(total_cost)

            idx_val = np.argmin(ot_score)
            print(f'Layer {idx} chooses cluster {unique_labels[idx_val]} with minimal cost {ot_score[idx_val]}')
            self.const_mean = torch.from_numpy(self.bayes_cluster.means_[idx_val])
            self.const_cov = torch.from_numpy(self.bayes_cluster.covariances_[idx_val])

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
            x_normed = (x - mu) / sig
            
            if not is_test and np.random.random() > self.cfg.TRAINER.CONSTSTYLE.PROB:
                return x
            
            if is_test:
                const_value = torch.reshape(self.const_mean, (2, -1))
                const_mean = const_value[0].float()
                const_std = const_value[1].float()
                const_mean = torch.reshape(const_mean, (1, const_mean.shape[0], 1, 1)).to('cuda')
                const_std = torch.reshape(const_std, (1, const_std.shape[0], 1, 1)).to('cuda')
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
            
            beta = const_std * self.alpha + (1 - self.alpha) * sig
            gamma = const_mean * self.alpha + (1 - self.alpha) * mu
            out = x_normed * beta + gamma
            # out = x_normed * const_std + const_mean
            
            return out
        else:
            return x