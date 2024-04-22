import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np
from dassl.utils import init_network_weights
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
import copy
import os
import ot
from sklearn.manifold import TSNE

from .build import BACKBONE_REGISTRY
from .backbone import Backbone

def KLdivergence(x, y):
    """Compute the Kullback-Leibler divergence between two multivariate samples.
    Parameters
    ----------
    x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
    y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
    Returns
    -------
    out : float
    The estimated Kullback-Leibler divergence D(P||Q).
    References
    ----------
    Pérez-Cruz, F. Kullback-Leibler divergence estimation of
    continuous distributions IEEE International Symposium on Information
    Theory, 2008.
    """
    from scipy.spatial import cKDTree as KDTree

    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n,d = x.shape
    m,dy = y.shape

    assert(d == dy)

    xtree = KDTree(x)
    ytree = KDTree(y)

    r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]

    return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))

def JSdivergence(x, y):
    m = 0.5 * (x + y)
    return 0.5 * KLdivergence(x, m) + 0.5 * KLdivergence(y, m)

def Bdistance(x, y):
    x /= x.sum(axis=0)
    y /= y.sum(axis=0)
    
    out = x * y
    out = np.sum(out, axis=1)
    out = np.sqrt(np.abs(out))
    out = np.sum(out)
    return -np.log(out)

class Convolution(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.conv(x))
    
class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].
        dim   (int): dimension of feature map channels

    """

    def __init__(self, p=0.5, eps=1e-6, dim=-1):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x

class CorrelatedDistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].
        dim   (int): dimension of feature map channels

    """

    def __init__(self, p=0.5, eps=1e-6, alpha=0.3):
        super(CorrelatedDistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.alpha = alpha
        self.beta = torch.distributions.Beta(alpha, alpha)
    
    def __repr__(self):
        return f'CorrelatedDistributionUncertainty with p {self.p} and alpha {self.alpha}'

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        B, C = x.size(0), x.size(1)
        mu = torch.mean(x, dim=[2, 3], keepdim=True)
        sig = (x.var(dim=[2, 3], keepdim=True) + self.eps).sqrt()
        # mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        factor = self.beta.sample((B, 1, 1, 1)).to(x.device)

        mu_squeeze = torch.squeeze(mu)
        mean_mu = torch.mean(mu_squeeze, dim=0, keepdim=True)
        correlation_mu = (mu_squeeze-mean_mu).T @ (mu_squeeze-mean_mu) / B

        sig_squeeze = torch.squeeze(sig)
        mean_sig = torch.mean(sig_squeeze, dim=0, keepdim=True)
        correlation_sig = (sig_squeeze.T-mean_sig.T) @ (sig_squeeze-mean_sig) / B

        with torch.no_grad():
            try:
                _, mu_eng_vector = torch.linalg.eigh(C*correlation_mu+self.eps*torch.eye(C, device=x.device))
                # mu_corr_matrix = mu_eng_vector @ torch.sqrt(torch.diag(torch.clip(mu_eng_value, min=1e-10))) @ (mu_eng_vector.T)
            except:
                mu_eng_vector = torch.eye(C, device=x.device)
            
            if not torch.all(torch.isfinite(mu_eng_vector)) or torch.any(torch.isnan(mu_eng_vector)):
                mu_eng_vector = torch.eye(C, device=x.device)

            try:
                _, sig_eng_vector = torch.linalg.eigh(C*correlation_sig+self.eps*torch.eye(C, device=x.device))
                # sig_corr_matrix = sig_eng_vector @ torch.sqrt(torch.diag(torch.clip(sig_eng_value, min=1e-10))) @ (sig_eng_vector.T)
            except:
                sig_eng_vector = torch.eye(C, device=x.device)

            if not torch.all(torch.isfinite(sig_eng_vector )) or torch.any(torch.isnan(sig_eng_vector)):
                sig_eng_vector = torch.eye(C, device=x.device)

        mu_corr_matrix = mu_eng_vector @ torch.diag(torch.sqrt(torch.clip(torch.diag((mu_eng_vector.T)@ correlation_mu @ mu_eng_vector),min=1e-12))) @ (mu_eng_vector.T)
        sig_corr_matrix = sig_eng_vector @ torch.diag(torch.sqrt(torch.clip(torch.diag((sig_eng_vector.T)@ correlation_sig @ sig_eng_vector), min=1e-12))) @ (sig_eng_vector.T)

        gaussian_mu = (torch.randn(B, 1, C, device=x.device) @ mu_corr_matrix)
        gaussian_mu = torch.reshape(gaussian_mu, (B, C, 1, 1))

        gaussian_sig = (torch.randn(B, 1, C, device=x.device) @ sig_corr_matrix)
        gaussian_sig = torch.reshape(gaussian_sig, (B, C, 1, 1))

        mu_mix = mu + factor*gaussian_mu
        sig_mix = sig + factor*gaussian_sig

        return x_normed * sig_mix + mu_mix

class ConstStyle(nn.Module):
    def __init__(self, cfg, eps=1e-6):
        super().__init__()
        self.cfg = cfg
        self.mean = []
        self.std = []
        self.mean_after = []
        self.std_after = []
        self.eps = eps
        self.const_mean = None
        self.const_cov = None
        self.domain_list = []
    
    def clear_memory(self):
        self.mean = []
        self.std = []
        self.mean_after = []
        self.std_after = []
        self.domain_list = []
        
    def get_style(self, x):
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        var = var.sqrt()
        mu, var = mu.detach().squeeze().cpu().numpy(), var.detach().squeeze().cpu().numpy()
        
        return mu, var
    
    def store_style(self, x, domains):
        mu, var = self.get_style(x)
        self.mean.extend(mu)
        self.std.extend(var)
        self.domain_list.extend([i.item() for i in domains])
    
    def store_style_after(self, x):
        mu, var = self.get_style(x)
        self.mean_after.extend(mu)
        self.std_after.extend(var)
    
    def cal_mean_std(self, idx, epoch):
        domain_list = np.array(self.domain_list)
        #clustering
        mean_list = copy.copy(self.mean)
        std_list = copy.copy(self.std)
        mean_list = np.array(mean_list)
        std_list = np.array(std_list)
        stacked_data = np.stack((mean_list, std_list), axis=1)
        reshaped_data = stacked_data.reshape((len(mean_list), -1))
        
        bayes_cluster = BayesianGaussianMixture(n_components=self.cfg.NUM_CLUSTERS, covariance_type='full', init_params='k-means++', max_iter=200)
        bayes_cluster.fit(reshaped_data)
        
        labels = bayes_cluster.predict(reshaped_data)
        unique_labels, counts = np.unique(labels, return_counts=True)
        
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
            
            cluster_means.append(bayes_cluster.means_[val])
            cluster_covs.append(bayes_cluster.covariances_[val])
        
        cluster_generated_samples = []
        for idx in range(len(unique_labels)):
            cluster_mean, cluster_cov = cluster_means[idx], cluster_covs[idx]
            generated_sample = np.random.multivariate_normal(cluster_mean, cluster_cov, 1000)
            cluster_generated_samples.append(generated_sample)
        
        if self.cfg.CLUSTER == 'barycenter':
            cluster_means = np.stack([bayes_cluster.means_[i] for i in range(len(unique_labels))])
            cluster_covs = np.stack([bayes_cluster.covariances_[i] for i in range(len(unique_labels))])
            weights = np.ones(len(unique_labels), dtype=np.float64) / len(unique_labels)
            
            total_mean, total_cov = ot.gaussian.bures_wasserstein_barycenter(cluster_means, cluster_covs, weights)
            self.const_mean = torch.from_numpy(total_mean)
            self.const_cov = torch.from_numpy(total_cov)
            print(f'Layer {idx} choose distribution with mean {np.mean(total_mean)} and std {np.mean(total_cov)}')
        elif self.cfg.CLUSTER == 'ot':
            ot_score = []
            for i in range(len(cluster_samples_idx)):
                total_cost = 0.0
                # cluster_sample_x = [reshaped_data[x] for x in cluster_samples_idx[i]]
                cluster_sample_x = cluster_generated_samples[i]
                for j in range(len(cluster_samples_idx)):
                    if i == j:
                        continue
                    else:
                        # cluster_sample_y = [reshaped_data[k] for k in cluster_samples_idx[j]]
                        cluster_sample_y = cluster_generated_samples[j]
                        cluster_sample_x = np.array(cluster_sample_x)
                        cluster_sample_y = np.array(cluster_sample_y)
                        if self.cfg.DISTANCE == 'wass':
                            M = ot.dist(cluster_sample_y, cluster_sample_x)
                            a, b = np.ones(len(cluster_sample_y)) / len(cluster_sample_y), np.ones(len(cluster_sample_x)) / len(cluster_sample_x) 
                            cost = ot.emd2(a, b, M)
                            # pwd = ot.sliced.sliced_wasserstein_distance(cluster_sample_y, cluster_sample_x, seed=self.cfg.SEED, n_projections=128)
                            print(f'Cost to move from cluster {j} to cluster {i} is {cost}')
                        elif self.cfg.DISTANCE == 'kl':
                            cost = KLdivergence(cluster_sample_y, cluster_sample_x)
                            print(f'KL div from cluster {j} to cluster {i} is {cost}')
                        elif self.cfg.DISTANCE == 'jensen':
                            cost = JSdivergence(cluster_sample_y, cluster_sample_x)
                            print(f'Jensen-Shanon distance from cluster {j} to cluster {i} is {cost}')
                        elif self.cfg.DISTANCE == 'bhatta':
                            cost = Bdistance(cluster_sample_y, cluster_sample_x)
                            print(f'Bhattacharyya distance from cluster {j} to cluster {i} is {cost}')
                        total_cost += cost
                print(f'Total cost of cluster {i}: {total_cost}')
                ot_score.append(total_cost)

            idx_val = np.argmin(ot_score)
            print(f'Layer {idx} chooses cluster {unique_labels[idx_val]} with minimal cost {ot_score[idx_val]}')
            self.const_mean = torch.from_numpy(bayes_cluster.means_[idx_val])
            self.const_cov = torch.from_numpy(bayes_cluster.covariances_[idx_val])
    
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

    def forward(self, x, domain, store_feature=False, apply_conststyle=False, is_test=False):
        if store_feature:
            self.store_style(x, domain)
        
        if apply_conststyle:
            mu = x.mean(dim=[2, 3], keepdim=True)
            var = x.var(dim=[2, 3], keepdim=True)
            sig = (var + self.eps).sqrt()
            mu, sig = mu.detach(), sig.detach()
            
            if not is_test and np.random.random() > 0.5:
                return x
            
            x_normed = (x-mu) / sig
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
                
            out = x_normed * const_std + const_mean
            
            self.store_style_after(out)
            return out
        else:
            return x
        
class ConvNet(Backbone):

    def __init__(self, c_hidden=64):
        super().__init__()
        self.conv1 = Convolution(3, c_hidden)
        self.conv2 = Convolution(c_hidden, c_hidden)
        self.conv3 = Convolution(c_hidden, c_hidden)
        self.conv4 = Convolution(c_hidden, c_hidden)

        self._out_features = 2**2 * c_hidden

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert H == 32 and W == 32, \
            'Input to network must be 32x32, ' \
            'but got {}x{}'.format(H, W)

    def forward(self, x):
        self._check_input(x)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        return x.view(x.size(0), -1)

class UConvNet(Backbone):

    def __init__(self, c_hidden=64, pertubration=None, uncertainty=0.0):
        super().__init__()
        self.conv1 = Convolution(3, c_hidden)
        self.conv2 = Convolution(c_hidden, c_hidden)
        self.conv3 = Convolution(c_hidden, c_hidden)
        self.conv4 = Convolution(c_hidden, c_hidden)
        
        self.pertubration0 = pertubration(p=uncertainty) if pertubration else nn.Identity()
        self.pertubration1 = pertubration(p=uncertainty) if pertubration else nn.Identity()
        self.pertubration2 = pertubration(p=uncertainty) if pertubration else nn.Identity()
        self.pertubration3 = pertubration(p=uncertainty) if pertubration else nn.Identity()

        self._out_features = 2**2 * c_hidden

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert H == 32 and W == 32, \
            'Input to network must be 32x32, ' \
            'but got {}x{}'.format(H, W)

    def forward(self, x):
        self._check_input(x)
        x = self.conv1(x)
        x = self.pertubration0(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.pertubration1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        # x = self.pertubration2(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        # x = self.pertubration3(x)
        x = F.max_pool2d(x, 2)
        return x.view(x.size(0), -1)

class CUConvNet(Backbone):

    def __init__(self, c_hidden=64, pertubration_list:list=['layer1'], uncertainty=0.0, alpha=0.3):
        super().__init__()
        self.conv1 = Convolution(3, c_hidden)
        self.conv2 = Convolution(c_hidden, c_hidden)
        self.conv3 = Convolution(c_hidden, c_hidden)
        self.conv4 = Convolution(c_hidden, c_hidden)
        
        self.pertubration_list = pertubration_list
        self.pertubration0 = CorrelatedDistributionUncertainty(p=uncertainty, alpha=alpha) if 'layer0' in pertubration_list else nn.Identity()
        self.pertubration1 = CorrelatedDistributionUncertainty(p=uncertainty, alpha=alpha) if 'layer1' in pertubration_list else nn.Identity()
        self.pertubration2 = CorrelatedDistributionUncertainty(p=uncertainty, alpha=alpha) if 'layer2' in pertubration_list else nn.Identity()
        self.pertubration3 = CorrelatedDistributionUncertainty(p=uncertainty, alpha=alpha) if 'layer3' in pertubration_list else nn.Identity()
        
        self._out_features = 2**2 * c_hidden

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert H == 32 and W == 32, \
            'Input to network must be 32x32, ' \
            'but got {}x{}'.format(H, W)

    def forward(self, x):
        self._check_input(x)
        x = self.conv1(x)
        x = self.pertubration0(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.pertubration1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        # x = self.pertubration2(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        # x = self.pertubration3(x)
        x = F.max_pool2d(x, 2)
        return x.view(x.size(0), -1)
    
class MSConvNet(Backbone):

    def __init__(self, c_hidden=64, ms_class=None, ms_layers=[], ms_p=0.5, ms_a=0.1):
        super().__init__()
        self.conv1 = Convolution(3, c_hidden)
        self.conv2 = Convolution(c_hidden, c_hidden)
        self.conv3 = Convolution(c_hidden, c_hidden)
        self.conv4 = Convolution(c_hidden, c_hidden)
        
        self.mixstyle = None
        if ms_layers:
            self.mixstyle = ms_class(p=ms_p, alpha=ms_a)
            for layer_name in ms_layers:
                assert layer_name in ["layer1", "layer2", "layer3"]
            print(
                f"Insert {self.mixstyle.__class__.__name__} after {ms_layers}"
            )
        self.ms_layers = ms_layers
        
        self._out_features = 2**2 * c_hidden

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert H == 32 and W == 32, \
            'Input to network must be 32x32, ' \
            'but got {}x{}'.format(H, W)

    def forward(self, x):
        self._check_input(x)
        x = self.conv1(x)
        if "layer1" in self.ms_layers:
            x = self.mixstyle(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        return x.view(x.size(0), -1)

class ConstConvNet(Backbone):

    def __init__(self, c_hidden=64, cfg=None):
        super().__init__()
        self.conv1 = Convolution(3, c_hidden)
        self.conv2 = Convolution(c_hidden, c_hidden)
        self.conv3 = Convolution(c_hidden, c_hidden)
        self.conv4 = Convolution(c_hidden, c_hidden)
        self.num_conststyle = 1
        self.conststyle = [ConstStyle(cfg) for i in range(self.num_conststyle)]

        self._out_features = 2**2 * c_hidden

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert H == 32 and W == 32, \
            'Input to network must be 32x32, ' \
            'but got {}x{}'.format(H, W)

    def forward(self, x, domain, store_feature=False, apply_conststyle=False, is_test=False):
        self._check_input(x)
        x = self.conv1(x)
        x = self.conststyle[0](x, domain, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        return x.view(x.size(0), -1)

@BACKBONE_REGISTRY.register()
def cnn_digitsdg(**kwargs):
    """
    This architecture was used for DigitsDG dataset in:

        - Zhou et al. Deep Domain-Adversarial Image Generation
        for Domain Generalisation. AAAI 2020.
    """
    model = ConvNet(c_hidden=64)
    init_network_weights(model, init_type='kaiming')
    return model

@BACKBONE_REGISTRY.register()
def cnn_digitsdg_mixstyle(**kwargs):
    """
    This architecture was used for DigitsDG dataset in:

        - Zhou et al. Deep Domain-Adversarial Image Generation
        for Domain Generalisation. AAAI 2020.
    """
    from dassl.modeling.ops import MixStyle
    model = MSConvNet(c_hidden=64, ms_class=MixStyle, ms_layers=["layer1", "layer2"])
    init_network_weights(model, init_type='kaiming')
    return model

@BACKBONE_REGISTRY.register()
def cnn_digitsdg_uncertainty(uncertainty=0.0, **kwargs):
    """
    This architecture was used for DigitsDG dataset in:

        - Zhou et al. Deep Domain-Adversarial Image Generation
        for Domain Generalisation. AAAI 2020.
    """
    model = UConvNet(c_hidden=64, pertubration=DistributionUncertainty, uncertainty=uncertainty)
    init_network_weights(model, init_type='kaiming')
    return model

@BACKBONE_REGISTRY.register()
def cnn_digitsdg_correlated_uncertainty(uncertainty=0.0, **kwargs):
    """
    This architecture was used for DigitsDG dataset in:

        - Zhou et al. Deep Domain-Adversarial Image Generation
        for Domain Generalisation. AAAI 2020.
    """
    model = CUConvNet(c_hidden=64, pertubration_list=['layer0', 'layer1', 'layer2', 'layer3'], uncertainty=uncertainty, alpha=0.1)
    init_network_weights(model, init_type='kaiming')
    return model

@BACKBONE_REGISTRY.register()
def cnn_digitsdg_conststyle(cfg=None, **kwargs):
    """
    This architecture was used for DigitsDG dataset in:

        - Zhou et al. Deep Domain-Adversarial Image Generation
        for Domain Generalisation. AAAI 2020.
    """
    model = ConstConvNet(c_hidden=64, cfg=cfg)
    init_network_weights(model, init_type='kaiming')
    return model