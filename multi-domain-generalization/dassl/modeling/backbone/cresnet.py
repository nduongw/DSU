import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
import torch
import copy
from .build import BACKBONE_REGISTRY
from .backbone import Backbone
from sklearn.manifold import TSNE
import os
import ot

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)

def wasstertein_distance_two_gaussians(mean_x, mean_y, std_x, std_y):
    distance = torch.sqrt((mean_x - mean_y) ** 2 + std_x ** 2 + std_y ** 2 - 2 * torch.sqrt(std_x * std_y))
    return torch.sum(distance)
    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )

class ConstStyle(nn.Module):
    def __init__(self, cfg, eps=1e-6):
        super().__init__()
        self.cfg = cfg
        self.mean = []
        self.std = []
        self.eps = eps
        self.const_mean = None
        self.const_cov = None
        self.cum_mean = None
        self.cum_std = None
        self.check = False
        self.domain_list = []
        self.scaled_feats = []
        self.cluster_samples = []
        self.factor = 1.0
    
    def clear_memory(self):
        self.mean = []
        self.std = []
        self.domain_list = []
        self.scaled_feats = []
        
    def get_style(self, x):
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        var = var.sqrt()
        mu, var = mu.detach().squeeze().cpu().numpy(), var.detach().squeeze().cpu().numpy()
        
        return mu, var
    
    def store_style(self, x, domains=False):
        mu, var = self.get_style(x)
        self.mean.extend(mu)
        self.std.extend(var)
        if domains:
            self.domain_list.extend([i.item() for i in domains])
    
    def cal_mean_std(self, idx, epoch):
        domain_list = np.array(self.domain_list)
        #clustering
        mean_list = copy.copy(self.mean)
        std_list = copy.copy(self.std)
        mean_list = np.array(mean_list)
        std_list = np.array(std_list)
        stacked_data = np.stack((mean_list, std_list), axis=1)
        reshaped_data = stacked_data.reshape((len(mean_list), -1))
        # pca = PCA(n_components=32)
        # pca_data = pca.fit_transform(reshaped_data)
        pca_data = reshaped_data
        # param_grid = {
        #     "n_components": range(1, 7),
        #     "covariance_type": ["full"],
        # }
        # bayes_cluster = GridSearchCV(
        #     GaussianMixture(init_params='k-means++'), param_grid=param_grid, scoring=gmm_bic_score
        # )
        num_cluster = self.cfg.NUM_CLUSTERS
        bayes_cluster = BayesianGaussianMixture(n_components=num_cluster, covariance_type='full', init_params='k-means++', max_iter=200)
        bayes_cluster.fit(pca_data)
        
        labels = bayes_cluster.predict(pca_data)
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
                cluster_sample_x = [pca_data[x] for x in cluster_samples_idx[i]]
                for j in range(len(cluster_samples_idx)):
                    if i == j:
                        continue
                    else:
                        cluster_sample_y = [pca_data[k] for k in cluster_samples_idx[j]]
                        cluster_sample_x = np.array(cluster_sample_x)
                        cluster_sample_y = np.array(cluster_sample_y)
                        M = ot.dist(cluster_sample_y, cluster_sample_x)
                        a, b = np.ones(len(cluster_sample_y)) / len(cluster_sample_y), np.ones(len(cluster_sample_x)) / len(cluster_sample_x) 
                        cost = ot.emd2(a, b, M)
                        # pwd = ot.sliced.sliced_wasserstein_distance(cluster_sample_y, cluster_sample_x, seed=self.cfg.SEED, n_projections=128)
                        print(f'Cost to move from cluster {j} to cluster {i} is {cost}')
                        total_cost += cost
                print(f'Total cost of cluster {i}: {total_cost}')
                ot_score.append(total_cost)
                        
            idx_val = np.argmin(ot_score)
            print(f'Layer {idx} chooses cluster {unique_labels[idx_val]} with optimal transport cost {ot_score[idx_val]}')
            
            self.const_mean = torch.from_numpy(bayes_cluster.means_[idx_val])
            self.const_cov = torch.from_numpy(bayes_cluster.covariances_[idx_val])
        
    def forward(self, x, store_feature=False, apply_conststyle=False, is_test=False):
        if store_feature:
            self.store_style(x)
        
        if apply_conststyle:
            mu = x.mean(dim=[2, 3], keepdim=True)
            var = x.var(dim=[2, 3], keepdim=True)
            sig = (var + self.eps).sqrt()
            mu, sig = mu.detach(), sig.detach()
            x_normed = (x-mu) / sig
            if is_test:
                const_value = torch.reshape(self.const_mean, (2, -1))
                const_mean = const_value[0].float()
                const_std = const_value[1].float()
                const_mean = torch.reshape(const_mean, (1, const_mean.shape[0], 1, 1)).to('cuda')
                const_std = torch.reshape(const_std, (1, const_std.shape[0], 1, 1)).to('cuda')
                # const_mean = torch.cum_mean.to('cuda')
                # const_std = torch.cum_std.to('cuda')
                # out = x_normed * const_std + const_mean
                # return out
            else:
                generator = torch.distributions.MultivariateNormal(loc=self.const_mean, covariance_matrix = self.const_cov)
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
                # mean = out.mean(dim=[2, 3], keepdim=True)
                # var = out.var(dim=[2, 3], keepdim=True)
                # std = var.sqrt()
                # if not self.check:
                #     self.cum_mean = torch.mean(mean, dim=0, keepdim=True)
                # else:
                #     self.cum_mean = 0.9 * self.cum_mean + 0.1 * torch.mean(mean, dim=0, keepdim=True)
                
                # if not self.check:
                #     self.cum_std = torch.mean(std, dim=0, keepdim=True)
                # else:
                #     self.cum_std = 0.9 * self.cum_std + 0.1 * torch.mean(std, dim=0, keepdim=True)
                # self.check = True
            return out
        else:
            return x
    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class CResNet(Backbone):

    def __init__(
        self, block, layers, cfg, **kwargs
    ):
        self.inplanes = 64
        super().__init__()
        # backbone network
        self.num_conststyle = 3
        self.conststyle = [ConstStyle(cfg) for i in range(self.num_conststyle)]
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion

        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def stylemaps(self, x, store_feature=False, apply_conststyle=False, is_test=False):
        x = self.conv1(x)
        # x = self.conststyle[0](x, store_feature=store_feature, apply_conststyle=apply_conststyle)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conststyle[0](x, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
        x = self.layer1(x)
        x = self.conststyle[1](x, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
        x = self.layer2(x)
        x = self.conststyle[2](x, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
        return x
    
    def featuremaps(self, x, store_feature=False, apply_conststyle=False, is_test=False):
        x = self.conv1(x)
        # x = self.conststyle[0](x, store_feature=store_feature, apply_conststyle=apply_conststyle)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conststyle[0](x, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
        x = self.layer1(x)
        x = self.conststyle[1](x, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
        x = self.layer2(x)
        x = self.conststyle[2](x, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
        x = self.layer3(x)
        return self.layer4(x)

    def forward(self, x, store_feature=False, apply_conststyle=False, is_test=False):
        f = self.featuremaps(x, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
        v = self.global_avgpool(f)
        return v.view(v.size(0), -1)


def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)
    model.load_state_dict(pretrain_dict, strict=False)


"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""
"""
Standard residual networks
"""

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

@BACKBONE_REGISTRY.register()
def cresnet18(pretrained=True, cfg=None, **kwargs):
    model = CResNet(block=BasicBlock, layers=[2, 2, 2, 2], cfg=cfg)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])
    return model

@BACKBONE_REGISTRY.register()
def cresnet50(pretrained=True, cfg=None, **kwargs):
    model = CResNet(block=BasicBlock, layers=[3, 4, 6, 3], cfg=cfg)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model
