import torch.nn as nn
from mmcv.runner import load_checkpoint
import torch
import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
import copy
from sklearn.manifold import TSNE
import os

BatchNorm = nn.BatchNorm2d

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


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
    def __init__(self, idx, args, eps=1e-6):
        super().__init__()
        self.idx = idx
        self.args = args
        self.eps = eps
        self.alpha = args.alpha
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
    
    def store_style(self, x):
        mu, var = self.get_style(x)
        self.mean.extend(mu)
        self.std.extend(var)
    
    def cal_mean_std(self, idx, epoch):
        mean_list = np.array(self.mean)
        std_list = np.array(self.std)
        stacked_data = np.stack((mean_list, std_list), axis=1)
        reshaped_data = stacked_data.reshape((len(mean_list), -1))
        
        print(f'Number of cluster: {self.args.num_cluster}')
        self.bayes_cluster = BayesianGaussianMixture(n_components=self.args.num_cluster, covariance_type='full', init_params='k-means++', max_iter=200)
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
        tsne = TSNE(n_components=2, random_state=self.args.seed)
        plot_data = tsne.fit_transform(reshaped_data)
        
        scatter = plt.scatter(plot_data[:, 0], plot_data[:, 1], c=domain_list)
        plt.legend(handles=scatter.legend_elements()[0], labels=classes)
        save_path = os.path.join(f'{self.args.output_dir}', f'testing-features_after{idx}_epoch{epoch}.png')
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

    def forward(self, x, store_feature=False, apply_conststyle=False, is_test=False, apply_rate=0.0):
        if store_feature:
            self.store_style(x)
        
        if (not is_test and np.random.random() > self.args.prob) or not apply_conststyle:
            return x
        
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        # style_feats = torch.hstack((mu, sig)).squeeze().detach().cpu()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig
        
        if is_test:
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

class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].

    """

    def __init__(self, p=0.5, eps=1e-6):
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

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, uncertainty=0.0, pos=[]):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        if uncertainty > 0:
            print("USE UNCERTAINY p:{} , pos: {}".format(uncertainty, pos))
            self.perbutation = DistributionUncertainty(p=uncertainty)
        else:
            self.perbutation = torch.nn.Identity()

        self.pos = pos

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if 0 in self.pos:
            x = self.perbutation(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if 1 in self.pos:
            x = self.perbutation(x)
        x = self.layer1(x)
        if 2 in self.pos:
            x = self.perbutation(x)
        x = self.layer2(x)
        if 3 in self.pos:
            x = self.perbutation(x)
        x = self.layer3(x)
        if 4 in self.pos:
            x = self.perbutation(x)
        x = self.layer4(x)
        if 5 in self.pos:
            x = self.perbutation(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

class CResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, num_conststyle=3, args=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.num_conststyle = num_conststyle
        self.conststyle = [ConstStyle5(i, args) for i in range(self.num_conststyle)]

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, store_feature=False, apply_conststyle=False, is_test=False, apply_rate=0.0):
        x = self.conv1(x)
        if 0 in self.pos:
            x = self.perbutation(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.args.num_conststyle >= 1:
            x = self.conststyle[0](x, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test, apply_rate=apply_rate)
        x = self.layer1(x)
        if self.args.num_conststyle >= 2:
            x = self.conststyle[1](x, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test, apply_rate=apply_rate)
        if self.args.num_conststyle >= 3:
            x = self.conststyle[2](x, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test, apply_rate=apply_rate)
        x = self.layer3(x)
        if self.args.num_conststyle >= 4:
            x = self.conststyle[3](x, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test, apply_rate=apply_rate)
        x = self.layer4(x)
        if self.args.num_conststyle >= 5:
            x = self.conststyle[4](x, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test, apply_rate=apply_rate)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

def _resnet(arch, block, layers, pretrained, progress, pretrained_weights, **kwargs):
    model = ResNet(block, layers, **kwargs)
    pretrained_weights = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
    if pretrained:
        load_checkpoint(model, pretrained_weights, map_location='cpu')
    return model

def _cresnet(arch, block, layers, pretrained, progress, pretrained_weights, **kwargs):
    model = CResNet(block, layers, **kwargs)
    pretrained_weights = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
    if pretrained:
        load_checkpoint(model, pretrained_weights, map_location='cpu')
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)

def cresnet101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _cresnet('cresnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-50 32x4d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
