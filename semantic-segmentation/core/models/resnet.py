import torch.nn as nn
from mmcv.runner import load_checkpoint
from torchvision.models.utils import load_state_dict_from_url
import torch
import numpy as np

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
        for val in unique_labels:
            print(f'Get samples belong to cluster {val}')
            samples = [reshaped_data[i] for i in range(len(labels)) if labels[i] == val]
            samples_idx = [i for i in range(len(labels)) if labels[i] == val]
            samples = np.stack(samples)
            print(f'Cluster {val} has {len(samples)} samples')
            cluster_samples.append(samples)
            cluster_samples_idx.append(samples_idx)
        
        if self.cfg.CLUSTER == 'llh':
            log_likelihood_score = []
            for cluster_idx, cluster_sample_idx in enumerate(cluster_samples_idx):
                cluster_sample = [pca_data[i] for i in cluster_sample_idx]
                sample_score = bayes_cluster.score_samples(cluster_sample)
                mean_score = np.mean(sample_score)
                print(f'Mean log likelihood of cluster {cluster_idx} is {mean_score}')
                log_likelihood_score.append(mean_score)

            idx_val = np.argmax(log_likelihood_score)
            print(f'Layer {idx} chooses cluster {unique_labels[idx_val]} with log likelihood score {log_likelihood_score[idx_val]}')
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
                # const_value = torch.reshape(self.const_mean, (2, -1))
                # const_mean = const_value[0].float()
                # const_std = const_value[1].float()
                # const_mean = torch.reshape(const_mean, (1, const_mean.shape[0], 1, 1)).to('cuda')
                # const_std = torch.reshape(const_std, (1, const_std.shape[0], 1, 1)).to('cuda')
                const_mean = torch.cum_mean.to('cuda')
                const_std = torch.cum_std.to('cuda')
                out = x_normed * const_std + const_mean
                return out
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
            mean = out.mean(dim=[2, 3], keepdim=True)
            var = out.var(dim=[2, 3], keepdim=True)
            std = var.sqrt()
            if not self.check:
                self.cum_mean = torch.mean(mean, dim=0, keepdim=True)
            else:
                self.cum_mean = 0.9 * self.cum_mean + 0.1 * torch.mean(mean, dim=0, keepdim=True)
            
            if not self.check:
                self.cum_std = torch.mean(std, dim=0, keepdim=True)
            else:
                self.cum_std = 0.9 * self.cum_std + 0.1 * torch.mean(std, dim=0, keepdim=True)
            self.check = True
            return out
        else:
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
                 norm_layer=None, uncertainty=0.0, pos=[], conststyle=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        if uncertainty > 0:
            print("USE UNCERTAINY p:{} , pos: {}".format(uncertainty, pos))
            self.perbutation = DistributionUncertainty(p=uncertainty)
        else:
            self.perbutation = torch.nn.Identity()
        
        if conststyle:
            self.conststyle = [ConstStyle(cfg) for i in pos]

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
        if self.conststyle:
            x = self.conststyle[0](x, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
        x = self.layer1(x)
        if 2 in self.pos:
            x = self.perbutation(x)
        if self.conststyle:
            x = self.conststyle[1](x, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
        x = self.layer2(x)
        if 3 in self.pos:
            x = self.perbutation(x)
        if self.conststyle:
            x = self.conststyle[2](x, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
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


def _resnet(arch, block, layers, pretrained, progress, pretrained_weights, **kwargs):
    model = ResNet(block, layers, **kwargs)
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
