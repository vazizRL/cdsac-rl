import torch
import torch.distributions as distr
import torch.nn.functional as F
import warnings
from torch.autograd.functional import jacobian
from torch.distributions import constraints
from math import prod
from typing import Dict


class MixtureSameFamilyMod(distr.distribution.Distribution):
    arg_constraints: Dict[str, constraints.Constraint] = {}
    has_sample = False

    def __init__(self, mixture_distribution: distr.Categorical, component_distribution: distr.Distribution,
                 validate_args=None):
        """
        - Reimplementation of MixtureSameFamily
        :param mixture_distribution: Distribution for selecting the kernels
        :param component_distribution: Kernel parametrized distribution 
        :param validate_args:
        """
        self._mixture_distribution = mixture_distribution
        self._component_distribution = component_distribution
        if not isinstance(self._mixture_distribution, distr.Categorical):
            raise ValueError('Mixture distribution must be an instance of torch.distributions.Categorical')
        if not isinstance(self._component_distribution, distr.Distribution):
            raise ValueError('Component distribution must be an instance of torch.distribution.Distribution')

        # Check that batch size matches
        md_bs = self._mixture_distribution.batch_shape
        cd_bs = self._component_distribution.batch_shape[:-1]
        for size1, size2 in zip(reversed(md_bs), reversed(cd_bs)):
            if size1 != 1 and size2 != 1 and size1 != size2:
                raise ValueError(f'mixture_distribution.batch_size {md_bs} is not compatible with'
                                 f'component_distribution.batch_size {cd_bs}')

        # Check that the number of mixture components matches
        km = self._mixture_distribution.logits.shape[-1]
        kc = self._component_distribution.batch_shape[-1]
        if km is not None and kc is not None and km != kc:
            raise ValueError(f'mixture_distribution component {km} does not equal '
                             f'component_distribution.batch_shape[-1] {kc}')
        self._n_component = km

        event_shape = self._component_distribution.event_shape
        self._event_ndims = len(event_shape)

        super(MixtureSameFamilyMod, self).__init__(batch_shape=cd_bs,
                                                   event_shape=event_shape,
                                                   validate_args=validate_args)

    def __repr__(self):
        arg_string = f'\n {self.mixture_distribution},\n {self.component_distribution}'
        return 'MixtureSameFamily' + '(' + arg_string + ')'

    @property
    def mixture_distribution(self):
        return self._mixture_distribution

    @property
    def component_distribution(self):
        return self._component_distribution

    @property
    def mean(self):
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)

        return torch.sum(probs * self.component_distribution.mean, dim=-1-self._event_ndims)

    @property
    def variance(self):
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        # Conditional Variance and Mean
        mean_cond_var = torch.sum(probs * self.component_distribution.variance, dim=-1 - self._event_ndims)
        mean_cond_mean = torch.sum(probs * (self.component_distribution.mean - self._pad(self.mean)).pow(2.0),
                                   dim=-1 - self._event_ndims)

        return mean_cond_var, mean_cond_mean

    @constraints.dependent_property
    def support(self):
        """
        - constraints.dependent_property: Decorator that extends @property to act like a `Dependent` constraint when
          called on a class and act like a property when called on an object.
        -
        :return:
        :rtype:
        """
        # FIXME this may have the wrong shape when support contains batched parameters
        return self._component_distribution.support

    def _pad(self, x):
        return x.unsqueeze(-1 - self._event_ndims)

    def _pad_mixture_dimensions(self, x):
        # numel() returns number of elements
        dist_batch_ndims = self.batch_shape.numel()
        cat_batch_ndims = self.mixture_distribution.batch_shape.numel()

        if cat_batch_ndims == 1:
            pad_ndims = 1
        else:
            pad_ndims = dist_batch_ndims - cat_batch_ndims

        xs = x.shape
        x = x.reshape(xs[:-1] + torch.Size(pad_ndims * [1]) + xs[-1:] + torch.Size(self._event_ndims))

        return x

    def cdf(self, x):
        x = self._pad(x)
        # self.component_distribtion: Batched distributions without the categorical. CDF accor. to components
        cdf_x = self.component_distribution.cdf(x)
        mix_prob = self.mixture_distribution.probs

        return torch.sum(cdf_x * mix_prob, dim=-1)

    def log_prob(self, x):
        if self._validate_args:
            self._validate_sample(x)
        x = self._pad(x)
        log_prob_x = self.component_distribution.log_prob(x)    # [S,B,k]
        log_mix_prob = torch.log_softmax(self.mixture_distribution.logits, dim=-1)  # [S, B]

        return torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)

    def expand(self, batch_shape, _isinstance=None):
        batch_shape = torch.Size(batch_shape)
        batch_shape_comp = batch_shape + (self._n_component,)
        new = self._get_checked_instance(MixtureSameFamilyMod, _isinstance)
        new._component_distribution = self._component_distribution.expand(batch_shape_comp)
        new._mixture_distribution  = self.mixture_distribution.expand(batch_shape)
        new._n_component = self._n_component
        new._event_ndims = self._event_ndims
        event_shape = new._component_distribution.event_shape
        super(MixtureSameFamilyMod, new).__init__(batch_shape=batch_shape, event_shape=event_shape,
                                                  validate_args=False)
        new._validate_args = self._validate_args

        return new

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            sample_len = len(sample_shape)
            batch_len = len(self.batch_shape)
            gather_dim = sample_len + batch_len
            es = self.event_shape

            # mixture samples [n, B]
            mix_sample = self.mixture_distribution.sample(sample_shape)
            mix_shape = mix_sample.shape

            # component samples [n, B, k, E]
            comp_samples = self.component_distribution.sample(sample_shape)

            # Gather along k dimension
            mix_sample_reshape = mix_sample.reshape(mix_shape + torch.Size([1] * (len(es) + 1)))
            mix_sample_reshape = mix_sample_reshape.repeat(torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es)

            samples = torch.gather(comp_samples, gather_dim, mix_sample_reshape)

            return samples.squeeze(gather_dim)


class ReMixtureSameFamilyMod(MixtureSameFamilyMod):
    has_rsample = True

    def __init__(self, *args, **kwargs):
        """
        - Adds rsample method to MixtureSameFamilyMod; implements implicit reparameterization (Figurnov et al. 2018)
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        """
        super().__init__(*args, **kwargs)

        if not self._component_distribution.has_rsample:
            raise ValueError('Cannot reparameterize a mixture of non-reparameterized components')

        # NOTE: Not necessary for implicit reparameterization
        if not callable(getattr(self._component_distribution, '_log_cdf', None)):
            warnings.warn(message=('The component distributions do not have numerically stable "_log_cdf", will'
                                   'use torch.log(cdf) instead, which may not be stable. NOTE: This will not affect'
                                   'implicit reparameterization'))

    def distributional_transform(self, x):
        pass

    def rsample(self, sample_shape=torch.Size()):
        pass

    def _log_cdf(self, x):
        x = self._pad(x)
        if callable(getattr(self._component_distribution, '_log_cdf', None)):
            log_cdf_x = self.component_distribution._log_cdf(x)
            



