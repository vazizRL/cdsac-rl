import torch


class TanhGaussDistribution:
    def __init__(self, logits, act_low_lim: tuple, act_up_lim: tuple):
        """
        - Initiates Gaussian distribution but limited in action and log prob according to App. B-C
        - Does it work if actions differ in their lower and upper bounds?
        :param logits: Raw outputs?
        :type logits
        :param act_low_lim: Vector containing lower bounds of a_i
        :param act_up_lim: Vector containing higher bounds of a_i
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.logits = logits
        # self.mean, self.std = torch.chunk(logits, chunks=2, dim=-1)
        self.mean, self.std = logits
        self.gauss_distribution = torch.distributions.Independent(
            base_distribution=torch.distributions.Normal(self.mean, self.std), reinterpreted_batch_ndims=1)
        self.act_low_lim = act_low_lim
        self.act_up_lim = act_up_lim
        # self.act_low_lim = torch.tensor([-1.0]).to(self.device)
        # self.act_up_lim = torch.tensor([1.0]).to(self.device)
        self.num_stab = torch.tensor([1e-6]).to(self.device)

    def sample(self, reparameterization=False):
        """
        - Sample with or without reparameterization trick
        - Limits action and calculates log_prob of limited action
        :param reparameterization: True for objective of policy, False otherwise
        :return: Limited action and tis log_prob
        """
        if reparameterization:
            # Sample action with reparametrization trick
            action = self.gauss_distribution.rsample()
        else:
            action = self.gauss_distribution.sample()
        # Funnel action
        action_limited = ((self.act_up_lim - self.act_low_lim) / 2) * torch.tanh(action) + \
                         (self.act_up_lim + self.act_low_lim) / 2

        # Implementation of restricted action probability, App. B-C
        log_prob_limited = self.gauss_distribution.log_prob(action) - \
            torch.log(1 + self.num_stab - torch.pow(torch.tanh(action), 2)).sum(-1) - \
            torch.log((self.act_up_lim - self.act_low_lim)/2).sum(-1)

        return action_limited, log_prob_limited

    def log_prob(self, action_limited):     # CHECKED
        """
        - Convert bounded action to unbounded and retrieve log_prob based on current distribution
        :param action_limited: Bounded action
        :return: Log probability of unbounded action
        :rtype: torch.tensor
        """
        action = torch.atanh((1-self.num_stab) * (2 * action_limited - (self.act_up_lim + self.act_low_lim)) /
                             (self.act_up_lim - self.act_low_lim))
        log_prob = self.gauss_distribution.log_prob(action) - torch.log((self.act_up_lim - self.act_low_lim) *
                      (1 + self.num_stab - torch.pow(torch.tanh(action), 2))).sum(-1)
        # log_prob = self.gauss_distribution.log_prob(action)
        return log_prob

    def entropy(self):
        """
        - Returns entropy of the distribution
        :return: Policy entropy
        :rtype: torch.tensor
        """
        return self.gauss_distribution.entropy()

    def mode(self):
        """
        - Mode: Value of the term that occurs the most often. Note: Can also be multi-modal
        :return: Mode scaled by lower and uppper action bounds
        :rtype: torch.tensor
        """
        return ((self.act_up_lim - self.act_low_lim) / 2) * torch.tanh(self.mean) + \
               ((self.act_up_lim + self.act_low_lim)/2)

    def kl_divergence(self, other: 'TanhGaussDistribution') -> torch.tensor:
        """
        - Calculates D_{KL} between two instances of this class
        - Note: Original: other: 'GaussDistribution'
        :param other: Other distirbution instance
        :type other: TanhGaussDistribution or GaussDistribution
        :return:
        :rtype:
        """
        return torch.distributions.kl.kl_divergence(self.gauss_distribution, other.gauss_distribution)


if __name__ == '__main__':
    mean = torch.tensor([0])
    std = torch.tensor([0.1])
    logits = torch.cat((mean, std), dim=0)

    distr = TanhGaussDistribution(logits, (-1,), (1,))



