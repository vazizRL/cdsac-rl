import torch


class LinearQ(torch.nn.Module):
    def __init__(self, n_params, dev):
        super(LinearQ, self).__init__()
        self.device = dev
        self.n_params = n_params
        self.params = torch.nn.Parameter(torch.empty(self.n_params, 1, dtype=torch.float32), requires_grad=False)
        self.init_params()
        self.to(self.device)

    def forward(self, phi_sa):
        """
        - Computes Q^w(s,a)= a^T \nabla_{\theta} \mu_{\theta}(s) w
        :param act_grad: batch_size X m X n
        """
        return torch.squeeze(torch.matmul(phi_sa, self.params), dim=-1)

    def update_params(self, lr, td_err, phi_sa):
        loss_stepped = lr * torch.unsqueeze(td_err, dim=2) * phi_sa
        # delta_w = (td_error.unsqueeze(-1) * phi_sa.squeeze(1)).mean(dim=0)
        # self.params.add_(torch.clamp(loss_stepped.mean(dim=0), -1, 1))
        self.params.add_(loss_stepped.mean(dim=0))

        return loss_stepped.mean()

    def init_params(self):
        # torch.nn.init.uniform(self.params, a=-1, b=1)
        torch.nn.init.xavier_uniform(self.params)


# Sanity tests
if __name__ == '__main__':
    device = 'cuda:0'
    n_parameters = 5000
    critic = LinearQ(n_parameters, device)
    # Construct test matrix
    phi_test = torch.rand((40, n_parameters), dtype=torch.float32).to(device)
    res = critic(phi_test)
    print(res.shape)
    print(res.sum())
