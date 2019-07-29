from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
from gpytorch.kernels.kernel import Kernel

class RQKernel(Kernel):
    def __init__(self, ard_num_dims=None, log_lengthscale_prior=None, eps=1e-6, active_dims=None, batch_size=1, log_alpha_prior=None):
        super(RQKernel, self).__init__(
            has_lengthscale=True,
            ard_num_dims=ard_num_dims,
            batch_size=batch_size,
            active_dims=active_dims,
            log_lengthscale_prior=log_lengthscale_prior,
            eps=eps,
        )
        self.register_parameter(
            name="log_alpha",
            parameter=torch.nn.Parameter(torch.zeros(batch_size, 1, 1)),
            prior=log_alpha_prior
        )
    def alpha(self):
        return self.log_alpha.exp()

    def forward(self, x1, x2, **params):
        alpha = self.alpha()
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        x1_, x2_ = self._create_input_grid(x1_, x2_, **params)

        diff = (x1_ - x2_).norm(2, dim=-1)
        return (1+diff.pow(2).div_(-2*alpha).exp_()).pow(-alpha)