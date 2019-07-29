from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import gpytorch
from gpytorch.kernels.kernel import Kernel

class OUKernel(Kernel):
    def __init__(self, ard_num_dims=None, log_lengthscale_prior=None, eps=1e-6, active_dims=None, batch_size=1):
        super(OUKernel, self).__init__(
            has_lengthscale=True,
            ard_num_dims=ard_num_dims,
            batch_size=batch_size,
            active_dims=active_dims,
            log_lengthscale_prior=log_lengthscale_prior,
            eps=eps,
        )
    def forward(self, x1, x2, **params):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        x1_, x2_ = self._create_input_grid(x1_, x2_, **params)

        diff = (x1_ - x2_).norm(2, dim=-1)
        return diff.div(-2).exp_()