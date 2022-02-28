# this code is based on https://github.com/pfnet-research/sngan_projection/blob/master/source/links/conditional_batch_normalization.py


import chainer
from chainer import configuration
from chainer import cuda
from chainer.functions.normalization import batch_normalization
from chainer.utils import argument
from chainer import variable
import chainer.functions as F

import numpy

from source.functions.moving_average_batchnorm import moving_average_batch_normalization


class ConditionalBatchNormalization(chainer.Chain):
    """
    Conditional Batch Normalization
    Args:
        size (int or tuple of ints): Size (or shape) of channel
            dimensions.
        decay (float): Decay rate of moving average. It is used on training.
        eps (float): Epsilon value for numerical stability.
        dtype (numpy.dtype): Type to use in computing.
        use_gamma (bool): If ``True``, use scaling parameter. Otherwise, use
            unit(1) which makes no effect.
        use_beta (bool): If ``True``, use shifting parameter. Otherwise, use
            unit(0) which makes no effect.
    See: `Batch Normalization: Accelerating Deep Network Training by Reducing\
          Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_
    .. seealso::
       :func:`~chainer.functions.batch_normalization`,
       :func:`~chainer.functions.fixed_batch_normalization`
    Attributes:
        gamma (~chainer.Variable): Scaling parameter.
        beta (~chainer.Variable): Shifting parameter.
        avg_mean (numpy.ndarray or cupy.ndarray): Population mean.
        avg_var (numpy.ndarray or cupy.ndarray): Population variance.
        N (int): Count of batches given for fine-tuning.
        decay (float): Decay rate of moving average. It is used on training.
        eps (float): Epsilon value for numerical stability. This value is added
            to the batch variances.
    """

    def __init__(self, size, decay=0.9, eps=2e-5, dtype=numpy.float32, use_moving_average=False):
        super(ConditionalBatchNormalization, self).__init__()
        self.avg_mean = numpy.zeros(size, dtype=dtype)
        self.register_persistent('avg_mean')
        self.avg_var = numpy.zeros(size, dtype=dtype)
        self.register_persistent('avg_var')
        self.N = 0
        self.register_persistent('N')
        self.decay = decay
        self.eps = eps
        self.use_moving_average = use_moving_average

    def __call__(self, x, gamma, beta, **kwargs):
        """__call__(self, x, c, finetune=False)
        Invokes the forward propagation of BatchNormalization.
        In training mode, the BatchNormalization computes moving averages of
        mean and variance for evaluatino during training, and normalizes the
        input using batch statistics.
        .. warning::
           ``test`` argument is not supported anymore since v2.
           Instead, use ``chainer.using_config('train', train)``.
           See :func:`chainer.using_config`.
        Args:
            x (Variable): Input variable.
            gamma (Variable): Input variable of gamma of shape
            finetune (bool): If it is in the training mode and ``finetune`` is
                ``True``, BatchNormalization runs in fine-tuning mode; it
                accumulates the input array to compute population statistics
                for normalization, and normalizes the input using batch
                statistics.
        """
        argument.check_unexpected_kwargs(
            kwargs, test='test argument is not supported anymore. '
                         'Use chainer.using_config')
        finetune, = argument.parse_kwargs(kwargs, ('finetune', False))
        with cuda.get_device_from_id(self._device_id):
            _gamma = variable.Variable(self.xp.ones(
                self.avg_mean.shape, dtype=x.dtype))
        with cuda.get_device_from_id(self._device_id):
            _beta = variable.Variable(self.xp.zeros(
                self.avg_mean.shape, dtype=x.dtype))
        if configuration.config.train:
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay
            if self.use_moving_average:
                ret = moving_average_batch_normalization(x, _gamma, _beta, eps=self.eps, running_mean=self.avg_mean,
                                                         running_var=self.avg_var, decay=decay)
            else:
                ret = chainer.functions.batch_normalization(x, _gamma, _beta, eps=self.eps, running_mean=self.avg_mean,
                                                            running_var=self.avg_var, decay=decay)
        else:
            # Use running average statistics or fine-tuned statistics.
            mean = variable.Variable(self.avg_mean)
            var = variable.Variable(self.avg_var)
            ret = batch_normalization.fixed_batch_normalization(
                x, _gamma, _beta, mean, var, self.eps)
        shape = ret.shape
        ndim = len(shape)
        gamma = F.broadcast_to(F.reshape(gamma, list(gamma.shape) + [1] * (ndim - len(gamma.shape))), shape)
        beta = F.broadcast_to(F.reshape(beta, list(beta.shape) + [1] * (ndim - len(beta.shape))), shape)
        self.output = gamma * ret + beta
        return self.output


def start_finetuning(self):
    """Resets the population count for collecting population statistics.
    This method can be skipped if it is the first time to use the
    fine-tuning mode. Otherwise, this method should be called before
    starting the fine-tuning mode again.
    """
    self.N = 0
