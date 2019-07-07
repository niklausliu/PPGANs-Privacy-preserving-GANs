import inspect
import keras
from keras import backend as K


def _get_shape(x):
    if hasattr(x, 'dense_shape'):
        return x.dense_shape

    return K.shape(x)


def add_gradient_noise(BaseOptimizer):
    if not (
        inspect.isclass(BaseOptimizer) and
        issubclass(BaseOptimizer, keras.optimizers.Optimizer)
    ):
        raise ValueError(
            'add_gradient_noise() expects a valid Keras optimizer'
        )

    class NoisyOptimizer(BaseOptimizer):
        def __init__(self, standard_deviation=0.3, **kwargs):
            super(NoisyOptimizer, self).__init__(**kwargs)
            with K.name_scope(self.__class__.__name__):
                self.standard_deviation = K.variable(standard_deviation, name='standard_deviation')

        def get_gradients(self, loss, params):
            grads = super(NoisyOptimizer, self).get_gradients(loss, params)

            t = K.cast(self.iterations, K.dtype(grads[0]))

            grads = [
                grad + K.random_normal(
                    _get_shape(grad),
                    mean=0.0,
                    stddev=self.standard_deviation,
                    dtype=K.dtype(grads[0])
                )
                for grad in grads
            ]

            return grads

        def get_config(self):
            config = {'standard_deviation': float(K.get_value(self.standard_deviation))}
            base_config = super(NoisyOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    NoisyOptimizer.__name__ = 'Noisy{}'.format(BaseOptimizer.__name__)

    return NoisyOptimizer

