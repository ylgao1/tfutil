import tensorflow as tf

__all__ = ['saliency_grad']


def saliency_grad(inputs, logits, saliency_class):
    logits_cls = logits[:, saliency_class]
    target = tf.reduce_mean(logits_cls, axis=0)
    grads = tf.gradients(target, [inputs])[0]
    grads /= (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
    return grads
