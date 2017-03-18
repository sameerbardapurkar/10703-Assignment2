"""Loss functions."""

import tensorflow as tf
import semver

def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """

    assert y_true.shape[0] == y_pred.shape[0]

    result = np.zeros(y_true.shape)

    for i in range(0, y_true.shape[0]):
      if(fabs(y_true[i] - y_pred[i]) < max_grad):
        result[i] = fabs(y_true[i] - y_pred[i])*fabs(y_true[i] - y_pred[i])
      else:
        result[i] = max_grad * (fabs(y_true[i] - y_pred[i]) - (max_grad/2.0))
    
    return result

def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    
    result = huber_loss(y_true, y_pred, max_grad=1.)

    return np.mean(result)
