from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf


# Helper function to enable loss function to be flexibly used for 
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn
def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5:
        return [1, 2, 3]
    # Two dimensional
    # (Batch, H, W, classes)
    elif len(shape) == 4:
        return [1, 2]
    # Exception - Unknown
    else:
        raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


################################
#           Dice loss          #
################################
def dice_loss(delta=0.5, smooth=0.000001):
    """Dice loss originates from Sørensen–Dice coefficient, which is a statistic developed in 1940s to gauge the similarity between two samples.
    
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.5
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """

    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1 - y_pred), axis=axis)
        fp = K.sum((1 - y_true) * y_pred, axis=axis)
        # Calculate Dice score
        dice_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)
        # Average class scores
        dice_loss = K.mean(1 - dice_class)

        return dice_loss

    return loss_function


################################
#       Dice coefficient       #
################################
def dice_coef(y_true, y_pred):
    """The Dice similarity coefficient, also known as the Sørensen–Dice index or simply Dice coefficient, 
    is a statistical tool which measures the similarity between two sets of data.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.5
        0.5 dice_coeff otherwise tversky_index
        By assigning a greater weight to FN, recall is improved leading to a better balance of precision and recall.
        delta > 0.5 improve Recall
        delta < 0.5 improve precision
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """
    delta = 0.5
    smooth = 0.000001
    axis = identify_axis(y_true.get_shape())
    # Calculate true positives (tp), false negatives (fn) and false positives (fp)   
    tp = K.sum(y_true * y_pred, axis=axis)
    fn = K.sum(y_true * (1 - y_pred), axis=axis)
    fp = K.sum((1 - y_true) * y_pred, axis=axis)
    dice_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)
    # Average class scores
    # with softmax dice_class shape [1,2] binary classification  --> return K.mean(dice_class)
    # with sigmoid dice_class shape [1]  --> return dice_class
    return K.mean(dice_class)
 

################################
#          Focal loss          #
################################
def focal_loss(alpha=None, beta=None, gamma_f=2.):
    """Focal loss is used to address the issue of the class imbalance problem. A modulation term applied to the Cross-Entropy loss function.
    Parameters
    ----------
    alpha : float, optional
        controls relative weight of false positives and false negatives. Beta > 0.5 penalises false negatives more than false positives, by default None
    gamma_f : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 2.
    """

    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)

        if alpha is not None:
            alpha_weight = np.array(alpha, dtype=np.float32)
            focal_loss = alpha_weight * K.pow(1 - y_pred, gamma_f) * cross_entropy
        else:
            focal_loss = K.pow(1 - y_pred, gamma_f) * cross_entropy

        focal_loss = K.mean(K.sum(focal_loss, axis=[-1]))
        return focal_loss

    return loss_function
  
  
################################
#          Combo loss          #
################################
def combo_loss(alpha=0.5):
    """Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation
    Link: https://arxiv.org/abs/1805.02798
    Parameters
    ----------
    alpha : float, optional
        controls weighting of dice and cross-entropy loss., by default 0.5
    """

    def loss_function(y_true, y_pred):
        dice = dice_loss()(y_true, y_pred)
        axis = identify_axis(y_true.get_shape())
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        # cross_entropy = K.binary_crossentropy(y_true, y_pred)
        # axis_to_reduce = range(1, K.ndim(cross_entropy))
        # cross_entropy = K.mean(x=cross_entropy, axis=axis_to_reduce)
        cross_entropy = K.mean(K.sum(cross_entropy, axis[-1]))

        # cross_entropy = K.mean(K.sum(cross_entropy, axis=[-1]))
        combo_loss = (alpha * cross_entropy) + ((1 - alpha) * dice)

        return combo_loss

    return loss_function
