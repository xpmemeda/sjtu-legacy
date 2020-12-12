'''
MMD functions implemented in tensorflow.
'''
from __future__ import division

import tensorflow as tf

from tf_ops import dot, sq_sum

import sys

_eps=1e-8

################################################################################
### Quadratic-time MMD with Gaussian RBF kernel

# 跳转“1” K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigmas, wts)
# X, Y: (batch_size, 28*28*1); sigmas=[2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
def _mix_rbf_kernel(X, Y, sigmas, wts=None):
    if wts is None:
        wts = [1] * len(sigmas) # wts=[1, 1, 1, 1, 1,1]

    XX = tf.matmul(X, X, transpose_b=True) # 第二个参数在相乘前，转置！
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)
    # XX,XY,YY: (batch_size, batch_size)

    X_sqnorms = tf.diag_part(XX) # 返回对角线
    Y_sqnorms = tf.diag_part(YY)
    # X_sqnorms, Y_sqnorms: (batch_size, )

    r = lambda x: tf.expand_dims(x, 0) # insert a dimension
    c = lambda x: tf.expand_dims(x, 1)
    '''
    print(c(X_sqnorms)) # (?, 1)
    print(r(X_sqnorms)) # (1, ?)
    sys.exit()
    '''
    K_XX, K_XY, K_YY = 0, 0, 0
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma**2)
        K_XX += wt * tf.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
        K_XY += wt * tf.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
        K_YY += wt * tf.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))

    # K_XX, K_XY, K_YY: (batch_size, batch_size); tf.reduce_sum是一个标量
    return K_XX, K_XY, K_YY, tf.reduce_sum(wts)


def rbf_mmd2(X, Y, sigma=1, biased=True):
    return mix_rbf_mmd2(X, Y, sigmas=[sigma], biased=biased)

# 入口 kernel_loss = mix_rbf_mmd2(G, images, sigmas=bandwidths)
# G,images: (batch_size, 28*28*1); sigmas=[2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
def mix_rbf_mmd2(X, Y, sigmas=(2.0, 5.0, 10.0, 20.0, 40.0, 80.0, ), wts=None, biased=True): # 原始的 sigmas = (1, )
    '''
    print(X) # (?, 50)
    print(Y) # (?, 50)
    sys.exit()
    '''
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigmas, wts)
    '''
    print(K_XX) # (?, ?)
    print(K_XY) # (?, ?)
    print(K_YY) # (?, ?)
    sys.exit()
    '''
    # K_XX, K_XY, K_YY: (batch_size, batch_size); d=6是一个标量
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)


def rbf_mmd2_and_ratio(X, Y, sigma=1, biased=True):
    return mix_rbf_mmd2_and_ratio(X, Y, sigmas=[sigma], biased=biased)


def mix_rbf_mmd2_and_ratio(X, Y, sigmas=(1,), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigmas, wts)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)


################################################################################
### Helper functions to compute variances based on kernel matrices

# 跳转“2”，返回值 return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
# K_XX, K_XY, K_YY: (batch_size, batch_size), d=6, biased=True
def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    '''
    print(K_XX) # (?, ?)
    print(K_XY) # (?, ?)
    print(K_YY) # (?, ?)
    sys.exit()
    '''
    m = tf.cast(tf.shape(K_XX)[0], tf.float32)
    n = tf.cast(tf.shape(K_YY)[0], tf.float32)
    '''
    print(m) # ()
    print(n) # ()
    sys.exit()
    '''
    # m, n 都等于 batch_size， 浮点数类型

    if biased:
        '''
        print(tf.reduce_sum(K_XX)) # ()
        sys.exit()
        '''
        mmd2 = (tf.reduce_sum(K_XX) / (m * m)
              + tf.reduce_sum(K_YY) / (n * n)
              - 2 * tf.reduce_sum(K_XY) / (m * n))
    else:
        if const_diagonal is not False:
            trace_X = m * const_diagonal
            trace_Y = n * const_diagonal
        else:
            trace_X = tf.trace(K_XX)
            trace_Y = tf.trace(K_YY)

        mmd2 = ((tf.reduce_sum(K_XX) - trace_X) / (m * (m - 1))
              + (tf.reduce_sum(K_YY) - trace_Y) / (n * (n - 1))
              - 2 * tf.reduce_sum(K_XY) / (m * n))

    return mmd2


def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False,
                    min_var_est=_eps):
    mmd2, var_est = _mmd2_and_variance(
        K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased)
    ratio = mmd2 / tf.sqrt(tf.maximum(var_est, min_var_est))
    return mmd2, ratio


def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = tf.cast(K_XX.get_shape()[0], tf.float32)  # Assumes X, Y are same shape

    ### Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        const_diagonal = tf.cast(const_diagonal, tf.float32)
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = tf.diag_part(K_XX)
        diag_Y = tf.diag_part(K_YY)

        sum_diag_X = tf.reduce_sum(diag_X)
        sum_diag_Y = tf.reduce_sum(diag_Y)

        sum_diag2_X = sq_sum(diag_X)
        sum_diag2_Y = sq_sum(diag_Y)

    Kt_XX_sums = tf.reduce_sum(K_XX, 1) - diag_X
    Kt_YY_sums = tf.reduce_sum(K_YY, 1) - diag_Y
    K_XY_sums_0 = tf.reduce_sum(K_XY, 0)
    K_XY_sums_1 = tf.reduce_sum(K_XY, 1)

    Kt_XX_sum = tf.reduce_sum(Kt_XX_sums)
    Kt_YY_sum = tf.reduce_sum(Kt_YY_sums)
    K_XY_sum = tf.reduce_sum(K_XY_sums_0)

    Kt_XX_2_sum = sq_sum(K_XX) - sum_diag2_X
    Kt_YY_2_sum = sq_sum(K_YY) - sum_diag2_Y
    K_XY_2_sum  = sq_sum(K_XY)

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
              + (Kt_YY_sum + sum_diag_Y) / (m * m)
              - 2 * K_XY_sum / (m * m))
    else:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * (m-1))
              + (Kt_YY_sum + sum_diag_Y) / (m * (m-1))
              - 2 * K_XY_sum / (m * m))

    var_est = (
          2 / (m**2 * (m-1)**2) * (
              2 * sq_sum(Kt_XX_sums) - Kt_XX_2_sum
            + 2 * sq_sum(Kt_YY_sums) - Kt_YY_2_sum)
        - (4*m-6) / (m**3 * (m-1)**3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4*(m-2) / (m**3 * (m-1)**2) * (
              sq_sum(K_XY_sums_1) + sq_sum(K_XY_sums_0))
        - 4 * (m-3) / (m**3 * (m-1)**2) * K_XY_2_sum
        - (8*m - 12) / (m**5 * (m-1)) * K_XY_sum**2
        + 8 / (m**3 * (m-1)) * (
              1/m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - dot(Kt_XX_sums, K_XY_sums_1)
            - dot(Kt_YY_sums, K_XY_sums_0))
    )

    return mmd2, var_est
