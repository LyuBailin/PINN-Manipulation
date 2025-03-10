import tensorflow as tf

from scipy.constants import g

# 物理参数
m1 = 1.8  # 第一连杆质量
m2 = 5.586  # 第二连杆质量
l1 = 0.18  # 第一连杆长度
l2 = 0.5586  # 第二连杆长度
lc1 = l1/2  # 第一连杆质心位置
lc2 = l2/2  # 第二连杆质心位置

@tf.function
def f(t, x, u):
    M_tf = M(x[1], i_PR90=1)
    k_tf = k(x[2], x[1], x[3])
    q_tf = q(x[0], x[2], x[1], x[3])
    B_tf = B()

    dx12dt = x[2:]

    with tf.device('/CPU:0'):
        # 执行线性求解或其他操作
        # result = tf.linalg.solve(M, B)  # 你的实际计算
        dx34dt = tf.linalg.solve(M_tf, tf.expand_dims(k_tf + q_tf + tf.linalg.matvec(B_tf, u), 1))[:, 0]

    # # 打印各个张量的形状
    # tf.print("M_tf shape:", tf.shape(M_tf))
    # tf.print("k_tf shape:", tf.shape(k_tf))
    # tf.print("q_tf shape:", tf.shape(q_tf))
    # tf.print("B_tf shape:", tf.shape(B_tf))
    # tf.print("u shape:", tf.shape(u))
    
    # # 打印中间结果的形状
    # matvec_result = tf.linalg.matvec(B_tf, u)
    # tf.print("matvec_result shape:", tf.shape(matvec_result))
    
    # sum_result = k_tf + q_tf + matvec_result
    # tf.print("sum_result shape:", tf.shape(sum_result))
    
    # expanded = tf.expand_dims(sum_result, 1)
    # tf.print("expanded shape:", tf.shape(expanded))
    
    # solved = tf.linalg.solve(M_tf, expanded)
    # tf.print("solved shape:", tf.shape(solved))
    
    # dx34dt = solved[:, 0]
    # tf.print("dx34dt shape:", tf.shape(dx34dt))

    dxdt = tf.concat((dx12dt, dx34dt), 0)

    return dxdt


def M(beta, i_PR90=1.):
    """
    Returns mass matrix of the robot for beta.

    :param tf.Tensor beta: tensor from beta value
    :param float i_PR90: motor constant
    :return: tf.Tensor M_tf: mass matrx of the robot
    """
    M_1 = tf.stack([m1*l1**2/3+m2*l1**2+m2*l2**2/3+m2*l1*l2*tf.cos(beta), 
                    m2*l2/3+m2*l1*tf.cos(beta)/2], axis=0)

    M_2 = tf.stack([m2*l2**2/3+m2*l1*tf.cos(beta)/2,
                    m2*l2**2/3], axis=0)

    M_tf = tf.stack([M_1, M_2], axis=1)

    return M_tf

def k(dalpha_dt, beta, dbeta_dt):
    """
    Returns stiffness vector (Coriolis and centrifugal forces) of the robot
    """

    k1 = m2*l1*l2*tf.sin(beta)*dbeta_dt*dalpha_dt
    k2 = 0

    return tf.stack([k1, k2], axis=0)


def q(alpha, dalpha_dt, beta, dbeta_dt):
    """
    Returns reaction forces vector of the robot for a set of generalized coordinates.

    :param tf.Tensor alpha: tensor from alpha values
    :param tf.Tensor dalpha_dt: tensor from values of the first derivation of alpha
    :param tf.Tensor beta: tensor from beta values
    :param tf.Tensor dbeta_dt: tensor from values of the first derivation of beta
    :return: tf.Tensor: reaction forces vectors of the robot
    """

    return tf.stack(
        [m2*l1*tf.sin(beta)*dbeta_dt + (m1/2+m2)*g*l1+tf.sin(alpha) + m2*g*l2*tf.sin(alpha+beta),
         -m2*l1*tf.sin(beta)*dalpha_dt + m2*g*l2*tf.sin(alpha+beta)/2], axis=0)


def B(i_PR90=1):
    """
    Returns input matrix of the robot.

    :param float i_PR90: constant
    :return: tf.Tensor: input matrix of the robot
    """
    i_PR90 = tf.convert_to_tensor(i_PR90, dtype=tf.float64)

    B_1 = tf.stack([i_PR90, 0.0], axis=0)

    B_2 = tf.stack([0.0, i_PR90], axis=0)

    B_tf = tf.stack([B_1, B_2], axis=1)

    return B_tf

def M_tensor(beta, i_PR90):
    """
    Returns mass matrices of the robot for multiple values for beta.

    :param tf.Tensor beta: tensor from beta values
    :param float i_PR90: constant
    :return: tf.Tensor M_tf: mass matrices of the robot
    """

    M_1 = tf.stack([m1*l1**2/3+m2*l1**2*i_PR90+m2*l2**2/3+m2*l1*l2*tf.cos(beta), 
                    m2*l2/3*i_PR90+m2*l1*tf.cos(beta)/2], axis=1)

    M_2 = tf.stack([m2*l2**2/3*i_PR90+m2*l1*tf.cos(beta)/2,
                    m2*l2**2/3*i_PR90], axis=1)

    M_tf = tf.stack([M_1, M_2], axis=2)

    return M_tf


def k_tensor(dalpha_dt, beta, dbeta_dt, i_PR90):
    """
    Returns stiffness vector (Coriolis and centrifugal forces) of the robot
    """
    k1 = m2*l1*l2*tf.sin(beta)*dbeta_dt*dalpha_dt
    k2 = 0*i_PR90

    return tf.stack([k1, k2], axis=1)


def q_tensor(alpha, dalpha_dt, beta, dbeta_dt):
    """
    Returns reaction forces vectors of the robot for multiple values of generalized coordinates.

    :param tf.Tensor alpha: tensor from alpha values
    :param tf.Tensor dalpha_dt: tensor from values of the first derivation of alpha
    :param tf.Tensor beta: tensor from beta values
    :param tf.Tensor dbeta_dt: tensor from values of the first derivation of beta
    :return: tf.Tensor: reaction forces vectors of the robot
    """

    return tf.stack(
        [m2*l1*tf.sin(beta)*dalpha_dt + (m1/2+m2)*g*l1+tf.sin(alpha) + m2*g*l2*tf.sin(alpha+beta),
         -m2*l1*tf.sin(beta)*dbeta_dt + m2*g*l2*tf.sin(alpha+beta)/2], axis=1)


def B_tensor(i_PR90):
    """
    Returns input matrices of the robot.

    :param float i_PR90: constant
    :return: tf.Tensor: input matrices of the robot
    """
    B_1 = tf.stack([i_PR90, tf.zeros(i_PR90.shape, dtype=tf.float64)], axis=1)

    B_2 = tf.stack([tf.zeros(i_PR90.shape, dtype=tf.float64), i_PR90], axis=1)

    B_tf = tf.stack([B_1, B_2], axis=2)

    return B_tf
