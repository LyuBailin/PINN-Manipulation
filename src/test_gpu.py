import tensorflow as tf
import numpy as np
import time
import logging

conda install matplotlib numpy scipy click pydoe tensorflow

def test_gpu_performance():
    """
    测试 GPU 性能和功能是否正常工作
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 检查GPU是否可用
    logger.info(f"GPU Available: {tf.test.is_built_with_cuda()}")
    logger.info(f"Listed GPUs: {tf.config.list_physical_devices('GPU')}")
    
    # 创建更大的矩阵进行测试
    matrix_size = 8000  # 增大矩阵尺寸以更好地测试GPU性能
    logger.info(f"Testing with matrix size: {matrix_size}x{matrix_size}")
    
    # 创建两个随机矩阵
    A = tf.random.normal([matrix_size, matrix_size])
    B = tf.random.normal([matrix_size, matrix_size])
    
    # 预热运行
    logger.info("Warming up...")
    warm_up = tf.matmul(A[:100, :100], B[:100, :100])
    tf.keras.backend.clear_session()  # 清理预热运行的内存
    
    # 在CPU上运行（多次测试取平均）
    logger.info("Testing on CPU...")
    cpu_times = []
    with tf.device('/CPU:0'):
        for i in range(3):
            start_time = time.time()
            cpu_result = tf.matmul(A, B)
            # 强制执行计算
            _ = cpu_result.numpy()
            cpu_times.append(time.time() - start_time)
    cpu_time = np.mean(cpu_times)
    
    # 在GPU上运行（多次测试取平均）
    if tf.test.is_built_with_cuda():
        logger.info("Testing on GPU...")
        gpu_times = []
        with tf.device('/GPU:0'):
            for i in range(3):
                start_time = time.time()
                gpu_result = tf.matmul(A, B)
                # 强制等待GPU完成计算
                _ = gpu_result.numpy()
                gpu_times.append(time.time() - start_time)
            
            # 验证结果是否一致
            diff = tf.reduce_max(tf.abs(cpu_result - gpu_result))
            logger.info(f"Maximum difference between CPU and GPU results: {diff}")
        gpu_time = np.mean(gpu_times)
    else:
        gpu_time = float('inf')
    
    # 打印性能比较
    logger.info("\nPerformance comparison:")
    logger.info(f"Matrix size: {matrix_size}x{matrix_size}")
    logger.info(f"CPU average time: {cpu_time:.4f} seconds")
    if tf.test.is_built_with_cuda():
        logger.info(f"GPU average time: {gpu_time:.4f} seconds")
        logger.info(f"Speedup: {cpu_time/gpu_time:.2f}x")

if __name__ == "__main__":
    print("\nGPU Information:")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("CUDA Visible Devices:", tf.config.get_visible_devices('GPU'))
    print("TensorFlow Built with CUDA:", tf.test.is_built_with_cuda())
    print("\nStarting Performance Test...")
    test_gpu_performance()