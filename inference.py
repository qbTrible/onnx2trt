'''
Description: In User Settings Edit
Author: quanbo
Date: 2022-12-13 16:47:09
LastEditTime: 2022-12-22 13:38:47
LastEditors: quanbo
FilePath: \onnx2trt\inference.py
'''
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host   = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def do_inference(context, bindings, inputs, outputs, stream, batch_size):
    # 将输入数据从主机拷贝到设备
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # 推理 
    context.set_binding_shape(0, inputs[0].host.shape)
    test_num = 100
    t1 = time.time()
    for i in range(test_num):
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
        # context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    t2 = time.time()
    avg_time = (t2 - t1) / test_num
    print("avg_time: ", avg_time)
    # 将输出数据从设备拷贝到主机
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # 同步流
    stream.synchronize()
    # 仅返回主机上的输出
    return [out.host for out in outputs]

def allocate_buffers(engine, max_batch):
    inputs   = []
    outputs  = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        # size  = trt.volume(
        #     engine.get_binding_shape(binding)) * engine.max_batch_size
        size = trt.volume(engine.get_binding_shape(binding)[1:]) * max_batch
        dtype = trt.nptype(
            engine.get_binding_dtype(binding))
        # 分配主机内存和设备内存
        host_mem   = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # 绑定设备内存
        bindings.append(int(device_mem))
        # 输入输出绑定
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream 

# 1. 确定batch size大小，与导出的trt模型保持一致
BATCH_SIZE = 1

# 3. 创建Runtime，加载TRT引擎
f = open("dynamic_v5Lite-640-fp16.trt", "rb")   
trt.init_libnvinfer_plugins(None, "")                  # 读取trt模型
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))   # 创建一个Runtime(传入记录器Logger)
engine = runtime.deserialize_cuda_engine(f.read())      # 从文件中加载trt引擎
context = engine.create_execution_context()             # 创建context
# print("implicit: ", engine.has_implicit_batch_dimension)
# print("binding_num: ", engine.num_bindings)

# 分配内存
inputs, outputs, bindings, stream = allocate_buffers(engine, BATCH_SIZE)

# output_shapes = [(BATCH_SIZE, 3, 85, 80, 80), (BATCH_SIZE, 3, 85, 40, 40), (BATCH_SIZE, 3, 85, 20, 20)]
output_shapes = []

# 为图像分配主机内存
inputs[0].host = np.random.randn(BATCH_SIZE, 3, 640, 640).astype(np.float32)

for i, binding in enumerate(engine):
    if i > 0:
        output_shapes.append((BATCH_SIZE, *engine.get_binding_shape(binding)[1:]))

# 推理并获得输出
trt_outputs = do_inference(context, bindings, inputs, outputs, stream, BATCH_SIZE)

# 由于 trt_outputs 为展开的张量，这里将其 reshape
trt_outputs = [output.reshape(shape) \
    for output, shape in zip(trt_outputs, output_shapes)]

for trt_output in trt_outputs:
    print(trt_output.shape, np.isnan(trt_output).sum())
