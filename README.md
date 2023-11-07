# UltraFace_tensorrt_cpp详细注释版
UltraFace是一个非常轻量化的人脸识别模型,原项目地址如下:

https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB

近期由于实习工作需要,需要将UltraFace部署在jetson上,并使用显卡加速.虽然有仓库已经将其做了tensorrt部署,但是经过测试感觉效果不好.同时为了后续方便整合到整个项目中,决定自己进行部署.**该仓库与本人的NanoTrace_tensorrt_cpp使用的是类似的结构.对于想在jetson板子上部署检测加追踪的用户们比较方便**.

## Step 1: 获取该仓库
 `git clone https://github.com/ZhangLi1210/UltraFace_Tensorrt_Cpp.git`

## Step 2: 获取UltraFace的ONNX文件
在项目的onnx文件夹下开启终端,并执行:
```wget https://bj.bcebos.com/paddlehub/fastdeploy/version-RFB-320.onnx```

## Step 3: 使用netron导入ONNX文件查看模型架构:
**netron网址**: https://netron.app 进入网址导入模型即可查看模型的输入输出.
>***输入张量:*** 
>个数为1, 大小为: 1X3X240X320. 
>***输出张量:*** 
>共两个输出张量,分别为: 
>boxes: 1X4420X4 
>scores: 1X4420X2 

## Step 4 onnx to Engine:
### 1. Amd64 ubuntu:
1. 请确保已经成功安装了Nvidia显卡驱动,CUDA,与tensorrt.找到安装好的tensorrt的trtexec, trtexec是tensorrt提供的ONNX转engine模型的工具,例如我的trtexec在: 
```/home/lizhang/TensorRT-8.5.2.2/bin/trtexec```

2. 替换 **create_trt_engine.sh** 中的**trtexec**路径 , 替换位置在**create_trt_engine.sh**中的 第6行,将该路径替换成你的路径:
```TRT_BIN="/home/lizhang/TensorRT-8.5.2.2/bin/trtexec"```

3. 在该项目目录下打开终端,并运行:
    `sudo chmod 777 create_trt_engine.sh`
   
    `./create_trt_engine.sh`
   
    程序将开始生成engine文件,该文件生成路径为./engine,此过程将花费一定时间,请耐心等待.

### 2. Arm64 ubuntu(like jetson):
1. Jetson如果安装了jetpack,通常会自带Opencv,CUDA和tensoort, 替换 **create_trt_engine.sh** 中的**trtexec**路径 , 替换位置在**create_trt_engine.sh**中的 第6行,Jetson中的trtexec路径通常在 **/usr/src/tensorrt/bin/trtexec** :
```TRT_BIN="/usr/src/tensorrt/bin/trtexec"```

2. 在该项目目录下打开终端,并运行:

    `sudo chmod 777 create_trt_engine.sh`
   
    `./create_trt_engine.sh`
   
    程序将开始生成engine文件,该文件生成路径为./engine,此过程将花费一定时间,请耐心等待.

## Step 5  编辑 cmakelist.txt. 

 Cmakelist.txt 中需要找到你自己的OpenCV和tensorrt,文件中包含注释,按照注释编辑你自己的cmakelist.txt即可.
 


## Step 6 编译与运行. 
`cd build`

`cmake ..`

`make -j4`

编译成功后, 使用 **./Ultra_Face_trt** 运行例程,程序将尝试获取你的摄像头并进行人脸识别.

# 项目代码是作者边学习边尝试构建的,代码中含有非常详细的注释,欢迎大家一起学习指正.
