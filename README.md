# BoquilaHUB

Run AI models to monitor and protect nature. Locally, no cloud.

![readme](readme.jpg)

## List of AIs:

| AI Name                           | Description                           |  Input Type   | Available?   |
| --------------------------------- | --------------------------------- | ------------ | ------------ |
|boquilanet-gen | Generic animal detection          | Image       | ✅ |
|boquilanet-cl | Chilean fauna classification    |  Image       |✅  |
|boquilanet-eu | European fauna classification                | Image  |✅   |
|megadetector v5 and v6 | MegaDetector (animals, vehicles, people) |  Image  | ✅   |
|boquila-fire | Wildfire detection                | Image  |✅   |
|unnamed | Bird detection |  Audio | On the way |
|unnamed | Chilean birds classification |  Audio | On the way |
|unnamed | Automated marine acoustics |  Audio | On the way |
|unnamed | Tiny LLM |  Text | On the way |

Image = Image files, video files, video feed.

## List of Platforms

| Platform                           |  Production ready  |
| --------------------------------- |------------ |
| Windows          | ✅ |
| Android          | On the way |
| Linux          | On the way |
| MacOS          | Not soon |
| iOS          | Not soon |

## List of Runtimes

| Runtime           | Description                                                                        | Requirements                                                        | Available?   |
|-------------------|------------------------------------------------------------------------------------|---------------------------------------------------------------------|--------------|
| cpu               | Your average CPU                                                                   | Having a CPU                                                        | ✅           |
| NVIDIA CUDA       | CUDA execution provider for NVIDIA GPUs (Maxwell 7xx and above)                    | Requires CUDA v12.4 and cuDNN 8.9.2.26+                             | ✅           |
| NVIDIA TensorRT   | TensorRT execution provider for NVIDIA GPUs (GeForce 9xx series and above)         | Requires CUDA v11.4+ and TensorRT v8.4+                             | On the way   |
| AMD ROCm          | ROCm execution provider for AMD GPUs                                               | Requires ROCm-supported AMD GPUs                                    | On the way   |
| AMD MIGraphX      | MIGraphX execution provider for AMD GPUs                                           | Requires AMD ROCm and MIGraphX                                      | On the way   |
| AMD Vitis AI      | Vitis AI execution provider for Xilinx FPGA devices                                | Requires Xilinx Vitis AI software stack                             | On the way   |
| Intel OpenVINO    | OpenVINO execution provider for Intel Core CPUs (6th generation and above)          |                                                                     | On the way   |
| Intel oneDNN      | Intel oneDNN execution provider for x86/x64 targets                                |                                                                     | On the way   |
| Microsoft DirectML| DirectML execution provider for Windows x86/x64 targets with dedicated GPUs        | Requires DirectX 12 support and dedicated GPUs                      | On the way   |
| Microsoft Azure   | Azure AI execution provider for cloud-based inference on Microsoft Azure           | Requires Azure cloud environment                                    | On the way   |
| Qualcomm QNN      | Qualcomm AI Engine Direct SDK execution provider for Qualcomm chipsets             |                                                                     | On the way   |
| Apple CoreML      | CoreML execution provider for Apple devices                                        | Requires macOS or iOS                                               | On the way   |
| XNNPACK           | XNNPACK execution provider for optimized inference on ARM and x86 devices          | Requires devices with SIMD instruction sets                         | On the way   |
| Huawei CANN       | Huawei CANN execution provider for Huawei Ascend AI processors                     | Requires Huawei Ascend chipsets                                     | On the way   |
| Android NNAPI     | Android NNAPI execution provider for mobile devices with NNAPI support             | Requires Android device with NNAPI                                  | On the way   |
| Apache TVM        | Apache TVM execution provider for multiple backends                                | Requires compilation with TVM                                       | On the way   |
| Arm ACL           | Arm Compute Library (ACL) execution provider for Arm devices                       | Requires devices with ARM processors                                | On the way   |
| ArmNN             | ArmNN execution provider for ARM-based devices                                     | Requires devices with ARM processors                                | On the way   |
| Rockchip RKNPU    | Rockchip RKNPU execution provider for Rockchip NPUs                                | Requires devices with Rockchip NPUs                                 | On the way   |

Requirements are gonna be clearer in the future.

## Tech stack

We use: 

- [Flutter 3.27.1](https://github.com/flutter/flutter) and [Dart 3.6.0](https://github.com/dart-lang/sdk) for the UI  

- [Rust 1.83.0](https://github.com/rust-lang/rust) for the inference pipeline

## Open source?

In the future, we'll be fully open source. Right now, you can see the UI code.