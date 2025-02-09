# BoquilaHUB

Cross- platform app to run AI models to monitor and protect nature. Locally, no cloud.

![readme](readme.jpg)

## Features

- Cross-platform. 
- Run AIs for computer vision locally
- Process image files
- Process video files (BoquilaHUB 0.2)
- Process camera feed (BoquilaHUB 0.2). Powered by [video-rs](https://github.com/oddity-ai/video-rs)
- Deploy Web APIs, with maximum efficiency. Powered by [Axum](https://github.com/tokio-rs/axum)
- Use these Web APIs to delegate processing, in case your computer doesn't have good hardware

## AIs and binaries

Go to [boquila.org/hub](https://boquila.org/hub), download the models you want to use and just put them in your models folder, that's it. The compiled binaries are also there.

You can load any [.bq model](https://github.com/boquila/.bq). Right now, only for object detection. But in the future we will expand the format.

Segmentation code is being rewritten, available soon.

Video files and Feed processing code is being rewritten, available soon.

## List of Platforms

| Platform                           |  Production ready  |
| --------------------------------- |------------ |
| Windows          | ✅ |
| Linux          | On the way |
| Android          | On the way |
| MacOS          | Not soon |
| iOS          | Not soon |

## List of Runtimes

| Runtime           | Description                                                                        | Requirements  | Available?   |
|-------------------|------------------------------------------------------------------------------------|--------------|--------------|
| CPU              | Your average CPU                                                                   | Having a CPU | ✅           |
| NVIDIA CUDA      | CUDA execution provider for NVIDIA GPUs (Maxwell 7xx and above)                    | CUDA v12.8 + cuDNN 9.7 | ✅ |
| NVIDIA TensorRT  | TensorRT execution provider for NVIDIA GPUs (GeForce 9xx series and above)         | 🚧           | 🚀 Soon      |
| AMD ROCm         | ROCm execution provider for AMD GPUs                                               | 🚧           | 🚀 Soon      |
| AMD MIGraphX     | MIGraphX execution provider for AMD GPUs                                           | 🚧           | 🚀 Soon      |
| AMD Vitis AI     | Vitis AI execution provider for Xilinx FPGA devices                                | 🚧           | 🚀 Soon      |
| Intel OpenVINO   | OpenVINO execution provider for Intel Core CPUs (6th gen and above)                | 🚧           | 🚀 Soon      |
| Intel oneDNN     | Intel oneDNN execution provider for x86/x64 targets                                | 🚧           | 🚀 Soon      |
| Microsoft DirectML | DirectML execution provider for Windows x86/x64 targets with dedicated GPUs     | 🚧           | 🚀 Soon      |
| Microsoft Azure  | Azure AI execution provider for cloud-based inference                              | 🚧           | 🚀 Soon      |
| Qualcomm QNN     | Qualcomm AI Engine Direct SDK execution provider for Qualcomm chipsets             | 🚧           | 🚀 Soon      |
| Apple CoreML     | CoreML execution provider for Apple devices                                        | 🚧           | 🚀 Soon      |
| XNNPACK         | XNNPACK execution provider for optimized inference on ARM and x86 devices           | 🚧           | 🚀 Soon      |
| Huawei CANN     | Huawei CANN execution provider for Huawei Ascend AI processors                     | 🚧           | 🚀 Soon      |
| Android NNAPI   | Android NNAPI execution provider for mobile devices with NNAPI support             | 🚧           | 🚀 Soon      |
| Apache TVM      | Apache TVM execution provider for multiple backends                                | 🚧           | 🚀 Soon      |
| Arm ACL        | Arm Compute Library (ACL) execution provider for Arm devices                        | 🚧           | 🚀 Soon      |
| ArmNN          | ArmNN execution provider for ARM-based devices                                     | 🚧           | 🚀 Soon      |
| Rockchip RKNPU | Rockchip RKNPU execution provider for Rockchip NPUs                                | 🚧           | 🚀 Soon      |

🚧 = Requirements TBA  
🚀 Soon = In progress

## Tech stack

We use: 

- [Flutter 3.27.1](https://github.com/flutter/flutter) and [Dart 3.6.0](https://github.com/dart-lang/sdk) for the UI  

- [Rust 1.83.0](https://github.com/rust-lang/rust) for the proccesing pipelines

## How to compile

If you want to compile from source just have to

```shell
cargo install flutter_rust_bridge_codegen 
git clone https://github.com/boquila/boquilahub/
cd boquilahub
flutter_rust_bridge_codegen generate
flutter run --release
```

Probably instead of cloning from main, you should prefer to get the source code from a tagged version