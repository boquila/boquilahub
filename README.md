# BoquilaHUB

Run AI models to monitor and protect nature. Locally, no cloud.

![readme](readme.jpg)

## List of AIs:

| AI Name                           | Description                           |  Input Type   | Available?   |
| --------------------------------- | --------------------------------- | ------------ | ------------ |
|boquilanet-gen | Generic animal detection          | Image       | ✅ |
|boquilanet-cl | Chilean fauna classification    |  Image       |✅  |
|boquilanet-eu | European fauna classification                | Image  |✅   |
|megadetector v5a | MegaDetector (animals, vehicles, people) |  Image  | ✅   |
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

| Runtime   | Description                                                                        | Requirements                                                        | Available? |
|-----------|------------------------------------------------------------------------------------|---------------------------------------------------------------------|------------|
| cpu      | Your average CPU                    | Having a CPU            | ✅|
| cuda      | CUDA execution provider for NVIDIA GPUs (Maxwell 7xx and above)                    | Requires CUDA v11.6+                                               | ✅ |
| tensorrt  | TensorRT execution provider for NVIDIA GPUs (GeForce 9xx series and above)         | Requires CUDA v11.4+ and TensorRT v8.4+                             |On the way |
| openvino  | OpenVINO execution provider for Intel Core CPUs (6th generation and above)        |                                                                     |On the way|
| onednn    | Intel oneDNN execution provider for x86/x64 targets                              |                                                                     |On the way |
| directml  | DirectML execution provider for Windows x86/x64 targets with dedicated GPUs       | Requires DirectX 12 support and dedicated GPUs                     |On the way |
| qnn       | Qualcomm AI Engine Direct SDK execution provider for Qualcomm chipsets            |                                                                     |On the way  |

## Tech stack

We use [Flutter](https://github.com/flutter/flutter) for the UI and [Rust](https://github.com/rust-lang/rust) for the inference pipeline

## Open source?

In the future, we'll be fully open source. Right now, you can see the UI code.