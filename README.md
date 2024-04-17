# BoquilaHUB

Run AI models to monitor and protect nature. Locally, no cloud.

![readme](readme.jpg)

## List of AIs:

| AI Name                           |  Input Type   | Available?   |
| --------------------------------- | ------------ | ------------ |
| Generic animal detection          | Image       | ✅ |
| Chilean fauna classification    |  Image       |✅  |
| European fauna classification                | Image  |✅   |
| MegaDetector (animals, vehicles, people) |  Image  | ✅   |
| Wildfire detection                | Image  |✅   |
| Bird detection |  Audio | On the way |
| Chilean birds classification |  Audio | On the way |
| Automated marine acoustics |  Audio | On the way |

Image = Image files, video files, video feed.

Audio = .wav, .mp3

## Tech stack

We use [Flutter](https://github.com/flutter/flutter) for the UI and [Rust](https://github.com/rust-lang/rust) for the inference pipeline

## Open source?

In the future, we'll be fully open source. Right now, you can see the UI code.

## List of Runtimes

We're just starting, but these are the goals:

| Runtime   | Description                                                                        | Requirements                                                        | Available? |
|-----------|------------------------------------------------------------------------------------|---------------------------------------------------------------------|------------|
| cpu      | Your average CPU                    | Having a CPU            | ✅|
| cuda      | CUDA execution provider for NVIDIA GPUs (Maxwell 7xx and above)                    | Requires CUDA v11.6+                                               | On the way  |
| tensorrt  | TensorRT execution provider for NVIDIA GPUs (GeForce 9xx series and above)         | Requires CUDA v11.4+ and TensorRT v8.4+                             |On the way |
| openvino  | OpenVINO execution provider for Intel Core CPUs (6th generation and above)        |                                                                     |On the way|
| onednn    | Intel oneDNN execution provider for x86/x64 targets                              |                                                                     |On the way |
| directml  | DirectML execution provider for Windows x86/x64 targets with dedicated GPUs       | Requires DirectX 12 support and dedicated GPUs                     |On the way |
| qnn       | Qualcomm AI Engine Direct SDK execution provider for Qualcomm chipsets            |                                                                     |On the way  |


