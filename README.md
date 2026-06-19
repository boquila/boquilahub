# BoquilaHUB

Cross-platform app to run AI models to monitor and protect nature. Locally, no cloud.

![readme](assets/readme.jpg)

## Features

- Cross-platform
- GUI, TUI and CLI tool
- Run AIs for computer vision and audio, locally
- Process image, video, live feed or audio files
- Deploy and consume REST APIs, with maximum efficiency. Powered by [axum](https://github.com/tokio-rs/axum)

## Installation

Download the latest binaries from [releases](https://github.com/boquila/boquilahub/releases)

We offer two versions, one with both dependencies (ffmpeg and onnxruntime) and one without, in case you have them in your computer already.

## AIs

You can load any [.bq model](https://github.com/boquila/.bq). You can find them on our [website](https://boquila.org/hub).

## List of Platforms

| Platform                           |  Production ready  |
| --------------------------------- |------------ |
| Windows          | ✅ |
| Linux          | ✅ |
| Android          | On the way |
| Web        | On the way |
| MacOS          | Not soon |
| iOS          | Not soon |

## List of Runtimes

| Runtime           | Description                                                                        | Requirements  | Available?   |
|-------------------|------------------------------------------------------------------------------------|--------------|--------------|
| CPU              | Your average CPU                                                                   | Having a CPU | ✅           |
| NVIDIA CUDA      | CUDA execution provider for NVIDIA GPUs (Maxwell 7xx and above)                    | CUDA v12.8 + cuDNN 9.7 | ✅ |
| Remote BoquilaHUB | A BoquilaHUB session in your network with a deployed REST API                     | Having the URL | ✅      |

And soon more

## How to compile

If you want to compile from source just have to

```shell
git clone https://github.com/boquila/boquilahub/
cd boquilahub
cargo build --release
```

Probably instead of cloning from main, you should prefer to get the source code from a tagged version
