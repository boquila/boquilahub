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
| MacOS          | ✅ |
| Android          | On the way |
| Web        | On the way |
| iOS          | Not soon |

## List of Runtimes

| Runtime           | Description                                                                        | Requirements  |
|-------------------|------------------------------------------------------------------------------------|--------------|
| CPU              | Your average CPU                                                                   | Having a CPU |
| NVIDIA CUDA      | CUDA execution provider for NVIDIA GPUs (Maxwell 7xx and above)                    | CUDA v12.8 + cuDNN 9.7 |
| WebGPU | GPU acceleration via the WebGPU API, runs on most devices that support graphics | Having a modern GPU | 
| Remote BoquilaHUB | A BoquilaHUB session in your network with a deployed REST API                     | Having the URL | 

And soon more

## How to compile

```shell
git clone https://github.com/boquila/boquilahub/
cd boquilahub
```

**Windows / Linux**

```shell
cargo xtask fetch
cargo build --release
```

That's it, really simple.

Other options:

**macOS**

```shell
brew install pkg-config x264 nasm
chmod +x .github/macos-cc-shim.sh
export CARGO_TARGET_$(rustc -vV | sed -n 's/host: //p' | tr 'a-z-' 'A-Z_')_LINKER="$PWD/.github/macos-cc-shim.sh"
export PKG_CONFIG_PATH="$(brew --prefix x264)/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
cargo build --release --features ffmpeg-static
```

**Linux (static FFmpeg)**

```shell
sudo apt install pkg-config yasm nasm libx264-dev
cargo build --release --features ffmpeg-static
```
