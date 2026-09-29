<div align="center">

<picture>
  <img alt="BoquilaHUB" src="assets/boquilahub-logo.svg" width="60%">
</picture>

Cross-platform app to run AI models to monitor and protect nature.

<h3>

[Features](#features) | [Installation](#installation)

</h3>

[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-33da72)](LICENSE)

</div>

![readme](assets/readme.jpg)

## Features

- Cross-platform
- GUI, TUI and CLI tool
- Run AIs to process image, video, live feed or audio files
- Deploy and consume REST APIs, with maximum efficiency. Powered by [axum](https://github.com/tokio-rs/axum)
- Scalable, same binary works for IoT systems, simple GUI workflows or multiple GPU servers.

## Installation

Download the latest binaries from [releases](https://github.com/boquila/boquilahub/releases)

## AIs

Download supported models on [website](https://boquila.org/hub). You can also use or port your own, it's super easy.

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
