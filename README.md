# rodio-wsola

[![crates.io](https://img.shields.io/crates/v/rodio-wsola.svg)](https://crates.io/crates/rodio-wsola)
[![license](https://img.shields.io/crates/l/rodio-wsola.svg)](https://github.com/axel10/rodio-wsola/blob/main/LICENSE)

A WSOLA (Waveform Similarity Overlap-Add) implementation for [rodio](https://github.com/RustAudio/rodio) sources to stretch audio time (adjust playback speed) without altering the pitch.

## What is WSOLA?

**WSOLA (Waveform Similarity Overlap-Add)** is an industry-standard algorithm used for time-scale modification of audio signals. When you speed up or slow down audio using naive resampling, the pitch changes (creating a high-pitched "chipmunk" or a low-pitched "slow-motion" effect). WSOLA solves this by cutting the audio into overlapping blocks and aligning them based on waveform similarity before combining (overlapping and adding) them back. This preserves the pitch and maintains natural-sounding speech or music even at significant speed adjustments.

## Features

- **Seamless Integration**: Directly wraps any existing `rodio::Source` to add time-stretching.
- **Convenient Extension Trait**: Import `rodio_wsola::WsolaSourceExt` and call `.wsola(speed)` on any `Source`.
- **Dynamic Speed Adjustment**: Change playback speed on the fly using `.set_speed(speed)`.
- **Customizable Parameters**: Fine-tune the algorithm parameters (window size, search interval, and min/max limits) via `Wsola::with_params` to suit speech, music, or specific performance/latency constraints.

## Installation

Add `rodio-wsola` to your `Cargo.toml` dependencies:

```toml
[dependencies]
rodio = "0.22"
rodio-wsola = "0.1.0"
```

## Usage

### Basic Usage

Use the `WsolaSourceExt` extension trait to easily apply speed adjustments to any `Source`:

```rust
use std::fs::File;
use std::io::BufReader;
use rodio::{Decoder, OutputStream, Sink};
use rodio_wsola::WsolaSourceExt;

fn main() {
    // Get a output stream handle to the default physical sound device
    let (_stream, stream_handle) = OutputStream::try_default().unwrap();
    let sink = Sink::try_new(&stream_handle).unwrap();

    // Load a sound from a file
    let file = BufReader::new(File::open("music.mp3").unwrap());
    let source = Decoder::new(file).unwrap();

    // Apply WSOLA to play the source at 1.5x speed without altering pitch
    let speed_stretched_source = source.wsola(1.5);

    sink.append(speed_stretched_source);
    sink.sleep_until_end();
}
```

### Dynamic Speed Control

You can also adjust the speed dynamically during playback:

```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;
use rodio::{Decoder, OutputStream, Sink};

fn main() {
    let (_stream, stream_handle) = OutputStream::try_default().unwrap();
    
    // Create the WSOLA wrapper directly
    let file = std::fs::File::open("audio.wav").unwrap();
    let source = Decoder::new(std::io::BufReader::new(file)).unwrap();
    
    let mut wsola_source = rodio_wsola::Wsola::new(source, 1.0);
    
    // We can change the speed later using set_speed
    wsola_source.set_speed(1.2);
}
```

### Advanced Configuration

If you need to customize the internal parameters of the WSOLA algorithm, use `Wsola::with_params`:

```rust
use rodio_wsola::Wsola;

let custom_source = Wsola::with_params(
    source,
    1.5,      // playback speed multiplier
    0.25,     // min_playback_rate
    8.0,      // max_playback_rate
    12.0,     // ola_window_size_ms (Overlap-Add window length in ms)
    40.0,     // wsola_search_interval_ms (Search interval size in ms)
);
```

#### Parameter Guidelines:
- **OLA Window Size (`ola_window_size_ms`)**: Usually between 10ms and 20ms. Smaller windows reduce processing latency but can introduce audio artifacts in low-frequency sounds.
- **Search Interval (`wsola_search_interval_ms`)**: Controls the search range for finding overlapping segments. Typically 30ms to 60ms.

## License

Licensed under the Apache License, Version 2.0 (the "License" or [LICENSE](LICENSE)). You may not use this library except in compliance with the License.
