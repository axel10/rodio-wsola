# Changelog

## 0.2 - 2026-07-20

### Changes

* Improved `Wsola` parameter validation: invalid, non-positive, `NaN`, and infinite values for speed, playback rates, OLA window size, and search interval are sanitized and clamped to safe ranges.
* The input source's initial channel count and sample rate are now fixed. A change during playback triggers a panic to prevent channel misalignment.
* Removed the invalid `ExactSizeIterator` implementation and adjusted `size_hint`: only the reliable buffered lower bound is reported, while the upper bound is `None`.
* Replaced silence output for playback rates outside the WSOLA range with clamping to the supported range.

### Performance

* Preallocated and reused temporary, search, and output buffers in the real-time audio thread to avoid heap allocations in `next()`.
* Replaced frequent `drain` and `resize` operations with in-place moves and fills.
* Added lazy, amortized input-buffer compaction to reduce memory movement during long playback sessions.

### Tests

* Added tests for parameter sanitization, speed clamping, `size_hint`, channel changes, and long-audio input-buffer compaction.

## 0.1

* Initial WSOLA-based Rodio audio source implementation with playback speed adjustment, multichannel processing, and configurable OLA window and search interval.
