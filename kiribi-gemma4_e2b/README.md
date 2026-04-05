# kiribi-gemma4_e2b

Google Gemma 4 E2B (2.3B parameters) multimodal model for text, image, and audio.

Based on [onnx-community/gemma-4-E2B-it-ONNX](https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX) (ONNX format, FP32).

**!!CAUTION!! :** This gem downloads ~22GB of model files from HuggingFace during installation. Be mindful of disk space and network bandwidth.

## Installation

```sh
gem install kiribi-gemma4_e2b
```

Model files (~22GB) are downloaded from HuggingFace during installation.

### Requirements

- Ruby >= 3.4.0
- `ffmpeg` / `ffprobe` (for image and audio preprocessing by the caller)

## Usage

### Text generation

```ruby
require "kiribi/gemma4/e2b"

model = Kiribi::Gemma4::E2B.load
model.generate("Hello!")
```

### Multi-turn chat

```ruby
model.chat([
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", content: "What is Ruby?" },
  { role: "model", content: "Ruby is a dynamic programming language." },
  { role: "user", content: "Who created it?" },
])
```

### Image understanding

Preprocessing is the caller's responsibility. Use `ffmpeg`/`ffprobe` to obtain raw RGB pixels:

```ruby
require "kiribi/gemma4/e2b"

model = Kiribi::Gemma4::E2B.load
encoder = model.load_vision_encoder  # loads vision_encoder.onnx

# 1. Get original dimensions
info = IO.popen(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", "photo.png"], &:read)
original_width, original_height = info.strip.split(",").map(&:to_i)

# 2. Compute the size to resize to
input_width, input_height = encoder.input_size_of(original_width, original_height)

# 3. Resize (caller's choice of tool)
blob = IO.popen(["ffmpeg", "-i", "photo.png", "-vf", "scale=#{input_width}:#{input_height}:flags=bicubic", "-f", "rawvideo", "-pix_fmt", "rgb24", "-v", "error", "-"], "rb", &:read)

# 4. Encode
features = encoder.encode(blob, input_width, input_height)

model.chat([
  { role: "user", content: [
    { type: "image", features: },
    { type: "text", text: "What is in this image?" },
  ] },
])
```

### Audio transcription

```ruby
require "kiribi/gemma4/e2b"

model = Kiribi::Gemma4::E2B.load
encoder = model.load_audio_encoder  # loads audio_encoder.onnx

# 1. Decode to 16kHz mono f32le PCM
pcm = IO.popen(["ffmpeg", "-i", "audio.mp3", "-f", "f32le", "-acodec", "pcm_f32le", "-ar", "16000", "-ac", "1", "-", err: "/dev/null"], "rb", &:read)

# 2. Encode
features = encoder.encode(pcm)

model.chat([
  { role: "user", content: [
    { type: "audio", features: },
    { type: "text", text: "Transcribe the following speech segment in its original language." },
  ] },
])
```

## License

This gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).

The model weights are licensed under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) by Google.
