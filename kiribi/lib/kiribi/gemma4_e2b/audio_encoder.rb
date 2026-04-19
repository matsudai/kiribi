# frozen_string_literal: true

require "onnxruntime"

module Kiribi
  module Gemma4E2B
    class AudioEncoder < Base
      FILES = %w[audio_encoder.onnx audio_encoder.onnx_data].freeze

      def initialize(dest_dir)
        FILES.each do |f|
          path = File.join(dest_dir, f)
          raise Kiribi::ModelNotDownloaded, %(gemma4-e2b/audio: #{f} missing. Run: Kiribi.download("gemma4-e2b/audio")) unless File.exist?(path)
        end
        @model = OnnxRuntime::Model.new(File.join(dest_dir, "audio_encoder.onnx"))
      end

      def encode(pcm_samples)
        pcm = pcm_samples.is_a?(String) ? pcm_samples.unpack("e*") : pcm_samples

        frame_length = 320
        hop_length = 160
        fft_length = 512
        num_mels = 128
        mel_floor = 0.001

        window = Array.new(frame_length) { 0.5 - 0.5 * Math.cos(2.0 * Math::PI * it / frame_length) }
        mel_filters = build_mel_filterbank(fft_length / 2 + 1, num_mels, 0.0, 8000.0, 16_000)

        pad_left = frame_length / 2
        padded = Array.new(pad_left, 0.0) + pcm
        mask_raw = Array.new(pad_left, false) + Array.new(pcm.length, true)

        frame_size = frame_length + 1
        num_frames = (padded.length - frame_size) / hop_length + 1

        input_features = []
        input_features_mask = []

        num_frames.times do |fi|
          start = fi * hop_length
          windowed = frame_length.times.map { padded[start + it] * window[it] }

          mag = rfft_magnitude(windowed, fft_length)

          mel = num_mels.times.map do |m|
            sum = 0.0
            mag.each_with_index { |v, i| sum += v * mel_filters[i][m] }
            Math.log(sum + mel_floor)
          end

          end_idx = fi * hop_length + frame_size - 1
          valid = end_idx < mask_raw.length && mask_raw[end_idx]

          input_features << (valid ? mel : Array.new(num_mels, 0.0))
          input_features_mask << valid
        end

        padded_frames = ((input_features.length + 127) / 128) * 128
        while input_features.length < padded_frames
          input_features << Array.new(num_mels, 0.0)
          input_features_mask << false
        end

        @model.predict({
          "input_features" => [input_features],
          "input_features_mask" => [input_features_mask]
        })["audio_features"]
      end

      private

      def build_mel_filterbank(num_fft_bins, num_mel_filters, min_freq, max_freq, sample_rate)
        fft_freqs = (0...num_fft_bins).map { it.to_f * sample_rate / ((num_fft_bins - 1) * 2) }
        mel_min = 2595.0 * Math.log10(1.0 + min_freq / 700.0)
        mel_max = 2595.0 * Math.log10(1.0 + max_freq / 700.0)
        mel_points = (0..num_mel_filters + 1).map { mel_min + it * (mel_max - mel_min) / (num_mel_filters + 1) }
        hz_points = mel_points.map { 700.0 * (10.0**(it / 2595.0) - 1.0) }

        filters = Array.new(num_fft_bins) { Array.new(num_mel_filters, 0.0) }
        num_mel_filters.times do |m|
          lower = hz_points[m]
          center = hz_points[m + 1]
          upper = hz_points[m + 2]
          fft_freqs.each_with_index do |f, i|
            if f >= lower && f <= center && center > lower
              filters[i][m] = (f - lower) / (center - lower)
            elsif f > center && f <= upper && upper > center
              filters[i][m] = (upper - f) / (upper - center)
            end
          end
        end
        filters
      end

      def rfft_magnitude(real_signal, n)
        padded = Array.new(n, 0.0)
        real_signal.each_with_index { |v, i| padded[i] = v if i < n }
        imag = Array.new(n, 0.0)
        r, i = fft(padded, imag)
        bins = n / 2 + 1
        bins.times.map { Math.sqrt(r[it]**2 + i[it]**2) }
      end

      def fft(x_real, x_imag)
        n = x_real.length
        return [x_real.dup, x_imag.dup] if n <= 1

        even_r, even_i = fft(
          (0...n / 2).map { x_real[it * 2] },
          (0...n / 2).map { x_imag[it * 2] }
        )
        odd_r, odd_i = fft(
          (0...n / 2).map { x_real[it * 2 + 1] },
          (0...n / 2).map { x_imag[it * 2 + 1] }
        )

        result_r = Array.new(n)
        result_i = Array.new(n)
        half = n / 2
        half.times do |k|
          angle = -2.0 * Math::PI * k / n
          tr = Math.cos(angle) * odd_r[k] - Math.sin(angle) * odd_i[k]
          ti = Math.sin(angle) * odd_r[k] + Math.cos(angle) * odd_i[k]
          result_r[k] = even_r[k] + tr
          result_i[k] = even_i[k] + ti
          result_r[k + half] = even_r[k] - tr
          result_i[k + half] = even_i[k] - ti
        end
        [result_r, result_i]
      end
    end
  end
end
