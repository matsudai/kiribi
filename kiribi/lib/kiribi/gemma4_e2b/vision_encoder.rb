# frozen_string_literal: true

require "onnxruntime"

module Kiribi
  module Gemma4E2B
    class VisionEncoder < Base
      FILES = %w[vision_encoder.onnx vision_encoder.onnx_data].freeze

      PATCH_SIZE = 16
      RESCALE_FACTOR = 1.0 / 255
      MAX_SOFT_TOKENS = 280
      POOLING_KERNEL = 3
      MAX_PATCHES = MAX_SOFT_TOKENS * POOLING_KERNEL**2
      SIDE_MULT = POOLING_KERNEL * PATCH_SIZE

      def initialize(dest_dir)
        FILES.each do |f|
          path = File.join(dest_dir, f)
          raise Kiribi::ModelNotDownloaded, %(gemma4-e2b/vision: #{f} missing. Run: Kiribi.download("gemma4-e2b/vision")) unless File.exist?(path)
        end
        @model = OnnxRuntime::Model.new(File.join(dest_dir, "vision_encoder.onnx"))
      end

      def input_size_of(original_width, original_height)
        target_px = MAX_PATCHES * PATCH_SIZE**2
        factor = Math.sqrt(target_px.to_f / (original_height * original_width))

        width = (factor * original_width / SIDE_MULT).floor * SIDE_MULT
        height = (factor * original_height / SIDE_MULT).floor * SIDE_MULT

        if width == 0 && height == 0
          raise "Image too small to resize"
        elsif height == 0
          height = SIDE_MULT
          width = [(original_width / original_height) * SIDE_MULT, MAX_SOFT_TOKENS * SIDE_MULT].min
        elsif width == 0
          width = SIDE_MULT
          height = [(original_height / original_width) * SIDE_MULT, MAX_SOFT_TOKENS * SIDE_MULT].min
        end

        [width, height]
      end

      def encode(blob_rgb, width, height)
        blob = blob_rgb.is_a?(String) ? blob_rgb.unpack("C*") : blob_rgb
        patches_w = width / PATCH_SIZE
        patches_h = height / PATCH_SIZE

        pixel_values = []
        pixel_position_ids = []

        patches_w.times do |col|
          patches_h.times do |row|
            patch = []
            PATCH_SIZE.times do |dy|
              PATCH_SIZE.times do |dx|
                y = row * PATCH_SIZE + dy
                x = col * PATCH_SIZE + dx
                idx = (y * width + x) * 3
                patch << blob[idx] * RESCALE_FACTOR
                patch << blob[idx + 1] * RESCALE_FACTOR
                patch << blob[idx + 2] * RESCALE_FACTOR
              end
            end
            pixel_values << patch
            pixel_position_ids << [col, row]
          end
        end

        while pixel_values.length < MAX_PATCHES
          pixel_values << Array.new(PATCH_SIZE**2 * 3, 0.0)
          pixel_position_ids << [-1, -1]
        end

        @model.predict({
          "pixel_values" => [pixel_values],
          "pixel_position_ids" => [pixel_position_ids]
        })["image_features"]
      end
    end
  end
end
