# frozen_string_literal: true

require "fileutils"

module Kiribi
  module Gemma4E2B
    HF_REPO = "matsudai17/gemma-4-E2B-it-ONNX"
    BASE_URL = "https://huggingface.co/#{HF_REPO}/resolve/main"

    class Base
      class << self
        def download(dest_dir, force: false)
          return if !force && self::FILES.all? { |f| File.exist?(File.join(dest_dir, f)) }
          FileUtils.rm_rf(dest_dir) if force
          FileUtils.mkdir_p(dest_dir)

          self::FILES.each do |f|
            path = File.join(dest_dir, f)
            File.open(path, "wb") do |io|
              Kiribi.http_get(url_for(f)) { |chunk| io.write(chunk) }
            end
          end
        end

        def url_for(filename)
          "#{BASE_URL}/onnx/#{filename}"
        end
      end
    end
  end
end

require_relative "gemma4_e2b/text"
require_relative "gemma4_e2b/vision_encoder"
require_relative "gemma4_e2b/audio_encoder"
