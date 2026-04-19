# frozen_string_literal: true

require "fileutils"
require "onnxruntime"
require "pathname"
require "rubygems/package"
require "stringio"
require "tokenizers"
require "zlib"

module Kiribi
  class RuriV3Small
    FILES = %w[model.onnx tokenizer.json].freeze
    URL = "https://github.com/matsudai/kiribi-externals/releases/download/sirasagi62%2Fruri-v3-30m-ONNX%2Fcdf9391/model.tar.gz"

    def self.download(dest_dir, force: false)
      return if !force && FILES.all? { |f| File.exist?(File.join(dest_dir, f)) }
      FileUtils.rm_rf(dest_dir) if force
      FileUtils.mkdir_p(dest_dir)

      io = StringIO.new
      Kiribi.http_get(URL) { |chunk| io.write(chunk) }
      io.rewind

      Gem::Package::TarReader.new(Zlib::GzipReader.new(io)) do |tar|
        tar.each do |entry|
          next unless entry.file?
          name = Pathname(entry.full_name).each_filename.to_a[1..].join("/")
          next unless FILES.include?(name)
          path = File.join(dest_dir, name)
          FileUtils.mkdir_p(File.dirname(path))
          File.binwrite(path, entry.read)
        end
      end
    end

    attr_reader :onnx_model, :tokenizer

    def initialize(dest_dir)
      FILES.each do |f|
        path = File.join(dest_dir, f)
        raise Kiribi::ModelNotDownloaded, %(ruri-v3-30m: #{f} missing. Run: Kiribi.download("ruri-v3-30m")) unless File.exist?(path)
      end
      @tokenizer = Tokenizers.from_file(File.join(dest_dir, "tokenizer.json"))
      @onnx_model = OnnxRuntime::Model.new(File.join(dest_dir, "model.onnx"))
    end

    def embedding(text)
      encoded = tokenizer.encode(text)
      batch = {
        input_ids: [encoded.ids],
        attention_mask: [encoded.attention_mask]
      }
      outputs = onnx_model.predict(batch)
      outputs["sentence_embedding"][0]
    end

    def embedding_normalized(text)
      vec = embedding(text)
      norm = Math.sqrt(vec.sum { it * it })
      vec.map { it / norm }
    end
  end
end
