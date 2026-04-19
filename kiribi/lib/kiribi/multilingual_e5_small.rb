# frozen_string_literal: true

require "fileutils"
require "onnxruntime"
require "pathname"
require "rubygems/package"
require "stringio"
require "tokenizers"
require "zlib"

module Kiribi
  class MultilingualE5Small
    ONNX_FILE = "model_qint8_avx512_vnni.onnx"
    FILES = [ONNX_FILE, "tokenizer.json"].freeze
    URL = "https://github.com/matsudai/kiribi-externals/releases/download/intfloat%2Fmultilingual-e5-small%2Fc007d7e/model_qint8_avx512_vnni.tar.gz"

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
        raise Kiribi::ModelNotDownloaded, %(multilingual-e5-small: #{f} missing. Run: Kiribi.download("multilingual-e5-small")) unless File.exist?(path)
      end
      @tokenizer = Tokenizers.from_file(File.join(dest_dir, "tokenizer.json"))
      @onnx_model = OnnxRuntime::Model.new(File.join(dest_dir, ONNX_FILE))
    end

    def embedding_query(input)
      embedding(:query, input)
    end

    def embedding_passage(input)
      embedding(:passage, input)
    end

    def embedding(prefix, input)
      prefix = prefix.to_s
      raise ArgumentError, "prefix must be :query or :passage" unless %w[query passage].include?(prefix)

      encoded = tokenizer.encode("#{prefix}: #{input}")
      batch = {
        input_ids: [encoded.ids],
        attention_mask: [encoded.attention_mask],
        token_type_ids: [[0] * encoded.ids.length]
      }
      outputs = onnx_model.predict(batch)
      last_hidden = outputs["last_hidden_state"][0]
      attentions = encoded.attention_mask

      output_matrix = last_hidden.filter.with_index { |_, i| attentions[i] == 1 }
      valid_tokens = attentions.sum
      output_matrix.transpose.map { it.sum / valid_tokens }
    end
  end
end
