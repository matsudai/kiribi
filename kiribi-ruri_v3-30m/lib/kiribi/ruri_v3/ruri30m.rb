# frozen_string_literal: true

require_relative "ruri30m/version"
require "kiribi"
require "onnxruntime"
require "tokenizers"

module Kiribi
  module RuriV3
    extend Kiribi::Loader

    module Ruri30M
      extend Kiribi::Loader

      TOKENIZER_FILEPATH = File.expand_path(File.join(__dir__, "../../../lib/kiribi-ruri_v3-30m/vendor/build/tokenizer.json"))
      MODEL_FILEPATH = File.expand_path(File.join(__dir__, "../../../lib/kiribi-ruri_v3-30m/vendor/build/model.onnx"))

      class Model
        attr_reader :onnx_model, :tokenizer

        def initialize
          @tokenizer = Tokenizers.from_file(TOKENIZER_FILEPATH)
          @onnx_model = OnnxRuntime::Model.new(MODEL_FILEPATH)
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
          norm = Math.sqrt(vec.sum { |v| v * v })
          vec.map { |v| v / norm }
        end
      end

      def self.instantiate
        Model.new
      end
    end
  end
end

Kiribi.register(Kiribi::RuriV3::Ruri30M, order: 100_200_100)
