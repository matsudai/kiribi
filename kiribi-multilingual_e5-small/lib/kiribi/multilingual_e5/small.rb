# frozen_string_literal: true

require_relative "small/version"
require "kiribi"
require "onnxruntime"
require "tokenizers"

module Kiribi
  module MultilingualE5
    extend Kiribi::Loader

    module Small
      extend Kiribi::Loader

      TOKENIZER_FILEPATH = File.expand_path(File.join(__dir__, "../../../lib/kiribi-multilingual_e5-small/vendor/build/tokenizer.json"))
      MODEL_FILEPATH = File.expand_path(File.join(__dir__, "../../../lib/kiribi-multilingual_e5-small/vendor/build/model_qint8_avx512_vnni.onnx"))

      class Model
        attr_reader :onnx_model, :tokenizer

        def initialize
          @tokenizer = Tokenizers.from_file(TOKENIZER_FILEPATH)
          @onnx_model = OnnxRuntime::Model.new(MODEL_FILEPATH)
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

          # https://huggingface.co/intfloat/multilingual-e5-small
          encoded = tokenizer.encode("#{prefix}: #{input}")
          batch = {
            input_ids: [encoded.ids],
            attention_mask: [encoded.attention_mask],
            token_type_ids: [[0] * encoded.ids.length]
          }
          outputs = onnx_model.predict(batch)
          last_hidden = outputs["last_hidden_state"][0]
          attentions = encoded.attention_mask

          output_matrix = last_hidden.filter.with_index {  |_, i| attentions[i] == 1 }
          valid_tokens = attentions.sum
          output_matrix.transpose.map { |v| v.sum / valid_tokens }
        end
      end

      def self.instantiate
        Model.new
      end
    end
  end
end

Kiribi.register(Kiribi::MultilingualE5::Small, order: 100_100_100)
