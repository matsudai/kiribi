# frozen_string_literal: true

require "onnxruntime"
require "tokenizers"

module Kiribi
  module Gemma4
    module E2B
      TOKENIZER_FILEPATH = File.expand_path(File.join(__dir__, "../../../kiribi-gemma4_e2b/vendor/build/tokenizer.json"))
      EMBED_MODEL_FILEPATH = File.expand_path(File.join(__dir__, "../../../kiribi-gemma4_e2b/vendor/build/embed_tokens.onnx"))
      DECODER_MODEL_FILEPATH = File.expand_path(File.join(__dir__, "../../../kiribi-gemma4_e2b/vendor/build/decoder_model_merged.onnx"))

      class Model
        EOS_TOKEN_IDS = [1, 106, 50]
        IMAGE_TOKEN_ID = 258_880
        AUDIO_TOKEN_ID = 258_881

        attr_reader :tokenizer

        def initialize
          @tokenizer = Tokenizers.from_file(TOKENIZER_FILEPATH)
          @embed_model = OnnxRuntime::Model.new(EMBED_MODEL_FILEPATH)
          @decoder_model = OnnxRuntime::Model.new(DECODER_MODEL_FILEPATH)

          decoder_sess = OnnxRuntime::InferenceSession.new(DECODER_MODEL_FILEPATH)
          @head_dims = decoder_sess.inputs
            .select { it[:name].match?(/\Apast_key_values\.\d+\.key\z/) }
            .sort_by { it[:name][/\d+/].to_i }
            .map { it[:shape].last }
          @num_layers = @head_dims.length

          @num_logits_to_keep_1 = OnnxRuntime::OrtValue.from_shape_and_type([], :int64)
          @num_logits_to_keep_1.data_ptr.write_int64(1)
        end

        # -------------------------------------------------
        # 遅延ロード（名前で初期化コストを明示）
        # -------------------------------------------------

        def load_vision_encoder
          @vision_encoder ||= VisionEncoder.new
        end

        def load_audio_encoder
          @audio_encoder ||= AudioEncoder.new
        end

        # -------------------------------------------------
        # 低レベル API: ONNX 呼び出しのみ
        # -------------------------------------------------

        def embed(input_ids)
          @embed_model.predict({"input_ids" => [input_ids]})
        end

        def forward(inputs_embeds:, per_layer_inputs:, attention_mask:, position_ids:, past_key_values: nil)
          past_kv = past_key_values || init_kv_cache
          input = {
            "inputs_embeds" => inputs_embeds,
            "attention_mask" => attention_mask,
            "position_ids" => position_ids,
            "num_logits_to_keep" => @num_logits_to_keep_1,
            "per_layer_inputs" => per_layer_inputs,
          }
          input.merge!(past_kv)
          out = @decoder_model.predict(input)

          new_kv = {}
          @num_layers.times do |i|
            new_kv["past_key_values.#{i}.key"] = out["present.#{i}.key"]
            new_kv["past_key_values.#{i}.value"] = out["present.#{i}.value"]
          end

          {logits: out["logits"], past_key_values: new_kv}
        end

        def init_kv_cache
          kv = {}
          @num_layers.times do |i|
            kv["past_key_values.#{i}.key"] = OnnxRuntime::OrtValue.from_shape_and_type([1, 1, 0, @head_dims[i]], :float)
            kv["past_key_values.#{i}.value"] = OnnxRuntime::OrtValue.from_shape_and_type([1, 1, 0, @head_dims[i]], :float)
          end
          kv
        end

        # -------------------------------------------------
        # 高レベル API
        # -------------------------------------------------

        def generate(prompt, max_new_tokens: 256)
          chat([{role: "user", content: prompt}], max_new_tokens:)
        end

        def chat(messages, max_new_tokens: 256)
          prompt_parts = ["<bos>"]
          encoded_media = []

          messages.each do |msg|
            role = msg[:role]
            content = msg[:content]
            prompt_parts << "<|turn>#{role}\n"

            if content.is_a?(String)
              prompt_parts << content
            elsif content.is_a?(Array)
              content.each do |part|
                case part[:type]
                when "text"
                  prompt_parts << part[:text]
                when "image"
                  features = part[:features]
                  prompt_parts << "<|image>" + "<|image|>" * features.length + "<image|>\n"
                  encoded_media << {token_id: IMAGE_TOKEN_ID, features:}
                when "audio"
                  features = part[:features]
                  prompt_parts << "<|audio>" + "<|audio|>" * features.length + "<audio|>\n"
                  encoded_media << {token_id: AUDIO_TOKEN_ID, features:}
                end
              end
            end

            prompt_parts << "<turn|>\n"
          end
          prompt_parts << "<|turn>model\n"

          input_ids = tokenizer.encode(prompt_parts.join).ids

          embeds = []
          encoded_media.each do |media|
            positions = input_ids.each_with_index
              .select { |t, _| t == media[:token_id] }
              .map(&:last)
              .reject { |pos| embeds.any? { it[:pos] == pos } }
            media[:features].each_with_index do |feat, idx|
              break if idx >= positions.length
              embeds << {pos: positions[idx], feat:}
            end
          end

          past_kv = nil
          generated = []

          max_new_tokens.times do |step|
            cur_ids = step == 0 ? input_ids : [generated.last]
            seq_len = cur_ids.length
            total_len = input_ids.length + generated.length

            embed_out = embed(cur_ids)
            inputs_embeds = embed_out["inputs_embeds"]
            per_layer_inputs = embed_out["per_layer_inputs"]

            if step == 0
              embeds.each { inputs_embeds[0][it[:pos]] = it[:feat] }
            end

            result = forward(
              inputs_embeds:,
              per_layer_inputs:,
              attention_mask: [Array.new(total_len, 1)],
              position_ids: [(total_len - seq_len...total_len).to_a],
              past_key_values: past_kv,
            )
            past_kv = result[:past_key_values]

            next_token = result[:logits][0][-1].each_with_index.max_by { |v, _| v }[1]
            break if EOS_TOKEN_IDS.include?(next_token)
            generated << next_token
          end

          tokenizer.decode(generated)
        end
      end
    end
  end
end
