# frozen_string_literal: true

require "fileutils"
require "net/http"

GEM_NAME = "kiribi-gemma4_e2b"
HF_REPO = "matsudai17/gemma-4-E2B-it-ONNX"
HF_BASE_URL = "https://huggingface.co/#{HF_REPO}/resolve/main/onnx"

MODEL_FILES = %w[
  embed_tokens.onnx
  embed_tokens.onnx_data
  embed_tokens.onnx_data_1
  decoder_model_merged.onnx
  decoder_model_merged.onnx_data
  decoder_model_merged.onnx_data_1
  decoder_model_merged.onnx_data_2
  decoder_model_merged.onnx_data_3
  decoder_model_merged.onnx_data_4
  vision_encoder.onnx
  vision_encoder.onnx_data
  audio_encoder.onnx
  audio_encoder.onnx_data
]
TOKENIZER_FILE = "tokenizer.json"
TOKENIZER_URL = "https://huggingface.co/#{HF_REPO}/resolve/main/#{TOKENIZER_FILE}"

BUILD_DIRPATH = File.expand_path(File.join(__dir__, "../../lib/#{GEM_NAME}/vendor/build"))

def download_file(url, dest)
  redirect_count = 0
  loop do
    raise "Too many redirects" if redirect_count >= 10

    uri = URI.parse(url)
    http = Net::HTTP.new(uri.host, uri.port)
    http.use_ssl = (uri.scheme == "https")
    request = Net::HTTP::Get.new(uri.request_uri)

    http.request(request) do |resp|
      case resp
      when Net::HTTPSuccess
        FileUtils.mkdir_p(File.dirname(dest))
        File.open(dest, "wb") do |f|
          resp.read_body { |chunk| f.write(chunk) }
        end
        return
      when Net::HTTPRedirection
        url = resp["Location"]
        redirect_count += 1
      else
        raise "HTTP request failed for #{url} (status code: #{resp.code})"
      end
    end
  end
end

if Dir.exist?(BUILD_DIRPATH)
  puts "#{BUILD_DIRPATH} already exists, skipping download."
else
  FileUtils.mkdir_p(BUILD_DIRPATH)

  # Download model files
  MODEL_FILES.each do |filename|
    dest = File.join(BUILD_DIRPATH, filename)
    if File.exist?(dest)
      puts "#{filename} already exists, skipping."
    else
      puts "Downloading #{filename}..."
      download_file("#{HF_BASE_URL}/#{filename}", dest)
      puts "  -> #{dest}"
    end
  end

  # Download tokenizer
  tokenizer_dest = File.join(BUILD_DIRPATH, TOKENIZER_FILE)
  unless File.exist?(tokenizer_dest)
    puts "Downloading #{TOKENIZER_FILE}..."
    download_file(TOKENIZER_URL, tokenizer_dest)
    puts "  -> #{tokenizer_dest}"
  end
end

File.write("Makefile", "all install clean:\n\t@echo \"Nothing to do for $(TARGET)\"\n")
