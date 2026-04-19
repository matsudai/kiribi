# frozen_string_literal: true

require "fileutils"
require "net/http"
require "uri"

require_relative "kiribi/version"

module Kiribi
  class Error < StandardError; end

  class UnknownModel < Error; end

  class ModelNotDownloaded < Error; end

  class DownloadFailed < Error; end

  MODELS = %w[
    ruri-v3-30m
    multilingual-e5-small
    gemma4-e2b/text
    gemma4-e2b/vision
    gemma4-e2b/audio
  ].freeze

  class << self
    attr_writer :cache_dir

    def cache_dir
      @cache_dir ||
        ENV["KIRIBI_CACHE_DIR"] ||
        File.join(ENV["XDG_CACHE_HOME"] || File.join(Dir.home, ".cache"), "kiribi")
    end

    def download(name, force: false)
      lookup(name).download(File.join(cache_dir, name), force: force)
      true
    end

    def load(name)
      lookup(name).new(File.join(cache_dir, name))
    end

    def http_get(url, &block)
      redirect_count = 0
      loop do
        raise DownloadFailed, "Too many redirects" if redirect_count >= 10
        uri = URI.parse(url)
        http = Net::HTTP.new(uri.host, uri.port)
        http.use_ssl = (uri.scheme == "https")
        http.request(Net::HTTP::Get.new(uri.request_uri)) do |resp|
          case resp
          when Net::HTTPSuccess
            resp.read_body(&block)
            return
          when Net::HTTPRedirection
            url = resp["Location"]
            redirect_count += 1
          else
            raise DownloadFailed, "HTTP #{resp.code} for #{url}"
          end
        end
      end
    end

    private

    def lookup(name)
      case name
      when "ruri-v3-30m"           then RuriV3Small
      when "multilingual-e5-small" then MultilingualE5Small
      when "gemma4-e2b/text"       then Gemma4E2B::Text
      when "gemma4-e2b/vision"     then Gemma4E2B::VisionEncoder
      when "gemma4-e2b/audio"      then Gemma4E2B::AudioEncoder
      else raise UnknownModel, "Unknown model: #{name.inspect}"
      end
    end
  end
end

require_relative "kiribi/ruri_v3_small"
require_relative "kiribi/multilingual_e5_small"
require_relative "kiribi/gemma4_e2b"
