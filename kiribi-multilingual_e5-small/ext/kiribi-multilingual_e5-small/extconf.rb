# frozen_string_literal: true

require "fileutils"
require "net/http"
require "pathname"
require "rubygems/package"
require "zlib"

GEM_NAME = "kiribi-multilingual_e5-small"
MODEL_URL = "https://github.com/matsudai/kiribi-externals/releases/download/intfloat%2Fmultilingual-e5-small%2Fc007d7e/model_qint8_avx512_vnni.tar.gz"

TMP_FILEPATH = File.expand_path(File.join(__dir__, "../../lib/#{GEM_NAME}/vendor/tmp/model.tar.gz"))
BUILD_DIRPATH = File.expand_path(File.join(__dir__, "../../lib/#{GEM_NAME}/vendor/build"))

if Dir.exist?(BUILD_DIRPATH)
  puts "#{BUILD_DIRPATH} already exists, skipping download and extraction."
else
  if File.exist?(TMP_FILEPATH)
    puts "#{TMP_FILEPATH} already exists, skipping download."
  else
    redirect_count = 0
    url = MODEL_URL
    loop do
      raise "Too many redirects" if redirect_count >= 10

      resp = Net::HTTP.get_response(URI.parse(url))
      case resp
      when Net::HTTPSuccess
        FileUtils.mkdir_p(File.dirname(TMP_FILEPATH))
        File.binwrite(TMP_FILEPATH, resp.body)
        break
      when Net::HTTPRedirection
        url = resp["Location"]
        redirect_count += 1
      else
        raise "HTTP request failed (status code: #{resp.code})"
      end
    end
  end

  Gem::Package::TarReader.new(Zlib::GzipReader.open(TMP_FILEPATH)) do |archive|
    archive.each do |entry|
      filepath = File.join(BUILD_DIRPATH, *Pathname(entry.full_name).each_filename.to_a[1..])

      if entry.directory?
        FileUtils.mkdir_p(filepath)
      elsif entry.file?
        FileUtils.mkdir_p(File.dirname(filepath))
        File.binwrite(filepath, entry.read)
      end
    end
  end
end

File.write("Makefile", "all install clean:\n\t@echo \"Nothing to do for $(TARGET)\"\n")
FileUtils.rm_f(TMP_FILEPATH)
