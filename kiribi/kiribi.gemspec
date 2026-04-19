# frozen_string_literal: true

require_relative "lib/kiribi/version"

Gem::Specification.new do |spec|
  spec.name = "kiribi"
  spec.version = Kiribi::VERSION
  spec.authors = ["matsudai"]

  spec.summary = "Easy to use some onnx models (ruri-v3, multilingual-e5, gemma4-e2b). Models are downloaded on demand via Kiribi.download."
  spec.homepage = "https://github.com/matsudai/kiribi"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 3.4.0"

  spec.bindir = "exe"
  spec.executables = ["kiribi"]

  spec.files = Dir.chdir(__dir__) do
    Dir["lib/**/*.rb", "exe/*", "LICENSE.txt", "README.md"].select { |f| File.file?(f) }
  end
  spec.require_paths = ["lib"]

  spec.add_dependency "onnxruntime", ">= 0.10.0"
  spec.add_dependency "tokenizers", ">= 0.6.0"
end
