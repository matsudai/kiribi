# frozen_string_literal: true

require_relative "lib/kiribi/version"

Gem::Specification.new do |spec|
  spec.name = "kiribi"
  spec.version = Kiribi::VERSION
  spec.authors = ["matsudai"]

  spec.summary = "Easy to use some onnx models. Use kiribi-multilingual_e5-small instead."
  spec.homepage = "https://github.com/matsudai/kiribi"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 3.4.0"

  # Specify which files should be added to the gem when it is released.
  # The `git ls-files -z` loads the files in the RubyGem that have been added into git.
  gemspec = File.basename(__FILE__)
  spec.files = IO.popen(%w[git ls-files -z], chdir: __dir__, err: IO::NULL) do |ls|
    ls.readlines("\x0", chomp: true).reject do |f|
      (f == gemspec) ||
        f.start_with?(*%w[bin/ Gemfile .gitignore])
    end
  end
  spec.require_paths = ["lib"]
end
