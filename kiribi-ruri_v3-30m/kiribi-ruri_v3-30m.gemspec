# frozen_string_literal: true

require_relative "lib/kiribi/ruri_v3/ruri30m/version"

Gem::Specification.new do |spec|
  spec.name = "kiribi-ruri_v3-30m"
  spec.version = Kiribi::RuriV3::Ruri30M::VERSION
  spec.authors = ["matsudai"]

  spec.summary = "[DEPRECATED] Use the kiribi gem directly (Kiribi.download/load). This addon will be removed."
  spec.homepage = "https://github.com/matsudai/kiribi"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 3.4.0"
  spec.extensions = ["ext/kiribi-ruri_v3-30m/extconf.rb"]

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

  # Uncomment to register a new dependency of your gem
  # spec.add_dependency "example-gem", "~> 1.0"
  spec.add_dependency "kiribi", ">= 0.0.1", "< 0.1.0"
  spec.add_dependency "onnxruntime", ">= 0.10.0"
  spec.add_dependency "tokenizers", ">= 0.6.0"
end
