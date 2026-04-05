# frozen_string_literal: true

require_relative "e2b/version"
require_relative "e2b/vision_encoder"
require_relative "e2b/audio_encoder"
require_relative "e2b/model"
require "kiribi"

module Kiribi
  module Gemma4
    extend Kiribi::Loader

    module E2B
      extend Kiribi::Loader

      def self.instantiate
        Model.new
      end
    end
  end
end

Kiribi.register(Kiribi::Gemma4::E2B, order: 100_300_100)
