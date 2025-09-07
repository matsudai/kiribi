# frozen_string_literal: true

require_relative "kiribi/version"

module Kiribi
  module Loader
    def self.registry
      @registry ||= []
    end

    def current_path
      caller = (instance_of?(Module) || instance_of?(Class)) ? self : self.class
      caller.name.split("::")
    end

    def registry
      Kiribi::Loader.registry
    end

    def register(klass, opt = {})
      order = opt[:order] || 0
      path = opt[:path] || klass.name.split("::")

      raise ArgumentError, "Order #{order} is already taken" if registry.any? { it[:order] == order }
      raise ArgumentError, "Path #{path.join("::")} is already taken" if registry.any? { it[:path] == path }

      registry << { klass: klass, order: order, path: path }
    end

    def find_model
      candidates = registry.filter do |entry|
        entry[:path][0...current_path.size] == current_path
      end

      candidates.min_by { it[:order] }&.[](:klass)
    end

    def load
      find_model.instantiate
    end

    def instantiate(*, **, &)
      new(*, **, &)
    end
  end

  extend Kiribi::Loader
end
