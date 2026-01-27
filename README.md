# Kiribi

## Examples

### Ruri-v3-30m

```sh
gem install kiribi-ruri_v3-30m
```

```rb
require "kiribi/ruri_v3/ruri30m"

model = Kiribi.load
# # equivalently, you can load via a specific module path:
# model = Kiribi::RuriV3.load
# model = Kiribi::RuriV3::Ruri30M.load

a = model.embedding_normalized("ドキュメントの内容")
b = model.embedding_normalized("検索クエリ")

def cosine_similarity(a, b) = a.zip(b).sum { |x, y| x * y }

puts cosine_similarity(a, b) # => -1.0 - 1.0
```

### Multilingual-E5-Small

```sh
gem install kiribi-multilingual_e5-small
```

```rb
require "kiribi/multilingual_e5/small"

model = Kiribi.load
# # equivalently, you can load via a specific module path:
# model = Kiribi::MultilingualE5.load
# model = Kiribi::MultilingualE5::Small.load

a = model.embedding_passage("Document contents")
b = model.embedding_query("Search query")

def cosine_similarity(a,b); a.zip(b).sum{|x,y| x*y} / Math.sqrt(a.sum{|x| x**2} * b.sum{|y| y**2}); end

puts cosine_similarity(a, b) # => 0.0 - 1.0
```
