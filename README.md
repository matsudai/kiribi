# Kiribi

```sh
gem install kiribi-multilingual_e5-small
```

```rb
require "kiribi/multilingual_e5/small"

model = Kiribi.load
a = model.embedding_passage("Document contents")
b = model.embedding_query("Search query")

def cosine_similarity(a,b); a.zip(b).sum{|x,y| x*y} / Math.sqrt(a.sum{|x| x**2} * b.sum{|y| y**2}); end

cosine_similarity(a, b) # => 0.0 - 1.0
```
