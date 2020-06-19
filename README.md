(A reference project)

Just some code toying with nearest neighbors algorithms in Python. Numpy is the major dependency (note numpy is much faster if you [install openblas](https://stackoverflow.com/questions/11443302/compiling-numpy-with-openblas-integration)). 

Demos use 400k vocab, 300 dimensional Glove vectors 

## Exact Nearest Neighbors

Will find nearest neighbors of provided terms

```
python3 exact_nn.py cat bunny kitty
```

## Simple Random Projections

Test 300 projection vectors to hash each vector, then compare recall to exact\_nn:

```
python3 rand_proj.py 300
```
