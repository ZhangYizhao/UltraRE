# UltraRE

This is PyTorch implementation for the paper published on NeurIPS 2023:

"UltraRE: Enhancing RecEraser for Recommendation Unlearning via Error Decomposition"

Paper Page : (https://openreview.net/forum?id=93NLxUojvc)

collaborative filtering model: 

- Matrix Factorization (MF)

division method:

- OT based clustering

unlearning workflow:

- Isolation (sisa)

## Environment Settings

- PyTorch versipn : 1.13.0

- POT version : 0.9.0

## Example to run the codes.

The instruction of commands has been clearly stated in the codes (see the  parse_args function). 

Train model:

```python
python main.py --dataset ml1m --epoch 50 --group 0 
```

Unlearn:

```python
python main.py --dataset ml1m --epoch 50 --group 5 --learn sisa --delper 2 --deltype rand
```

Note you can edit this two functions to change the backbone model

```
config.py run_Group() run_Full()  
```

Also you can add dataset by editing config.py

## Dataset

We provide one processed datasets : MovieLens 1 Million (ml-1m)  in data/ml1m/

ratings.dat : original interaction data

pro.ipynb : process data and split the train/test data

## Test
Run test.ipynb to compare the performance of several clustering methods
