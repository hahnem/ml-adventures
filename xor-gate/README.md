# xor gate test
Sometimes the training get stuck on 0.75 accuracy. Why? Gradient descent starting point?
If it happends, restart training.
TODO, figure out why..

## train.py
```
$ python train.py 
Epoch 1/3
400000/400000 [==============================] - 3s 7us/sample - loss: 0.1092 - accuracy: 0.9993
Epoch 2/3
400000/400000 [==============================] - 3s 7us/sample - loss: 0.0034 - accuracy: 1.0000
Epoch 3/3
400000/400000 [==============================] - 3s 7us/sample - loss: 6.3340e-04 - accuracy: 1.0000
400000/400000 [==============================] - 2s 4us/sample - loss: 2.6502e-04 - accuracy: 1.0000
score: [0.00026502468972466886, 1.0]
Saving model to xor_model.5
```

## predict.py
$ python predict.py 
```
XOR Predictions table:
-------------
| A | B | Q |
-------------
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |
-------------
```

