# Results during testing and developing

## `2021-12-06`

### `lambda=0.1`

Node that this `J` is calculated on the training set, not the test set

```
[2021-12-06 20:05:35,370 | INFO] Load set: {'cv1', 'cv2', 'cv0'}
[2021-12-06 20:05:45,129 | INFO] Shape of X: (1516, 2550)
[2021-12-06 20:05:45,129 | INFO] Shape of Y: (1516,)
[2021-12-06 20:05:45,225 | INFO] Neural network with shape (2550, 1275, 4) was created
[2021-12-06 20:05:46,910 | INFO] Initial J: 3.765471329930902
[2021-12-06 20:05:46,921 | INFO] Optimize with max 100 iterations ...
[2021-12-06 20:09:08,874 | INFO] New J: 3.495858644279812
[2021-12-06 20:09:08,885 | INFO] Optimize with max 100 iterations ...
[2021-12-06 20:12:37,598 | INFO] New J: 3.4757706421609855
[2021-12-06 20:12:37,610 | INFO] Optimize with max 100 iterations ...
J's: [3.765471329930902, 3.495858644279812, 3.4757706421609855]
{'layer_sizes': {'layer_0': 2550, 'layer_1': 1275, 'layer_2': 4}, 'lambda': 0.1, 'training_iterations': 300}
```

```
[2021-12-06 20:15:29,129 | INFO] Load set: {'cv2', 'cv3', 'cv1'}
[2021-12-06 20:15:41,563 | INFO] Shape of X: (1515, 2550)
[2021-12-06 20:15:41,563 | INFO] Shape of Y: (1515,)
[2021-12-06 20:15:41,651 | INFO] Neural network with shape (2550, 1275, 4) was created
[2021-12-06 20:15:43,415 | INFO] Initial J: 3.1905649947583514
[2021-12-06 20:15:43,428 | INFO] Optimize with max 100 iterations ...
[2021-12-06 20:18:46,609 | INFO] New J: 2.9136367407914863
[2021-12-06 20:18:46,619 | INFO] Optimize with max 100 iterations ...
J's: [3.1905649947583514, 2.9136367407914863]
{'layer_sizes': {'layer_0': 2550, 'layer_1': 1275, 'layer_2': 4}, 'lambda': 0.1, 'training_iterations': 200}
```

```
[2021-12-06 20:21:39,596 | INFO] Load set: {'cv3', 'cv2', 'cv0'}
[2021-12-06 20:21:49,046 | INFO] Shape of X: (1516, 2550)
[2021-12-06 20:21:49,046 | INFO] Shape of Y: (1516,)
[2021-12-06 20:21:49,139 | INFO] Neural network with shape (2550, 1275, 4) was created
[2021-12-06 20:21:50,889 | INFO] Initial J: 4.224096311838043
[2021-12-06 20:21:50,900 | INFO] Optimize with max 100 iterations ...
[2021-12-06 20:24:46,698 | INFO] New J: 4.095932400243479
[2021-12-06 20:24:46,708 | INFO] Optimize with max 100 iterations ...
J's: [4.224096311838043, 4.095932400243479]
{'layer_sizes': {'layer_0': 2550, 'layer_1': 1275, 'layer_2': 4}, 'lambda': 0.1, 'training_iterations': 200}
```

```
[2021-12-06 20:27:17,997 | INFO] Optimize with max 100 iterations ...
[2021-12-06 20:30:35,415 | INFO] New J: 2.910980747570245
[2021-12-06 20:30:35,425 | INFO] Optimize with max 100 iterations ...
[2021-12-06 20:33:56,073 | INFO] New J: 2.837081846781543
[2021-12-06 20:33:56,084 | INFO] Optimize with max 100 iterations ...
[2021-12-06 20:37:20,446 | INFO] New J: 2.824758761072383
[2021-12-06 20:37:20,457 | INFO] Optimize with max 100 iterations ...
[2021-12-06 20:40:30,571 | INFO] New J: 2.751067800968105
[2021-12-06 20:40:30,583 | INFO] Optimize with max 100 iterations ...
[2021-12-06 20:43:34,070 | INFO] New J: 2.750599952268968
[2021-12-06 20:43:34,082 | INFO] Optimize with max 100 iterations ...
J's: [4.442316847161754, 2.910980747570245, 2.837081846781543, 2.824758761072383, 2.751067800968105, 2.750599952268968]
{'layer_sizes': {'layer_0': 2550, 'layer_1': 1275, 'layer_2': 4}, 'lambda': 0.1, 'training_iterations': 600}
```
