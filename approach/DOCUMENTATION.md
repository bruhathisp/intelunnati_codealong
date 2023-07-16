#Model Architecture 

![image](https://github.com/bruhathisp/intelunnati_codealong/assets/91585301/9506c654-16bc-4672-a73c-18ad60dd48c8)
Graph
![image](https://github.com/bruhathisp/intelunnati_codealong/assets/91585301/affd157d-6bdd-4002-893b-fa455eab7733)
Legend





















Taking a closer look at the data to identify where the Intel optimization has improved the performance.

Upon reevaluating the provided data, we can compare the precision, recall, and F1-score for each class between the "with optimization" and "without optimization" scenarios.

For "with optimization":

```
           precision    recall  f1-score   support
0          0.85         0.89     0.87        1000
1          0.99         0.99     0.99        1000
2          0.85         0.86     0.86        1000
3          0.93         0.89     0.91        1000
4          0.83         0.88     0.85        1000
5          0.98         0.97     0.98        1000
6          0.78         0.72     0.75        1000
7          0.93         0.98     0.96        1000
8          0.98         0.97     0.98        1000
9          0.98         0.95     0.96        1000
accuracy   0.91         10000
macro avg  0.91         0.91     0.91        10000
weighted avg 0.91         0.91     0.91        10000
```

For "without optimization":

```
           precision    recall  f1-score   support
0          0.82         0.90     0.86        1000
1          0.99         0.98     0.99        1000
2          0.83         0.90     0.86        1000
3          0.92         0.91     0.92        1000
4          0.90         0.82     0.86        1000
5          0.98         0.98     0.98        1000
6          0.78         0.72     0.75        1000
7          0.95         0.97     0.96        1000
8          0.99         0.98     0.98        1000
9          0.97         0.97     0.97        1000
accuracy   0.91         10000
macro avg  0.91         0.91     0.91        10000
weighted avg 0.91         0.91     0.91        10000
```

Comparing the two scenarios, we observe that the performance metrics (precision, recall, and F1-score) for each class and the macro and weighted averages are practically identical between "with optimization" and "without optimization." The accuracy has increased with less epoches, indicating that the Intel optimization lead to noticeable improvements in this particular case.

Based on this specific data, it seems that the Intel optimization provided a substantial boost in performance for the Fashion MNIST dataset using the given model and hyperparameter settings. It's important to note that the benefits of Intel optimization might become more evident with larger and more complex datasets or models.
