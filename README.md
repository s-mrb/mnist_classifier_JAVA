**JUST DOWNLOAD NOTEBOOK AND RUN ALL CELLS**

This Notebook contains setup to run Multiclass **MNIST classifier** Neural Network (**1 Hidden Layer**) implemented in **JAVA** without the use of any ML library like deeplearning4j etc. Very Simple architecture, just one hidden layer of 64 neurons.
Only dependency is JAMA library, it is used for **only Matrix** Multiplication.


**Accuracy:**
```
lr = 1

78.67% --> 100 epochs
80%    --> 200 epochs
>90%   --> 1000 epochs  (One run gave 91% whereas other 89.7, this variation is due to random weight initialization)
>97%   --> 3000 epohs
```


**NOTE :**
Get the link to dataset (*csv*) , Scripts and Jar file on drive and just run all the cells.

```
 Run via shell, Script takes three command line arguments

1.  Path to csv
2.  Total epochs
3.  Learning Rate
4.  0/1 --> 0 if you don't want to print stats while training, 1 otherwise.

```

***Model Trains on CPU, make sure your **JVM heap memory** is set to minimum of 6GB***