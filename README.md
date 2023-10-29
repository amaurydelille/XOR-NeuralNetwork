## Introduction

This file explain and show how I developed a neural network that learns the XOR function. I implemented this neural network in C with standard library only, no installation are needed to make my program work.

### Installation

Once you cloned the repository, if your computer recognize the ```make``` command line, then type this :

```
$ make            #compile all the files.
$ ./main          #execute the .exe file.
$ make clean      #delete the *.o files.
```
 
 If not, then type this :
 
```
$ gcc -Wextra -Wall main.c xor.c  #compile all the files.
$ ./main                          #execute all the files.
```

## Explanation

My neural network learns the XOR function, as a reminder, here is the XOR function :

| $x_1$ | $x_2$ | $y$ |
|---------|---------|--------|
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 0      |


The neural network is made of one input layer of 2 neurons, one hidden layer of 2 neurons too and a output layer of one neuron. 




