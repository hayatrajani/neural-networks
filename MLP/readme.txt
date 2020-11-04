========================================================================
Written in python3, tested on Ubuntu 18.04
========================================================================

usage: main.py [-h] -s N M [-t file_path] [-w file_path] [-p pattern]
               [-e file_path]

Script to test the MLP implementation

optional arguments:
  -h, --help            show this help message and exit
  -s N M, --size N M    the number of input (N) and output (M) neurons
  -t file_path, --train file_path
                        path of training file
  -w file_path, --weights file_path
                        path of file containing network weights
  -p pattern, --predict pattern
                        input pattern
  -e file_path, --evaluate file_path
                        path of test file for evaluation

========================================================================
Examples
========================================================================
1. Predict an output without training
	python3 main.py -s 5 2 -p 1,0,0,1,0

2. Train the MLP
	python3 main.py -s 5 2 -t train.dat

3. Train the MLP and Evaluate
	python3 main.py -s 5 2 -t train.dat -e test.dat

4. Train and predict
	python3 main.py -s 5 2 -t train.dat -p 1,0,0,1,0

5. Load saved weights and predict an output:
	python3 main.py -s 5 2 -w weights.dat -p 1,0,0,1,0


To Plot the learning curve,
	gnuplot -p -e "plot 'learning.curve' with lines"

========================================================================
