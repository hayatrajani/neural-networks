#!/usr/bin/env python3

"""
@file   main.py
@author Hayat Rajani  [hayatrajani@gmail.com]

November 04, 2019
"""

from network import Network
import numpy as np
import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        description='Script to test the MLP implementation')
    parser.add_argument('-s', '--size',
                        required=True,
                        type=int,
                        nargs=2,
                        metavar=('N', 'M'),
                        help='the number of input (N) and output (M) neurons')
    parser.add_argument('-t', '--train',
                        type=str,
                        metavar='file_path',
                        help='path of training file')
    parser.add_argument('-w', '--weights',
                        type=str,
                        metavar='file_path',
                        help='path of file containing network weights')
    parser.add_argument('-p', '--predict',
                        type=float,
                        nargs='+',
                        metavar='pattern',
                        help='input pattern')
    parser.add_argument('-e', '--evaluate',
                        nargs=2,
                        type=str,
                        metavar=('test_file_path', 'threshold'),
                        help='path of test file for evaluation')
    args = parser.parse_args()

    if args.size[0] > 1000 or args.size[1] > 1000:
        print('Invalid size! At most 1000 neurons per layer')
        sys.exit(0)
    else:
        size = [args.size[0], 8, 4, args.size[1]]
        activations = ['relu', 'relu', 'tanh']
        net = Network(size, activations, 0.005)

    if args.weights:
        if not os.path.isfile(args.weights):
            print('Invalid path! File does not exist')
            sys.exit(0)
        net.load_weights(args.weights)

    if args.train:
        if not os.path.isfile(args.train):
            print('Invalid path! File does not exist')
            sys.exit(0)
        fin = open(args.train)
        c = 0
        _ = fin.readline()                                      # skip first header
        for line in fin.readlines():
            if line[0] == '#':
                params = line[1:].split()
                P = int(params[0][2:])
                if P > 1000:
                    print('Error! At most 1000 training patterns')
                N = int(params[1][2:])
                M = int(params[2][2:])
                if N != size[0] or M != size[-1]:
                    size[0] = N
                    size[-1] = M
                    net = Network(size, activations)
                inputs = np.zeros((N, P))
                outputs = np.zeros((M, P))
            else:
                l = line.split()
                inputs[:, c] = np.array(l[:N], dtype=np.float)
                outputs[:, c] = np.array(l[N:], dtype=np.float)
                c += 1
        errors = net.train(P, inputs, outputs)
        fout = open('learning.curve', 'w')
        fout.write('# Epochs  Errors\n')
        for i, e in zip(range(1, len(errors)+1), errors):
            fout.write(str(i)+'\t'+str(e)+'\n')
        fout.close()
        print("Errors saved in file 'learning.curve'")

    if args.evaluate:
        if not os.path.isfile(args.evaluate[0]):
            print('Invalid path! File does not exist')
            sys.exit(0)
        fin = open(args.evaluate[0])
        c = 0
        _ = fin.readline()                  # skip first header
        for line in fin.readlines():
            if line[0] == '#':
                params = line[1:].split()
                P = int(params[0][2:])
                if P > 1000:
                    print('Error! At most 1000 training patterns')
                N = int(params[1][2:])
                M = int(params[2][2:])
                if N != size[0] or M != size[-1]:
                    print('Error! Size mismatch.')
                    sys.exit(0)
                inputs = np.zeros((N, P))
                outputs = np.zeros((M, P))
            else:
                l = line.split()
                inputs[:, c] = np.array(l[:N], dtype=np.float)
                outputs[:, c] = np.array(l[N:], dtype=np.float)
                c += 1
        net.evaluate(P, inputs, outputs, float(args.evaluate[1]))

    if args.predict:
        if len(args.predict) != args.size[0]:
            print(
                'Invalid pattern! Pattern length must be equal to the no. of input neurons')
            sys.exit(0)
        output = net.predict(np.asarray(args.predict, dtype=np.float))
        print('Predicted output:', end=' ')
        print(np.array2string(output))


if __name__ == '__main__':
    main()
