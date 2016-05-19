#!/usr/bin/python
import argparse
import sys
import matrix_pb2
import numpy as np
from matplotlib.pyplot import *
from scipy import io

def mat_to_array(m):
    return np.array(m.idata).reshape(m.rows,m.cols)

def main(filePath):
    print(filePath)
    f = open(filePath,'rb')
    p = matrix_pb2.MatrixClientMessage.FromString(f.read())
    f.close()

    arr = None
    arr_energy = None
    for m in p.matrix_list:
        if m.id == 'feature_chunk':
            if arr == None:
                arr = mat_to_array(m)
            else:
                arr = np.append(arr,mat_to_array(m),axis=0)
        elif m.id == 'energy_chunk':
            if arr_energy == None:
                arr_energy = mat_to_array(m)
            else:
                arr_energy = np.append(arr_energy,mat_to_array(m),axis=1)

    arr_energy = arr_energy.transpose()

    #"mfccs"
    arr2 = arr * np.tile(arr_energy,(1,16)) / 1024.0 / 8.0

    output = {}
    output['arr'] = arr
    output['arr_energy'] = arr_energy
    output['arr2'] = arr2

    io.savemat(filePath+'.mat', output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('filePath', type=str)
    args = parser.parse_args()

    main(args.filePath)

