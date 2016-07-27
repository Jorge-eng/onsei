#!/usr/bin/python

import numpy as np
import sys


def write(my_str):
    sys.stdout.write(my_str)


def gen_exp_table():
    r = range(-512,513)

    write('const static int16_t exp_table[%d] = {' % (512*2 + 1))

    for i,x in enumerate(r):
        y = int(np.round(np.exp(x/128.0)*128))

        if (i > 0):
            write(',')

        write(str(y))

    write('};\n')


                    
def gen_tanh_table():
    #assumes input is s16q7
    #and output 8s s8q7

    #bounds are between -1 and 1 on the output, it's symmetric
    #so assume only inputs on domain [0,inf)
    Q = 7

    largest_number = (2**Q - 1) / float(2 ** Q)

    xmax = int(np.arctanh(largest_number) * (2**Q)) + 2


    table = []
    for i in range(xmax):
        y = int((np.tanh( (i + 0.5) / float(2**Q)) * (2**Q)))
        table.append(str(y))

    write('const static int8_t tanh_table[%d] = {' % (xmax))
    for i,item in enumerate(table):
        if (i > 0):
            write(',')
            
        write(item)
        
    write('};\n')
                  
if __name__ == '__main__':
    gen_tanh_table()
    gen_exp_table()
