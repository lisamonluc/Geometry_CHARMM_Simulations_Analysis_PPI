#!/bin/bash

dvdl='dvdl.dat'
ndvdl="restart/$dvdl"

nxt_st=$(awk "NR==2{print \$1;exit}" "$ndvdl")

awk "\$1 >= $nxt_st{print \$1,\$2,\$3,\$4,\$5,\$6}" "$dvdl" > cur_over.dat

cur_end=$(tail -1 cur_over.dat | cut -d' ' -f 1)

awk "\$1 <= $cur_end{print \$1,\$2,\$3,\$4,\$5,\$6}" "$ndvdl" > nxt_over.dat
