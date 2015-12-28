#!/bin/bash

for size in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536; do 
	sed -ire "s/VECSIZE [0-9]\+/VECSIZE $size/g" src/benchmark.c
	mpicc -std=c99 -lm src/benchmark.c -o bin/mpi.o
	for numnodes in 1 2 4 8 16 32; do
		# echo -n "n=$numnodes size=$size "
		mpirun -np $numnodes -machinefile mpi.machines bin/mpi.o
		echo -n ", "
	done
	echo ""
done
