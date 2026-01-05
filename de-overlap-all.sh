#!/bin/bash

for host in hosts/* ; do
	for system in ${host}/* ; do
		for geo in ${system}/* ; do
			echo  "processing $geo"
			./merge_restarts.py "$geo"
		done
	done
done
