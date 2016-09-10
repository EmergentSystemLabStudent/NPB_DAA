#!/bin/zsh
for var in `seq 1 10`
do
	echo $var
	python startDAA.py >> tmp
done
cat tmp | grep total > TIME.txt
rm tmp
