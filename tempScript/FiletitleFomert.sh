#!/bin/zsh
for file in `\find ./LABEL/ ! -name '.DS_Store' `; do
  filename=`basename ${file} .lab`
  echo ${filename}
  echo "\""${filename}"\"," >> filename.txt

  sleep 1

done


exit 0