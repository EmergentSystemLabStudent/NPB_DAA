#!/bin/bash

label=sample_results
begin=1
end=20

while getopts l:b:e: OPT
do
  case $OPT in
    "l" ) label="${OPTARG}" ;;
    "b" ) begin="${OPTARG}" ;;
    "e" ) end="${OPTARG}" ;;
  esac
done

bash runner.sh -l ${label} -b ${begin} -e ${end}

bash summary_runner.sh -l ${label}

bash clean.sh
