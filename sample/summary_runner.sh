#!/bin/bash

label=sample_results

while getopts l: OPT
do
  case $OPT in
    "l" ) label="${OPTARG}" ;;
  esac
done

result_dirs=`ls ${label} | grep -o "[0-9]*"`

mkdir -p summary_files
mkdir -p results

for dir in ${result_dirs}
do
  echo ${dir}

  rm -f summary_files/*
  rm -f results/*
  rm -f log.txt

  cp ${label}/${dir}/log.txt ./
  cp ${label}/${dir}/results/* results/
  cp ${label}/${dir}/summary_files/* summary_files/

  python summary_and_plot.py
  # python summary_and_plot_light.py
  # python summary_and_plot_without_label.py

  cp -r summary_files/ ${label}/${dir}/
  cp -r figures/ ${label}/${dir}/

done

rm -f summary_files/*
rm -f figures/*
rm -f results/*
rm -f log.txt

python summary_summary.py ${label} ${result_dirs}
# python summary_summary_without_ARI.py ${label} ${result_dirs}

cp -r summary_files ${label}
cp -r figures ${label}

rm -rf summary_files
rm -rf figures
rm -rf results
