for var in `ls -d startDAA_result_*`
do
    echo $var
    cd $var
    python ../summary.py
    cd ..
done
