for var in `ls -d SyntheticDemo_result_*`
do
    echo $var
    cp LABEL/* $var/
    cp DATA/* $var/
    cp summary.py $var/
    cd $var
    python summary.py
    cd ..
done
