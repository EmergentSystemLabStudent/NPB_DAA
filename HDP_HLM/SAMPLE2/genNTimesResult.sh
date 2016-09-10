echo "number,adjusted_rand_score_l,adjusted_rand_score_s,per,wer,loglikelihood" > NTIMESRESULT.csv
for var in `ls -d startDAA_result_*`
do
    var1=`cat "$var/summary_figs/maxLk_adjusted_rand_index_l.txt" | tr -d "adjusted_rand_score," | tr -d "\r"`
    var2=`cat "$var/summary_figs/maxLk_adjusted_rand_index_s.txt" | tr -d "adjusted_rand_score," | tr -d "\r"`
    var3=`cat $var/loglikelihood.txt | perl -nle '(!defined($max) || $max<$_) and $max=$_; END{print $max}'`
    var4=`head -1 "$var/summary_figs/PERandWER.txt" | tr -d "PER," | tr -d "\r"`
    var5=`tail -1 "$var/summary_figs/PERandWER.txt" | tr -d "WER," | tr -d "\r"`
    echo $var,$var1,$var2,$var4,$var5,$var3 >> NTIMESRESULT.csv
done
