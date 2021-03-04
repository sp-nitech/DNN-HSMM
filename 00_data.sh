#!/bin/bash

declare -A dnames
dnames[lab.trn]=path/lab.trn
dnames[lf0.trn]=path/lf0.trn
dnames[mgc.trn]=path/mgc.trn
dnames[bap.trn]=path/bap.trn
dnames[lab.test]=path/lab.test
dnames[lf0.test]=path/lf0.test
dnames[mgc.test]=path/mgc.test
dnames[bap.test]=path/bap.test

declare -A dims
dims[lab]=706
dims[lf0]=1
dims[mgc]=50
dims[bap]=25

declare -A dttype
dttype[lab]=f
dttype[lf0]=f
dttype[mgc]=f
dttype[bap]=f

rm -rf normfac tmp
mkdir -p data/trn data/test normfac tmp/meansdev

# serialization
for feat in lab lf0 mgc bap
do
    find -L ${dnames[${feat}.trn]} -name "*.$feat" -exec basename {} .$feat \; | sort -u > tmp/$feat.trn.scp
    find -L ${dnames[${feat}.test]} -name "*.$feat" -exec basename {} .$feat \; | sort -u > tmp/$feat.test.scp
done
for i in trn test
do
    cp tmp/lab.$i.scp tmp/scp
    for feat in lf0 mgc bap
    do
	grep -x -f tmp/scp tmp/$feat.$i.scp > tmp/tmp.scp
	mv tmp/tmp.scp tmp/scp
    done
    for bname in `cat tmp/scp`
    do
	cmd="python data/scripts/serialization.py"
	for feat in lab lf0 mgc bap
	do
	    cmd=$cmd" $feat ${dims[$feat]} ${dttype[$feat]} ${dnames[${feat}.${i}]}/$bname.$feat"
	done
	cmd=$cmd" data/$i/$bname.npz"
	$cmd
    done
done

# meansdev
echo -0.5 0.0 0.5 | x2x +af > tmp/delta
echo 1.0 -2.0 1.0 | x2x +af > tmp/accel
find ${dnames[lab.trn]} -name "*.lab" -exec cat {} + | vstat -l ${dims[lab]} -o 1 > tmp/meansdev/lab.mean
find ${dnames[lab.trn]} -name "*.lab" -exec cat {} + | vstat -l ${dims[lab]} -o 2 -d | sopr -f 0.0 -R  > tmp/meansdev/lab.sdev
echo 0 | x2x +af > tmp/meansdev/vuv.mean
echo 1 | x2x +af > tmp/meansdev/vuv.sdev
find ${dnames[lf0.trn]} -name "*.lf0" -exec bash -c "python3 data/scripts/interpolate.py {} | delta -l ${dims[lf0]} -d tmp/delta -d tmp/accel" \; | vstat -l `expr 3 \* ${dims[lf0]}` -o 1 > tmp/meansdev/lf0.mean
find ${dnames[lf0.trn]} -name "*.lf0" -exec bash -c "python3 data/scripts/interpolate.py {} | delta -l ${dims[lf0]} -d tmp/delta -d tmp/accel" \; | vstat -l `expr 3 \* ${dims[lf0]}` -o 2 -d | sopr -f 0.0 -R > tmp/meansdev/lf0.sdev
for feat in mgc bap
do
    find ${dnames[${feat}.trn]} -name "*.$feat" -exec delta -l ${dims[$feat]} -d tmp/delta -d tmp/accel {} \; | vstat -l `expr 3 \* ${dims[$feat]}` -o 1 > tmp/meansdev/$feat.mean
    find ${dnames[${feat}.trn]} -name "*.$feat" -exec delta -l ${dims[$feat]} -d tmp/delta -d tmp/accel {} \; | vstat -l `expr 3 \* ${dims[$feat]}` -o 2 -d | sopr -f 0.0 -R > tmp/meansdev/$feat.sdev
done
cmd="python data/scripts/serialization.py"
for i in lab vuv lf0 mgc bap
do
    cmd=$cmd" $i.mean 0 f tmp/meansdev/$i.mean"
    cmd=$cmd" $i.sdev 0 f tmp/meansdev/$i.sdev"
done
cmd=$cmd" normfac/meansdev.npz"
$cmd

# clean
rm -rf tmp
