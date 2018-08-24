# 10x20K samples

name=$0
name="${name%.*}"
outdir=out_${name}

if [ ! -e ${outdir} ]; then
    mkdir -p ${outdir}
fi

for n in `seq -f %03g 0 999`
do    
    log=${outdir}/${n}.log
    out_path=${outdir}/$n
    if [ ! -e ${log} ]; then
	echo "calc for " $n
	echo "" > ${log}

	python calc.py -mode read -n_seed 32 \
	       -submission_root out_06_merge_test_all \
	       -test true -i0 ${n} \
	       -out_path ${out_path} >> ${log}
	exit
    fi
done
