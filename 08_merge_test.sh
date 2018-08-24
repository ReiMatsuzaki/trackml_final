# 10x20K samples

name=$0
name="${name%.*}"
outdir=out_${name}

if [ ! -e ${outdir} ]; then
    mkdir -p ${outdir}
fi

log=${outdir}/${n}.log
out_path=${outdir}/$n

echo "calc for " $n
echo "" > ${log}

python calc.py -mode read -n_seed 20 \
       -submission_root out_07_merge_test \
       -test true \
       -out_path ${out_path} >> ${log}
