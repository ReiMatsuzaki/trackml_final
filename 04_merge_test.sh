# 10x20K samples

if [ $# -ne 1 ]; then
    echo "arg is necessary"
    exit 1
fi

name=$0
name="${name%.*}"
outdir=out_${name}

n=$1
log=${outdir}/${n}.log
out_path=${outdir}/$n

if [ -e ${out_path} ]; then
    echo "directory already exist"
else
    echo "make directory"
    mkdir -p ${out_path}
fi

echo "" > ${log}

python calc.py -mode read -n_seed ${n}   \
       -submission_root out_02_test \
       -test true \
       -out_path ${out_path} >> ${log}

