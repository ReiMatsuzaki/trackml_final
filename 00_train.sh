# 1K samples

name=$0
i=$1
name="${name%.*}"
outdir=out_${name}

if [ ! -e ${outdir} ]; then
    mkdir -p ${outdir}
fi

for i in `seq -f %03g 0 999`
do
    log=${outdir}/${i}.log
    out_path=${outdir}/${i}
    if [ ! -e ${log} ]; then
	echo "calc run:" $i
	echo "" > ${log}
	python calc.py -mode bin \
	       -n_seed 1024 \
	       -std_z0 5.5 \
	       -n_bins_theta0xy 700 \
	       -n_bins_theta0z  900 \
	       -out_path ${out_path} \
	       -idx 0 -num 1 >> ${log}
	exit
    fi
done




