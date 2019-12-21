
for var in pr sph srad tmmn tmmx pet etr 
do
	for i in {1979..2019}
	do
		wget -nc -c -nd http://www.northwestknowledge.net/metdata/data/${var}_${i}.nc
	done
done
