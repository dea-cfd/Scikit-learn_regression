#!/bin/bash
# @author : D. Chibouti
# Paris

rep_here=`pwd`

echo "Welcome to this script ChD"
list_sigma=$(echo 1e-1 1e1)
list_sigma=$(echo 1)
for sigma in $list_sigma
echo "this is a script to run many simulations at the same time"
do

    list_ou=$(seq 1)
    for ou in $list_ou
    do


	list_op=$(seq 1 5)
	#list_op=$(echo 1 2 3 4 5 7 10 15 20) #ChD values
	for op in $list_op
	do

	    list_base=$(echo -1 1)
	    for base in $list_base
	    do

		#rep_save=sigma${sigma}_1000iter_p${op}_u${ou}
		if [ $base -eq 1 ]
		then
		    rep_save=test_bruit_p${op}_u${ou}
		else
		    rep_save=test_bruit_ip${op}_u${ou}
		fi

		echo $rep_save

		mkdir -p $rep_save
		cp base0.txt $rep_save

		sed    "s/ORDRE_P/${op}/g" reg_2d_no_grad.py > tmp.py 
		sed -i "s/ORDRE_U/${ou}/g" tmp.py
		sed -i "s/VAL_SIG/${sigma}/g" tmp.py
		sed -i "s/VAL_BASE/${base}/g" tmp.py

		mv tmp.py $rep_save
		cd $rep_save

		python3 tmp.py > screen.out

		cd $rep_here

	    done
	    unset list_base
	done
	unset list_op

    done
    unset list_ou

done
unset list_sigma
