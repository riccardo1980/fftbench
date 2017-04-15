#!/bin/sh
PR=CUFFT
TYPES=(R2C D2Z C2C Z2Z)
TESTCASES=(A B C D E)
LOOPS=100;
nthreads=(1 2 4 8)
for TT in ${TYPES[@]}; do
	for CC in ${TESTCASES[@]}; do
		case "$CC" in
			'A') 
			testSizes=( 256x256 )
			;;
			'B') 
			testSizes=( $((2*256))x$((2*256))  )
			;;
 			'C') 
			testSizes=( $((4*256))x$((4*256))  )
      ;;
 			'D') 
			testSizes=( $((8*256))x$((8*256))  )
      ;;
 			'E') 
			testSizes=( $((16*256))x$((16*256))  )
      ;;
 esac
		for ii in ${testSizes[@]}; do
			echo "$PR (averaged on $LOOPS loops) $TT $ii" 
			bin/fftime -l $LOOPS -p $PR -t $TT -s $ii 
		done
	done
done

exit

