#!/bin/sh
PR=CUFFT
TYPES=(R2C D2Z C2C Z2Z)
TESTCASES=(A B C D)
LOOPS=100;
nthreads=(1 2 4 8)
for TT in ${TYPES[@]}; do
	for CC in ${TESTCASES[@]}; do
		case "$CC" in
			'A') 
			testSizes=( 1048576 1024x1024 256x256x16 )
			;;
			'B') 
			testSizes=( $(( 4*1048576)) $((2*1024))x$((2*1024)) $((1*256))x$((1*256))x$((4*16)) )
			;;
			'C')
			testSizes=( $((16*1048576)) $((4*1024))x$((4*1024)) $((2*256))x$((2*256))x$((4*16)) )
			;;
			'D')
			testSizes=( $((64*1048576)) $((8*1024))x$((8*1024)) $((4*256))x$((4*256))x$((4*16)) )
			;;
		esac
		for ii in ${testSizes[@]}; do
			echo $PR $TT $ii 
			bin/fftime -p $PR -t $TT -s $ii 
		done
	done
done

exit

