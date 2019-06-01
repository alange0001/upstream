#!/usr/bin/gnuplot
#
# Modified from: Hagen Wierstorf (www.gnuplotting.org)

reset

# wxt
set terminal wxt size 410,250 enhanced font 'Verdana,9' persist
# png
#set terminal pngcairo size 410,250 enhanced font 'Verdana,9'
#set output 'nice_web_plot.png'
# svg
#set terminal svg size 410,250 fname 'Verdana, Helvetica, Arial, sans-serif' \
#fsize '9' rounded dashed
#set output 'nice_web_plot.svg'

# define axis
# remove border on top and right and set color to gray
set style line 11 lc rgb '#808080' lt 1
set border 3 back ls 11
set tics nomirror
# define grid
set style line 12 lc rgb '#808080' lt 0 lw 1
set grid back ls 12

# color definitions
set style line 1 linecolor rgb '#3366FF' ps 0 linetype 1 linewidth 1.2
set style line 2 linecolor rgb '#3399FF' ps 0 linetype 1 linewidth 1.2
set style line 3 linecolor rgb '#CC0033' ps 0 linetype 1 linewidth 1.2
set style line 4 linecolor rgb '#CC3366' ps 0 linetype 1 linewidth 1.2

#set style line 1 lc rgb '#8b1a0e' pt 1 ps 1 lt 1 lw 2 # --- red
#set style line 2 lc rgb '#5e9c36' pt 6 ps 1 lt 1 lw 2 # --- green

set key outside top right

set xlabel 'running time (s)'
set ylabel 'throughput (MB/s)'
set y2label 'CPU (%)'
set xrange [0:120]
set yrange [10:30]
set y2range [0:105]
set y2tics

plot 'load1-load.dat' u ($1-1555511305):(10/$2) t 'vm1\_tp'    w lp ls 1 axes x1y1, \
     'load1-load.dat' u ($1-1555511305):3       t 'vm1\_cpu'   w lp ls 2 axes x1y2, \
     'load2-load.dat' u ($1-1555511305):(10/$2) t 'vm2\_tp'    w lp ls 3 axes x1y1, \
	  'load2-load.dat' u ($1-1555511305):3       t 'vm2\_cpu'   w lp ls 4 axes x1y2
