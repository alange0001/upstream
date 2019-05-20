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
set style line 1 lc rgb '#33CC66' ps 0 lt 1 lw 2
set style line 2 lc rgb '#3399FF' ps 0 lt 1 lw 1.2
set style line 3 lc rgb '#CC3366' ps 0 lt 1 lw 1.2

#set style line 1 lc rgb '#8b1a0e' pt 1 ps 1 lt 1 lw 2 # --- red
#set style line 2 lc rgb '#5e9c36' pt 6 ps 1 lt 1 lw 2 # --- green

set key outside top right

set xlabel 'running time (s)'
set ylabel 'CPU (%)'
set xrange [0:120]
set yrange [-3:103]

plot 'host-sys.dat'  u ($1-1555511305):2 t 'cpu\_host' w lp ls 1, \
     'load1-sys.dat' u ($1-1555511305):2 t 'cpu\_vm1'   w lp ls 2, \
     'load2-sys.dat' u ($1-1555511305):2 t 'cpu\_vm2'   w lp ls 3
