#!/usr/bin/gnuplot -persist
#
#    
#    	G N U P L O T
#    	Version 5.2 patchlevel 8    last modified 2019-12-01 
#    
#    	Copyright (C) 1986-1993, 1998, 2004, 2007-2019
#    	Thomas Williams, Colin Kelley and many others
#    
#    	gnuplot home:     http://www.gnuplot.info
#    	faq, bugs, etc:   type "help FAQ" 180619
#    	immediate help:   type "help"  (plot window ChD: hit 'h')
# set terminal qt 0 font "Sans,9"
# set output
unset clip points
set clip one
unset clip two
set errorbars front 1.000000 
set border 31 front lt black linewidth 1.000 dashtype solid
set zdata 
set ydata 
set xdata 
set y2data 
set x2data 
set boxwidth
set style fill  empty border
set style rectangle back fc  bgnd fillstyle   solid 1.00 border lt -1
set style circle radius graph 0.02 
set style ellipse size graph 0.05, 0.03 angle 0 units xy
set dummy x, y
set format x "% h" 
set format y "% h" 
set format x2 "% h" 
set format y2 "% h" 
set format z "% h" 
set format cb "% h" 
set format r "% h" 
set ttics format "% h"
set timefmt "%d/%m/%y,%H:%M"
set angles radians
set tics back
unset grid
unset raxis
set theta counterclockwise right
set style parallel front  lt black linewidth 2.000 dashtype solid
set key title "" center
set key fixed right top vertical Right noreverse enhanced autotitle nobox
set key noinvert samplen 4 spacing 1 width 0 height 0 
set key maxcolumns 0 maxrows 0
set key noopaque
unset label
unset arrow
set style increment default
unset style line
unset style arrow
set style histogram clustered gap 2 title textcolor lt -1
unset object
set style textbox transparent margins  1.0,  1.0 border  lt -1 linewidth  1.0
set offsets 0, 0, 0, 0
set pointsize 1
set pointintervalbox 1
set encoding default
unset polar
unset parametric
unset decimalsign
unset micro
unset minussign
set view 60, 30, 1, 1
set view azimuth 0
set rgbmax 255
set samples 100, 100
set isosamples 10, 10
set surface 
unset contour
set cntrlabel  format '%8.3g' font '' start 5 interval 20
set mapping cartesian
set datafile separator whitespace
unset hidden3d
set cntrparam order 4
set cntrparam linear
set cntrparam levels 5
set cntrparam levels auto
set cntrparam firstlinetype 0 unsorted
set cntrparam points 5
set size ratio 0 1,1
set origin 0,0
set style data points
set style function lines
unset xzeroaxis
unset yzeroaxis
unset zzeroaxis
unset x2zeroaxis
unset y2zeroaxis
set xyplane relative 0.5
set tics scale  1, 0.5, 1, 1, 1
set mxtics default
set mytics default
set mztics default
set mx2tics default
set my2tics default
set mcbtics default
set mrtics default
set nomttics
set xtics border in scale 1,0.5 mirror norotate  autojustify
set xtics  norangelimit autofreq 
set ytics border in scale 1,0.5 mirror norotate  autojustify
set ytics  norangelimit logscale autofreq 
set ztics border in scale 1,0.5 nomirror norotate  autojustify
set ztics  norangelimit autofreq 
unset x2tics
unset y2tics
set cbtics border in scale 1,0.5 mirror norotate  autojustify
set cbtics  norangelimit autofreq 
set rtics axis in scale 1,0.5 nomirror norotate  autojustify
set rtics  norangelimit autofreq 
unset ttics
set title "" 
set title  font "" textcolor lt -1 norotate
set timestamp bottom 
set timestamp "" 
set timestamp  font "" textcolor lt -1 norotate
set trange [ * : * ] noreverse nowriteback
set urange [ * : * ] noreverse nowriteback
set vrange [ * : * ] noreverse nowriteback
set xlabel "" 
set xlabel  font "" textcolor lt -1 norotate
set x2label "" 
set x2label  font "" textcolor lt -1 norotate
set xrange [ * : * ] noreverse writeback
set x2range [ * : * ] noreverse writeback
set ylabel "" 
set ylabel  font "" textcolor lt -1 rotate
set y2label "" 
set y2label  font "" textcolor lt -1 rotate
set yrange [ * : * ] noreverse writeback
set y2range [ * : * ] noreverse writeback
set zlabel "" 
set zlabel  font "" textcolor lt -1 norotate
set zrange [ * : * ] noreverse writeback
set cblabel "" 
set cblabel  font "" textcolor lt -1 rotate
set cbrange [ * : * ] noreverse writeback
set rlabel "" 
set rlabel  font "" textcolor lt -1 norotate
set rrange [ * : * ] noreverse writeback
unset logscale
set logscale y 10
unset jitter
set zero 1e-08
set lmargin  -1
set bmargin  -1
set rmargin  -1
set tmargin  -1
set locale "fr_FR.UTF-8"
set pm3d explicit at s
set pm3d scansautomatic
set pm3d interpolate 1,1 flush begin noftriangles noborder corners2color mean
set pm3d nolighting
set palette positive nops_allcF maxcolors 0 gamma 1.5 color model RGB 
set palette rgbformulae 7, 5, 15
set colorbox default
set colorbox vertical origin screen 0.9, 0.2 size screen 0.05, 0.6 front  noinvert bdefault
set style boxplot candles range  1.50 outliers pt 7 separation 1 labels auto unsorted
set loadpath 
set fontpath 
set psdir
set fit brief errorvariables nocovariancevariables errorscaling prescale nowrap v5
GNUTERM = "qt"
## Last datafile plotted: "sigma1e-3_ip1_u1/f_convergence_sigma.dat"

set grid

set term postscript enhanced color solid 14
set size 0.65,0.65
set output "case.eps"

set xr [:50]
set yr [:10]

set format y "10^{%L}"

set key box height 1 opaque


plot 'sigma1e-3_ip1_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 1 pt 1 ps 2 lw 3 w l t "(1,1)", \
     'sigma1e-3_ip2_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 2 pt 2 ps 2 lw 3 w l t "(2,1)", \
     'sigma1e-3_ip3_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 3 pt 2 ps 2 lw 3 w l t "(3,1)", \
     'sigma1e-3_ip4_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 4 pt 2 ps 2 lw 3 w l t "(4,1)", \
     'sigma1e-3_ip5_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 6 pt 2 ps 2 lw 3 w l t "(5,1)", \
     'sigma1e-3_ip7_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 7 pt 2 ps 2 lw 3 w l t "(7,1)", \
     'sigma1e-3_ip10_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 8 pt 2 ps 2 lw 3 w l t "(10,1)", \
     'sigma1e-3_ip15_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 9 pt 2 ps 2 lw 3 w l t "(15,1)", \
     'sigma1e-3_ip20_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 10 pt 2 ps 2 lw 3 w l t "(20,1)", \
     'sigma1e-3_p1_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 1 pt 1 ps 1 lw 3 w p t "", \
     'sigma1e-3_p2_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 2 pt 2 ps 1 lw 3 w p t "", \
     'sigma1e-3_p3_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 3 pt 3 ps 1 lw 3 w p t "", \
     'sigma1e-3_p4_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 4 pt 4 ps 1 lw 3 w p t "", \
     'sigma1e-3_p5_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 6 pt 5 ps 1 lw 3 w p t "", \
     'sigma1e-3_p7_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 7 pt 6 ps 1 lw 3 w p t "", \
     'sigma1e-3_p10_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 8 pt 7 ps 1 lw 3 w p t "", \
     'sigma1e-3_p15_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 9 pt 8 ps 1 lw 3 w p t "", \
     'sigma1e-3_p20_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 10 pt 9 ps 1 lw 3 w p t "", \

plot 'sigma1e-1_ip1_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 1 pt 1 ps 2 lw 3 w l t "(1,1)", \
     'sigma1e-1_ip2_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 2 pt 2 ps 2 lw 3 w l t "(2,1)", \
     'sigma1e-1_ip3_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 3 pt 2 ps 2 lw 3 w l t "(3,1)", \
     'sigma1e-1_ip4_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 4 pt 2 ps 2 lw 3 w l t "(4,1)", \
     'sigma1e-1_ip5_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 6 pt 2 ps 2 lw 3 w l t "(5,1)", \
     'sigma1e-1_p1_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 1 pt 1 ps 1 lw 3 w p t "", \
     'sigma1e-1_p2_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 2 pt 2 ps 1 lw 3 w p t "", \
     'sigma1e-1_p3_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 3 pt 3 ps 1 lw 3 w p t "", \
     'sigma1e-1_p4_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 4 pt 4 ps 1 lw 3 w p t "", \
     'sigma1e-1_p5_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 6 pt 5 ps 1 lw 3 w p t "", \

plot 'sigma1_ip1_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 1 pt 1 ps 2 lw 3 w l t "(1,1)", \
     'sigma1_ip2_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 2 pt 2 ps 2 lw 3 w l t "(2,1)", \
     'sigma1_ip3_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 3 pt 2 ps 2 lw 3 w l t "(3,1)", \
     'sigma1_ip4_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 4 pt 2 ps 2 lw 3 w l t "(4,1)", \
     'sigma1_ip5_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 6 pt 2 ps 2 lw 3 w l t "(5,1)", \
     'sigma1_ip7_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 7 pt 2 ps 2 lw 3 w l t "(7,1)", \
     'sigma1_ip10_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 8 pt 2 ps 2 lw 3 w l t "(10,1)", \
     'sigma1_ip15_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 9 pt 2 ps 2 lw 3 w l t "(15,1)", \
     'sigma1_ip20_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 10 pt 2 ps 2 lw 3 w l t "(20,1)", \
     'sigma1_p1_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 1 pt 1 ps 1 lw 3 w p t "", \
     'sigma1_p2_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 2 pt 2 ps 1 lw 3 w p t "", \
     'sigma1_p3_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 3 pt 3 ps 1 lw 3 w p t "", \
     'sigma1_p4_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 4 pt 4 ps 1 lw 3 w p t "", \
     'sigma1_p5_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 6 pt 5 ps 1 lw 3 w p t "", \
     'sigma1_p7_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 7 pt 6 ps 1 lw 3 w p t "", \
     'sigma1_p10_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 8 pt 7 ps 1 lw 3 w p t "", \
     'sigma1_p15_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 9 pt 8 ps 1 lw 3 w p t "", \
     'sigma1_p20_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 10 pt 9 ps 1 lw 3 w p t "", \

plot 'sigma1e1_ip1_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 1 pt 1 ps 2 lw 3 w l t "(1,1)", \
     'sigma1e1_ip2_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 2 pt 2 ps 2 lw 3 w l t "(2,1)", \
     'sigma1e1_ip3_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 3 pt 2 ps 2 lw 3 w l t "(3,1)", \
     'sigma1e1_ip4_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 4 pt 2 ps 2 lw 3 w l t "(4,1)", \
     'sigma1e1_ip5_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 6 pt 2 ps 2 lw 3 w l t "(5,1)", \
     'sigma1e1_p1_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 1 pt 1 ps 1 lw 3 w p t "", \
     'sigma1e1_p2_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 2 pt 2 ps 1 lw 3 w p t "", \
     'sigma1e1_p3_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 3 pt 3 ps 1 lw 3 w p t "", \
     'sigma1e1_p4_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 4 pt 4 ps 1 lw 3 w p t "", \
     'sigma1e1_p5_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 6 pt 5 ps 1 lw 3 w p t "", \

plot 'sigma1e3_ip1_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 1 pt 1 ps 2 lw 3 w l t "(1,1)", \
     'sigma1e3_ip2_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 2 pt 2 ps 2 lw 3 w l t "(2,1)", \
     'sigma1e3_ip3_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 3 pt 2 ps 2 lw 3 w l t "(3,1)", \
     'sigma1e3_ip4_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 4 pt 2 ps 2 lw 3 w l t "(4,1)", \
     'sigma1e3_ip5_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 6 pt 2 ps 2 lw 3 w l t "(5,1)", \
     'sigma1e3_ip7_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 7 pt 2 ps 2 lw 3 w l t "(7,1)", \
     'sigma1e3_ip10_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 8 pt 2 ps 2 lw 3 w l t "(10,1)", \
     'sigma1e3_ip15_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 9 pt 2 ps 2 lw 3 w l t "(15,1)", \
     'sigma1e3_ip20_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 10 pt 2 ps 2 lw 3 w l t "(20,1)", \
     'sigma1e3_p1_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 1 pt 1 ps 1 lw 3 w p t "", \
     'sigma1e3_p2_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 2 pt 2 ps 1 lw 3 w p t "", \
     'sigma1e3_p3_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 3 pt 3 ps 1 lw 3 w p t "", \
     'sigma1e3_p4_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 4 pt 4 ps 1 lw 3 w p t "", \
     'sigma1e3_p5_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 6 pt 5 ps 1 lw 3 w p t "", \
     'sigma1e3_p7_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 7 pt 6 ps 1 lw 3 w p t "", \
     'sigma1e3_p10_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 8 pt 7 ps 1 lw 3 w p t "", \
     'sigma1e3_p15_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 9 pt 8 ps 1 lw 3 w p t "", \
     'sigma1e3_p20_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 10 pt 9 ps 1 lw 3 w p t "", \

plot 'sigma1_ip1_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 1 pt 1 ps 2 lw 3 w l t "(1,1)", \
     'sigma1_ip2_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 2 pt 2 ps 2 lw 3 w l t "(2,1)", \
     'sigma1_ip3_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 3 pt 2 ps 2 lw 3 w l t "(3,1)", \
     'sigma1_ip4_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 4 pt 2 ps 2 lw 3 w l t "(4,1)", \
     'sigma1_ip5_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 6 pt 2 ps 2 lw 3 w l t "(5,1)", \
     'sigma1_ip1_u2/f_convergence_sigma.dat'  every 2 u 1:2 lt 1 pt 1 ps 1 lw 3 w p t "", \
     'sigma1_ip2_u2/f_convergence_sigma.dat'  every 2 u 1:2 lt 2 pt 2 ps 1 lw 3 w p t "", \
     'sigma1_ip3_u2/f_convergence_sigma.dat'  every 2 u 1:2 lt 3 pt 3 ps 1 lw 3 w p t "", \
     'sigma1_ip4_u2/f_convergence_sigma.dat'  every 2 u 1:2 lt 4 pt 4 ps 1 lw 3 w p t "", \
     'sigma1_ip5_u2/f_convergence_sigma.dat'  every 2 u 1:2 lt 6 pt 5 ps 1 lw 3 w p t "", \

set xr [:1000]
set yr [1e-15:]
set logscale

plot 'sigma1e-3_ip1_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 1 pt 1 ps 2 lw 3 w l t "(1,1)", \
     'sigma1e-3_1000iter_ip2_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 2 pt 2 ps 2 lw 3 w l t "(2,1)", \
     'sigma1e-3_1000iter_ip3_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 3 pt 2 ps 2 lw 3 w l t "(3,1)", \
     'sigma1e-3_1000iter_ip4_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 4 pt 2 ps 2 lw 3 w l t "(4,1)", \
     'sigma1e-3_1000iter_ip5_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 5 pt 2 ps 2 lw 3 w l t "(5,1)", \
     'sigma1e-3_1000iter_ip7_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 7 pt 2 ps 2 lw 3 w l t "(7,1)", \
     'sigma1e-3_1000iter_ip10_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 8 pt 2 ps 2 lw 3 w l t "(10,1)", \


set autoscale
unset logscale x

plot 'donneesSI_ip1_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 1 pt 1 ps 2 lw 3 w l t "(1,1)", \
     'donneesSI_ip2_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 2 pt 2 ps 2 lw 3 w l t "(2,1)", \
     'donneesSI_ip3_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 3 pt 2 ps 2 lw 3 w l t "(3,1)", \
     'donneesSI_ip4_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 4 pt 2 ps 2 lw 3 w l t "(4,1)", \
     'donneesSI_ip5_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 5 pt 2 ps 2 lw 3 w l t "(5,1)", \
     'donneesSI_ip7_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 7 pt 2 ps 2 lw 3 w l t "(7,1)", \
     'donneesSI_ip10_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 8 pt 2 ps 2 lw 3 w l t "(10,1)", \
     'donneesSI_ip15_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 9 pt 2 ps 2 lw 3 w l t "(15,1)", \
     'donneesSI_ip20_u1/f_convergence_sigma.dat' every 1 u 1:2 lt 10 pt 2 ps 2 lw 3 w l t "(20,1)", \
     'donneesSI_p1_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 1 pt 1 ps 1 lw 3 w p t "", \
     'donneesSI_p2_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 2 pt 2 ps 1 lw 3 w p t "", \
     'donneesSI_p3_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 3 pt 3 ps 1 lw 3 w p t "", \
     'donneesSI_p4_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 4 pt 4 ps 1 lw 3 w p t "", \
     'donneesSI_p5_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 5 pt 5 ps 1 lw 3 w p t "", \
     'donneesSI_p7_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 7 pt 6 ps 1 lw 3 w p t "", \
     'donneesSI_p10_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 8 pt 7 ps 1 lw 3 w p t "", \
     'donneesSI_p15_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 9 pt 8 ps 1 lw 3 w p t "", \
     'donneesSI_p20_u1/f_convergence_sigma.dat'  every 2 u 1:2 lt 10 pt 9 ps 1 lw 3 w p t "", \
     



! ps2pdf case.eps
! rm case.eps

#    EOF
