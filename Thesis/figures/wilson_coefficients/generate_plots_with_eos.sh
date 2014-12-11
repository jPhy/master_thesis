mkdir Thesis_plots

OUTFILENAME="all_samples_KDE_cont.svg"
GOF="--gof 0 -4.16611761 --gof 1 0 --gof 2 0 --gof 3 0 --gof 4 0 --gof 5 0 --gof 6 0 --gof 7 0 --gof 8 0 "
Options="all_samples.npy --contours --single-ext svg "
cuts="--cut 0 -6 6 --cut 1 -6 6 --cut 2 -.5 .5 --cut 3 -.5 .5 --cut 4 -.6 .6 --cut 5 -.6 .6 --cut 6 -.7 .5 --cut 7 -.6 .6 "
plot="eos-plot $Options $GOF $cuts"

# C_10 vs. C_10'
$plot --single-2D 0 1 --use-KDE --bandwidth 0.014
mv $OUTFILENAME Thesis_plots/C10_C10p.svg

# C_S vs. C_S'
$plot --single-2D 2 3 --use-KDE --bandwidth 0.02
mv $OUTFILENAME Thesis_plots/CS_CSp.svg

# C_P vs. C_P'
$plot --single-2D 4 5 --use-KDE --bandwidth 0.03
mv $OUTFILENAME Thesis_plots/CP_CPp.svg

# C_T vs. C_T5
# Not too interesting, looks just like a Gauss
$plot --single-2D 6 7 --use-KDE --bandwidth 0.05
mv $OUTFILENAME Thesis_plots/CT_CT5.svg

# C_10 vs. C_P
$plot --single-2D 0 4 --use-KDE --bandwidth .04
mv $OUTFILENAME Thesis_plots/C10_CP.svg

# C_10 vs. C_P'
$plot --single-2D 0 5 --use-KDE --bandwidth .04
mv $OUTFILENAME Thesis_plots/C10_CPp.svg

# C_10' vs. C_P
$plot --single-2D 1 4 --use-KDE --bandwidth .05
mv $OUTFILENAME Thesis_plots/C10p_CP.svg

# C_10' vs. C_P'
$plot --single-2D 1 5 --use-KDE --bandwidth .045
mv $OUTFILENAME Thesis_plots/C10p_CPp.svg

