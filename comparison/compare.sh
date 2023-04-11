#brew install gnu-time

gtime -f "%e\t%M\t%t\t%K" Rscript heatmap.R

sleep 5

gtime -f "%e\t%M\t%t\t%K" python heatmap.py

#%e: real time (second)
#%M: maximal memory
#%t: average memory
#%K: total memory
