# Sequence alignment of representative from X group

All vs. All data. Alignment done by [HH-suite3 v3.1](https://github.com/soedinglab/hh-suite7)  

**IMPORTANT** To save time and space, alignments with probability below 10% and less then 30 residues long are not shown

Loads huge heatmap (2Kx2K) and some details. Histograms contains 100K points. Please be patient, loads can take up to 2 minutes.  

## Required packages 
`numpy => 1.18.4`  
`pandas => 1.03`  
`plotly => 4.7.1`   
`dash => 1.12.0` 

## Steps to generate data for plotting 
- make annotation `make_heatmap_anotation.py`  
- make heatmap plot `make_heatmap.py`  
- run app `main_app.py`  


**TODO**
- Move to server
- Add links to ECOD 
- Load heatmap in WebGL (upload in low res and increase res with zoom in)
- Pre-compile details, figures (may save time on update)
