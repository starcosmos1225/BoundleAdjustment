# BoundleAdjustment
Implement the BA use levenberg-marquardt method. Different with Ceres, I use the block struct to make the compute sparse.
***
# HOW TO RUN:
<br>git clone https://github.com/starcosmos1225/BoundleAdjustment.git
<br>cd BoundleAdjustment/ba
<br>mkdir build
<br>cd build
<br>cmake ..
<br>make -j4
<br>./BoundleAdjustmentByNode_accelebrate
<br>It will appear:
<br>choose the data you want to test:
<br>0: problem-138-19878-pre.txt
<br>1: problem-49-7776-pre.txt
<br>2: test.txt
<br>3: test_1.txt
<br>that is all the data in the file "data"
<br>choose one to optimize it.
<br>The cmake also create a excu "bound_adjustment_ceres". But it depends on the ceres library. If you don't want to compare the running results of ceres, you can comment out the corresponding part in CMakeLists.

