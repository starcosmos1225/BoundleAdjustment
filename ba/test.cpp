#include "stdlib.h"
#include "algorithm"
#include <queue>
#include <set>
#include <vector>
#include <map>
#include <math.h>
#include<iostream>
#include <fstream>
#include "Eigen/Core"
#include<Eigen/Dense>
#include "Eigen/Eigen"
#include "Eigen/SparseQR"
#include <Eigen/Sparse>
#include "jet.h"
#include <time.h>
#include<Eigen/SVD>
using namespace std;

int main()
{
    Eigen::Matrix<double,9,3> a;
    a.setOnes();
    Eigen::Matrix<double,3,9> b;
    b.setOnes();
    Eigen::Matrix<double,9,9> c;
    c = a.lazyProduct(b);
    cout<<c<<endl;
    return 0;
}
