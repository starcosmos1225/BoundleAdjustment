#include<iostream>
#include <fstream>
#include<ceres/ceres.h>
#include<ceres/rotation.h>
#include<chrono>
#include<cmath>
#include<vector>
#include <time.h>
#ifdef linux
#include <unistd.h>
#include <dirent.h>
#endif
#ifdef WIN32
#include <direct.h>
#include <io.h>
#endif
using namespace std;

struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const {
      T result[3];
      ceres::AngleAxisRotatePoint(camera, point, result);



      result[0]=result[0]+camera[3];
      result[1]=result[1]+camera[4];
      result[2]=result[2]+camera[5];
      T xp=-result[0]/result[2];
      T yp=-result[1]/result[2];
      T r2=xp*xp+yp*yp;
      T distortion=T(1.0)+r2*(camera[7]+camera[8]*r2);
      T predicted_x=camera[6]*distortion*xp;
      T predicted_y=camera[6]*distortion*yp;
      residuals[0]=predicted_x-T(observed_x);
      residuals[1]=predicted_y-T(observed_y);

    return true;
  }

   // Factory to hide the construction of the CostFunction object from
   // the client code.
   static ceres::CostFunction* Create(const double observed_x,
                                      const double observed_y) {
     return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                 new SnavelyReprojectionError(observed_x, observed_y)));
   }

  double observed_x;
  double observed_y;
};
struct observation{
  int camera_id;
  int point_id;
  double observation_x;
  double observation_y;
};
struct Camera{
  double parameter[9];
};
struct Points{
  double parameter[3];
};
vector<string > getData()
{
    vector<string> files;
#ifdef WIN32
        _finddata_t file;
        long lf;
        if ((lf=_findfirst("../data", &file)) == -1) {
                cout<<"data"<<" not found!!!"<<endl;
        } else {
                while(_findnext(lf, &file) == 0) {
                        if (strcmp(file.name, ".") == 0 || strcmp(file.name, "..") == 0)
                                continue;
                        files.push_back(file.name);
                }
        }
        _findclose(lf);
#endif

#ifdef linux
        DIR *dir;
        struct dirent *ptr;
        char base[1000];

        if ((dir=opendir("../data")) == NULL)
        {
                perror("Open dir error...");
                exit(1);
        }

        while ((ptr=readdir(dir)) != NULL)
        {
                if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
                        continue;
                else if(ptr->d_type == 8)
                        files.push_back(ptr->d_name);
                else if(ptr->d_type == 10)
                        continue;
                else if(ptr->d_type == 4)
                {
                        files.push_back(ptr->d_name);
                }
        }
        closedir(dir);
#endif
    sort(files.begin(), files.end());
    return files;
}
int main(int argc,char ** argv)
{
  vector<string> files = getData();
  cout<<"choose the data you want to test:"<<endl;
  for (unsigned int i=0;i<files.size();++i)
      cout<<i<<": "<<files[i]<<endl;
  cout<<"input:";
  int n;
  cin>>n;
  string filename = "../data/"+files[n];
  ifstream infile;
  int cn,pn,obn;

  infile.open(filename);

  vector<Camera> camera;
  vector<Points> point;
  vector<observation> ob;
   //printf("gethere2");
if(infile.is_open())          //文件打开成功,说明曾经写入过东西
{
   infile>>cn>>pn>>obn;
   for (int i=0;i<obn;i++)
   {
     observation temp_ob;
     infile>>temp_ob.camera_id>>temp_ob.point_id>>temp_ob.observation_x>>temp_ob.observation_y;
     ob.push_back(temp_ob);
  }
  for (int i=0;i<cn;i++)
  {
    Camera temp_camera;
    for (int j=0;j<9;j++)
      infile>>temp_camera.parameter[j];
    camera.push_back(temp_camera);
  }
  for (int i=0;i<pn;i++)
  {
    Points temp_point;
    for (int j=0;j<3;j++)
      infile>>temp_point.parameter[j];
    point.push_back(temp_point);
  }
}
 //printf("gethere1");
  infile.close();
  ceres::Problem problem;
  for (int i = 0; i < obn; i++) {
  ceres::CostFunction* cost_function =
      SnavelyReprojectionError::Create(
           ob[i].observation_x,
           ob[i].observation_y);
  problem.AddResidualBlock(cost_function,
                           NULL ,
                           camera[ob[i].camera_id].parameter,
                           point[ob[i].point_id].parameter);
  }
  //printf("gethere");
  ceres::Solver::Options options;
   options.sparse_linear_algebra_library_type=ceres::EIGEN_SPARSE;
  options.linear_solver_type =ceres::DENSE_SCHUR;
  options.preconditioner_type=ceres::JACOBI;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  clock_t start,finish;
  double totaltime;
  start=clock();
  ceres::Solve(options, &problem, &summary);
  finish=clock();
  totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
  cout<<"time is :"<<totaltime<<endl;
  std::cout << summary.FullReport() << "\n";
 
  return 0;
}
