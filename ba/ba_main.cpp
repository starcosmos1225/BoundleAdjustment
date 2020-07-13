#include <fstream>
#include <iostream>
#include "BoundleAdjustmentByNode_accelebrate.h"
#include "point_camera.h"
#include <ceres/rotation.h>
#ifdef linux
#include <unistd.h>
#include <dirent.h>
#endif
#ifdef WIN32
#include <direct.h>
#include <io.h>
#endif
using namespace BoundleAdjustment;
using namespace std;
class costfunction
{
public:
    double x_;
    double y_;
    costfunction(double x,double y):x_(x),y_(y){}
    template <class T>
    void Evaluate(const T *camera,const T *point,T *residual)
    {
        T result[3];
        ceres::AngleAxisRotatePoint(camera, point, result);
        result[0]=result[0]+camera[3];
        result[1]=result[1]+camera[4];
        result[2]=result[2]+camera[5];
        T xp=-result[0]/result[2];
        T yp=-result[1]/result[2];
        T r2=xp*xp+yp*yp;
        T distortion=1.0+r2*(camera[7]+camera[8]*r2);
        T predicted_x=camera[6]*distortion*xp;
        T predicted_y=camera[6]*distortion*yp;
        residual[0]=predicted_x-x_;
        residual[1]=predicted_y-y_;

    }
};


struct myobservation
{
    int camera_id;
    int point_id;
    double observation[2];
};
struct myCamera
{
    double parameter[9];
};
struct myPoints
{
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
int main()
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
    vector<myCamera> camera;
    vector<myPoints> point;
    vector<myobservation> ob;
    if(infile.is_open())
    {
        infile>>cn>>pn>>obn;
        for (int i=0;i<obn;i++)
        {

            myobservation temp_ob;
            infile>>temp_ob.camera_id>>temp_ob.point_id>>temp_ob.observation[0]>>temp_ob.observation[1];
            ob.push_back(temp_ob);
        }
        for (int i=0;i<cn;i++)
        {
            myCamera temp_camera;
            for (int j=0;j<9;j++)
                infile>>temp_camera.parameter[j];
            camera.push_back(temp_camera);
        }
        for (int i=0;i<pn;i++)
        {
            myPoints temp_point;
            for (int j=0;j<3;j++)
            {
                infile>>temp_point.parameter[j];
            }
            point.push_back(temp_point);
        }
    }
    infile.close();
    Problem<2,9,3> problem;
    for (int i = 0; i < obn; i++)
    {
        Residual_node<costfunction,2,9,3>* residual_node = \
                new Residual_node<costfunction,2,9,3>(new costfunction(ob[i].observation[0],ob[i].observation[1]));
        problem.addParameterBlock(camera[ob[i].camera_id].parameter,\
               point[ob[i].point_id].parameter,residual_node);
    }
    printf("begin solve\n");
    clock_t start,finish;
    double totaltime;
    start=clock();
    problem.solve();
    finish=clock();
    totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"\n此程序的运行时间为"<<totaltime<<"秒！"<<endl;
        return 0;
}
