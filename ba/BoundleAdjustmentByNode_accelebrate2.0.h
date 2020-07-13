#include<Eigen/Core>
#include<Eigen/Dense>
#include "Eigen/Eigen"
#include "Eigen/SparseQR"
#include <Eigen/Sparse>
#include "jet.h"
#include "point_camera.h"
#include <stdio.h>
#include "stdlib.h"
#include "algorithm"
#include <queue>
#include <set>
#include <vector>
#include <map>
#include <math.h>
#include <string>
#include <algorithm>
#include <iomanip>
#define MAXSTEP 1e8
#define MINSTEP 1e-9
#define MINDIF 1e-6
#define PARAMETERMIN 1e-8
#define max_consecutive_nonmonotonic_steps 0
#define OUTTER_ITERATION_NUMBER 50
#define INNER_ITERATION_NUMBER 50
#define INITMIU 1e4
using namespace std;

namespace BoundleAdjustment
{

class Problem
{
public:
    Problem();
    ~Problem();
    struct Residual_block
    {
        Residual_block(int a,int b,int row,int col,int offset_,BoundleAdjustment::CostFunction *node):
            parameter_a(a),
            parameter_b(b),
            size_a(offset_-1),
            size_b(col-offset),
            offset(offset_)
        {
            //jacobi_block.resize(row,col);
            //jacobi_block.setZero();
            jacobi_parameter_1.resize(row,offset_-1);
            jacobi_parameter_1.setZero();
            jacobi_parameter_2.resize(row,col-offset_);
            jacobi_parameter_2.setZero();
            hessian_W.resize(size_a,size_b);
            hessian_W.setZero();
            residual.resize(row);
            residual.setZero();
            residual_node = node;
        }
        int parameter_a;
        int parameter_b;
        int offset;
        int size_a;
        int size_b;
        BoundleAdjustment::CostFunction* residual_node;
        //Eigen::MatrixXd jacobi_block;
        Eigen::MatrixXd jacobi_parameter_1;
        Eigen::MatrixXd jacobi_parameter_2;
        Eigen::MatrixXd hessian_W;
        Eigen::VectorXd residual;
    };
    struct Parameter
    {
        Parameter(int n):size(n)
        {
            params.resize(n);
            candidate.resize(n);
            delta.resize(n);
            residual.resize(n);
            jacobi_scaling.resize(n);
            jacobi_scaling.setZero();
            hessian.resize(n,n);
            hessian.setZero();
            hessian_inverse.resize(n,n);
            hessian_inverse.setZero();
            params.setZero();
            candidate.setZero();
            delta.setZero();
            residual.setZero();
        }
        int size;
        Eigen::VectorXd params;
        Eigen::VectorXd candidate;
        Eigen::VectorXd delta;
        Eigen::VectorXd residual;
        Eigen::VectorXd jacobi_scaling;
        Eigen::MatrixXd hessian;
        Eigen::MatrixXd hessian_inverse;
    };
    vector<Residual_block* > residual_block;
    vector<Parameter* > parameter_1_vector;
    vector<Parameter* > parameter_2_vector;
    map<double*, int> parameter_1_map;
    map<double*, int> parameter_2_map;
    int parameter_a_size;
    Eigen::MatrixXd Schur_A;
    Eigen::VectorXd Schur_B;
    bool update_parameter();
    bool checkParameter_1(double *parameter_1);
    bool checkParameter_2(double *parameter_2);
    template <class T,int N,int N1,int N2>
    void addParameterBlock(double *parameter_1,double *parameter_2,BoundleAdjustment::CostFunction* costfunction);
    void pre_process();
    void init_scaling();
    void solve();
    inline void schur_complement();
    void pre_construct_schur();
    void post_process();
    //bool cmp(int A,int B);
    vector< vector< Residual_block* > > parameter_2_link;
};
bool cmp( BoundleAdjustment::Problem::Residual_block* A,BoundleAdjustment::Problem::Residual_block* B)
{
    if (A->parameter_a<B->parameter_a) return true;
    return false;
}
Problem::Problem():parameter_a_size(0){}
Problem::~Problem()
{
    for (int i=0;i<parameter_1_vector.size();++i)
    {
        delete(parameter_1_vector[i]);
    }
    for (int i=0;i<parameter_2_vector.size();++i)
    {
        delete(parameter_2_vector[i]);
    }
    for (int i=0;i<residual_block.size();++i)
    {
        //residual_node_vector[i]->destroy();
        delete(residual_block[i]);
    }
}
void Problem::pre_process()
{
    /* pre_process need to be before solve()
     * in the process,the numbers of parameter_a and parameter_b are known.So need construct these matrix vecotrs:
     * the vector<MatrixXd> parameter, the vector<MatrixXd> residual, the MatrixXd schur_matrix
     *
    */
    //Need a construct like map<(Ci,Cj),(vector<P,Wi,Wj>)>
    //And then scan the map to add to matrix.
    //scan can do in the schur_complement().
    //Notice that if the camera and point are not changed,the map would only construct once.
    //So the map would like map<pair<int Ci,int Cj>,vector<struct:<int Pi,residual_node *Wi,residual_node *Wj>>  >
    //use the parameter_2_link,which is a link_list as Point--->residual_node--->next residual_node
    /*
     *                                                                                  |
     *                                                                                  |
     *                                                                                  V
     *                                                                           nextPoint
     * In each point we can find the pair<int Ci,int Cj>. Since the schur matrix is Symmetric Matrix,we only need Ci<Cj
     * so in this function we order the link_list by id.
     * Compute with the link happened in the function schur_complement().
     */
    Schur_A.resize(parameter_a_size,parameter_a_size);
    Schur_B.resize(parameter_a_size);
    Schur_A.setZero();
    Schur_B.setZero();
    for (int i=0;i<parameter_2_link.size();++i)
    {
        sort(parameter_2_link[i].begin(),parameter_2_link[i].end(),cmp);
    }
}
bool Problem::update_parameter()
{
    double step_norm=0.0;
    for (int i=0;i<parameter_1_vector.size();++i)
    {
        parameter_1_vector[i]->params =  parameter_1_vector[i]->candidate;
        parameter_1_vector[i]->hessian.setZero();
        parameter_1_vector[i]->residual.setZero();
        step_norm = step_norm + parameter_1_vector[i]->delta.norm();
    }
    for (int i=0;i<parameter_2_vector.size();++i)
    {
        parameter_2_vector[i]->params =  parameter_2_vector[i]->candidate;
        parameter_2_vector[i]->hessian.setZero();
        parameter_2_vector[i]->residual.setZero();
        step_norm = step_norm + parameter_2_vector[i]->delta.norm();
    }
    if (step_norm<PARAMETERMIN)
    {
        return false;
    }
    return true;
}
bool Problem::checkParameter_1(double *parameter_1)
{
  if ( parameter_1_map.find(parameter_1)!=parameter_1_map.end()) return true;
  return false;
}
bool Problem::checkParameter_2(double *parameter_2)
{
  if ( parameter_2_map.find(parameter_2)!=parameter_2_map.end()) return true;
  return false;
}
template <int N,int N1,int N2>
void Problem::addParameterBlock(double *parameter_1,double *parameter_2,BoundleAdjustment::CostFunction* new_residual_node)
{
    //for saving time,we don't check the camera and point if there is
    //already a same tuple.So the input tuple must be different.
    //cout<<N1<<" "<<N2<<endl;

    if (!checkParameter_1(parameter_1))
    {
        Parameter* new_parameter = new Parameter(N1);
        new_parameter->params = Eigen::Map<Eigen::VectorXd>(parameter_1,N1);
        parameter_1_map.insert(std::pair<double *, int >(parameter_1,parameter_1_vector.size()));
        parameter_1_vector.push_back(new_parameter);
        parameter_a_size = parameter_a_size + N1;
    }
    if (!checkParameter_2(parameter_2))
    {
        Parameter* new_parameter = new Parameter(N2);
        new_parameter->params = Eigen::Map<Eigen::VectorXd>(parameter_2,N2);
        parameter_2_map.insert(std::pair<double *, int >(parameter_2,parameter_2_vector.size()));
        parameter_2_vector.push_back(new_parameter);
        vector<Residual_block* > parameter_2_list;
        parameter_2_link.push_back(parameter_2_list);
    }
    Residual_block *block = new Residual_block(parameter_1_map[parameter_1],parameter_2_map[parameter_2],N,N1+N2+1,1+N1,new_residual_node);
    int id = block->parameter_b;
    parameter_2_link[id].push_back(block);
    residual_block.push_back(block);

}

/****************************************schur complement********************************/
inline void Problem::schur_complement()
{
    int length_camera = parameter_1_vector.size();
    int test;
    int parameter_2_link_size = parameter_2_link.size();
    for (int i=0;i<parameter_2_link_size;++i)
    {
        int inner_size = parameter_2_link[i].size();
        for (int j=0;j<inner_size;++j)
        {
            int id_1 = parameter_2_link[i][j]->parameter_a;
            int size = parameter_2_link[i][j]->size_a;
            Eigen::MatrixXd WT_Einv;
            WT_Einv.noalias()=parameter_2_link[i][j]->hessian_W*parameter_2_vector[i]->hessian_inverse;
            Schur_B.segment(id_1*size,size).noalias() -=
                    //Schur_B.segment(id_1*size,size)-
                     WT_Einv*parameter_2_vector[i]->residual;
            for (int k=j;k<inner_size;++k)
            {

                int id_2 = parameter_2_link[i][k]-> parameter_a;
                Schur_A.block(id_1*size,id_2*size,size,size).noalias() -=
                        //Schur_A.block(id_1*size,id_2*size,size,size) -
                        WT_Einv*parameter_2_link[i][k]->hessian_W.transpose();
            }
        }
    }
    //Schur_A.bottomLeftCorner(n1*length_camera,n1*length_camera).triangularView<Eigen::Lower>()= Schur_A.transpose();
    Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> llt =Schur_A.selfadjointView<Eigen::Upper>().llt();
    Schur_B.noalias() = llt.solve(Schur_B);
    //schur_B.noalias() = schur_A.llt().solve(schur_B);
    //schur_B.noalias() = schur_A.colPivHouseholderQr().solve(schur_B);
    for (int i=0;i<length_camera;++i)
    {
        parameter_1_vector[i]->delta = Schur_B.segment(i*parameter_1_vector[i]->size,parameter_1_vector[i]->size);
    }
}

/****************************************end schur complement********************************/
void Problem::init_scaling()
{
    int parameter_1_length = parameter_1_vector.size();
    int parameter_2_length = parameter_2_vector.size();
    for (int i=0;i<parameter_1_length;++i)
    {
        parameter_1_vector[i]->jacobi_scaling = 1.0/(1.0+sqrt(parameter_1_vector[i]->jacobi_scaling.array()));
    }
    for (int i=0;i<parameter_2_length;++i)
    {
        parameter_2_vector[i]->jacobi_scaling = 1.0/(1.0+sqrt(parameter_2_vector[i]->jacobi_scaling.array()));
    }
}
void Problem::solve()
{
    int outcount=0;
    double minimum_cost = -1;
    double old_residual_cost =0.0;
    double model_cost=0.0;
    int num_consecutive_nonmonotonic_steps=0;
    double accumulated_reference_model_cost_change=0;
    double accumulated_candidate_model_cost_change=0;
    double current_cost;
    double candidate_cost;
    double reference_cost;
    double Miu=INITMIU;
    double v0=2.0;
    clock_t t1 = clock();
    int parameter_1_vector_length =parameter_1_vector.size();
    int parameter_2_vector_length =parameter_2_vector.size();
    int residual_node_length = residual_block.size();
    double residual_cost = 0.0;
    for (int i=0;i<residual_node_length;++i)
    {
        residual_block[i]->residual_node->computeJacobiandResidual(
                &parameter_1_vector[residual_block[i]->parameter_a]->params,
                &parameter_2_vector[residual_block[i]->parameter_b]->params,
                &residual_block[i]->jacobi_parameter_1,
                &residual_block[i]->jacobi_parameter_2,
                &residual_block[i]->residual
                );
        //Eigen::VectorXd jacobi_squared = residual_block[i]->jacobi_block.colwise().squaredNorm();
        residual_cost =residual_cost + residual_block[i]->residual.squaredNorm();
        parameter_1_vector[residual_block[i]->parameter_a]->jacobi_scaling +=
                residual_block[i]->jacobi_parameter_1.colwise().squaredNorm();
        parameter_2_vector[residual_block[i]->parameter_b]->jacobi_scaling +=
                residual_block[i]->jacobi_parameter_2.colwise().squaredNorm();
    }
    int test;
    residual_cost /=2;
    pre_process();
    init_scaling();
    //string summary="iteration| inner_loop|   new_residual    |   old_residual  |  step_norm   |   radius   |  iter time";
    while (outcount<OUTTER_ITERATION_NUMBER)
    {
        for (int i=0;i<residual_node_length;++i)
        {

            residual_block[i]->jacobi_parameter_1 =residual_block[i]->jacobi_parameter_1.array().rowwise()*parameter_1_vector[residual_block[i]->parameter_a]->jacobi_scaling.transpose().array();
            residual_block[i]->jacobi_parameter_2 =residual_block[i]->jacobi_parameter_2.array().rowwise()*parameter_2_vector[residual_block[i]->parameter_b]->jacobi_scaling.transpose().array();
             residual_block[i]->hessian_W.noalias() = residual_block[i]->jacobi_parameter_1.transpose()*residual_block[i]->jacobi_parameter_2;
        }

        if (outcount!=0 && abs(residual_cost-old_residual_cost)/old_residual_cost < MINDIF)
        {
            cout<<"leave by MINDIF reached!"<<endl;
            break;
        }else
        {
            ++outcount;
            double totaltime=(double)(clock()-t1)/CLOCKS_PER_SEC;
            cout<<"\n第"<<outcount<<"循环的运行时间为"<<totaltime<<"秒！"<<endl;
            cout<<"diff is :"<<abs(residual_cost-old_residual_cost)/old_residual_cost<<endl;
            t1=clock();
            old_residual_cost = residual_cost;
        }

        if (outcount<=1)
        {
            candidate_cost=residual_cost;
            reference_cost=residual_cost;
        }
        current_cost=residual_cost;
        accumulated_candidate_model_cost_change+=model_cost;
        accumulated_reference_model_cost_change+=model_cost;
        if (outcount==1||current_cost<minimum_cost)
        {
            minimum_cost=current_cost;
            num_consecutive_nonmonotonic_steps=0;
            candidate_cost=current_cost;
            accumulated_candidate_model_cost_change=0;
        }else
        {
            ++num_consecutive_nonmonotonic_steps;
            if (current_cost > candidate_cost)
            {
                candidate_cost = current_cost;
                accumulated_candidate_model_cost_change=0.0;
            }
        }
        if (num_consecutive_nonmonotonic_steps ==  max_consecutive_nonmonotonic_steps)
        {
            reference_cost = candidate_cost;
            accumulated_reference_model_cost_change =  accumulated_candidate_model_cost_change;
        }
      /*****************begin the inner loop******************************/
      int innercount=0;
      double r=-1.0;

      double new_residual_cost;
      while(innercount<INNER_ITERATION_NUMBER)
      {
          ++innercount;
          //do make schur_matrix residual_c parameter_2_hessian init
          Schur_A.setZero();
          Schur_B.setZero();
          for (int i=0;i<residual_node_length;++i)
          {
              int id_a = residual_block[i]->parameter_a;
              int id_b = residual_block[i]->parameter_b;
              int size = residual_block[i]->size_a;
              Schur_A.block(id_a*size,id_a*size,size,size).noalias()+=
                      //Schur_A.block(id_a*size,id_a*size,size,size)+
                      residual_block[i]->jacobi_parameter_1.transpose()*residual_block[i]->jacobi_parameter_1;
              Schur_B.segment(id_a*size,size).noalias() -=
                      //Schur_B.segment(id_a*size,size)-
                       residual_block[i]->jacobi_parameter_1.transpose()*residual_block[i]->residual;
              parameter_2_vector[id_b]->hessian.noalias() +=
                      //parameter_2_vector[id_b]->hessian +
                      residual_block[i]->jacobi_parameter_2.transpose()*residual_block[i]->jacobi_parameter_2;
              parameter_2_vector[id_b]->residual.noalias() -=
                      //parameter_2_vector[id_b]->residual -
                      residual_block[i]->jacobi_parameter_2.transpose()*residual_block[i]->residual;

          }
          /*for (int i=0;i<residual_node_length;++i)
          {
              int id_b = residual_block[i]->parameter_b;
              parameter_2_vector[id_b]->hessian = parameter_2_vector[id_b]->hessian + residual_block[i]->jacobi_parameter_2.transpose()*residual_block[i]->jacobi_parameter_2;
              parameter_2_vector[id_b]->residual =  parameter_2_vector[id_b]->residual - residual_block[i]->jacobi_parameter_2.transpose()*residual_block[i]->residual;

          }*/

          Schur_A.diagonal() = Schur_A.diagonal() + 1/Miu*Schur_A.diagonal();

          for (int i=0;i<parameter_2_vector_length;++i)
          {
              //update the parameter_2_hessian by Miu and compute the inverse

              parameter_2_vector[i]->hessian.diagonal().noalias() +=
                      //parameter_2_vector[i]->hessian.diagonal()+
                      1/Miu*parameter_2_vector[i]->hessian.diagonal();
              parameter_2_vector[i]->hessian_inverse = parameter_2_vector[i]->hessian.inverse();

          }
          //double totaltime=(double)(clock()-t1)/CLOCKS_PER_SEC;
          //cout<<"\n第"<<outcount<<"循环的schur init时间为"<<totaltime<<"秒！"<<endl;
          //t1=clock();
          schur_complement();
          //totaltime=(double)(clock()-t1)/CLOCKS_PER_SEC;
          //cout<<"\n第"<<outcount<<"循环的schur时间为"<<totaltime<<"秒！"<<endl;
          //t1=clock();
          for (int i=0;i<residual_node_length;++i)
          {

              int id =residual_block[i]->parameter_b;
              parameter_2_vector[id]->residual.noalias() -=
                      //parameter_2_vector[id]->residual -
                      residual_block[i]->hessian_W.transpose()*parameter_1_vector[residual_block[i]->parameter_a]->delta;


          }
          //totaltime=(double)(clock()-t1)/CLOCKS_PER_SEC;
          //cout<<"\n第"<<outcount<<"循环的update_residual时间为"<<totaltime<<"秒！"<<endl;
          //t1=clock();
          for (int i=0;i<parameter_2_vector_length;++i)
          {
              //parameter_2_vector[i]->delta = parameter_2_vector[i]->hessian.llt().solve(parameter_2_vector[i]->residual);
              parameter_2_vector[i]->delta.noalias() = parameter_2_vector[i]->hessian_inverse*parameter_2_vector[i]->residual;
          }
          //totaltime=(double)(clock()-t1)/CLOCKS_PER_SEC;
          //cout<<"\n第"<<outcount<<"循环的solveF时间为"<<totaltime<<"秒！"<<endl;
          //t1=clock();
          model_cost=0.0;
          Eigen::VectorXd delta_parameter;
          for (int i=0;i<residual_node_length;++i)
          {
              int id_1 = residual_block[i]->parameter_a;
              int id_2 = residual_block[i]->parameter_b;
              delta_parameter = residual_block[i]->jacobi_parameter_1*parameter_1_vector[id_1]->delta+residual_block[i]->jacobi_parameter_2*parameter_2_vector[id_2]->delta;
               model_cost= model_cost +(delta_parameter.transpose()*(2*residual_block[i]->residual+delta_parameter))(0,0);


          }
          model_cost = model_cost/2;

          if (model_cost<0)
          {
              for (int i=0;i<parameter_1_vector_length;++i)
              {

                  parameter_1_vector[i]->delta = parameter_1_vector[i]->delta.array()*parameter_1_vector[i]->jacobi_scaling.array();

              }
              for (int i=0;i<parameter_2_vector_length;++i)
              {

                  parameter_2_vector[i]->delta = parameter_2_vector[i]->delta.array()*parameter_2_vector[i]->jacobi_scaling.array();
              }
          }//else continue;
          new_residual_cost=0.0;
          for (int i=0;i<residual_node_length;++i)
          {
              int id_1 = residual_block[i]->parameter_a;
              int id_2 = residual_block[i]->parameter_b;
              parameter_1_vector[id_1]->candidate = parameter_1_vector[id_1]->params+parameter_1_vector[id_1]->delta;
              parameter_2_vector[id_2]->candidate = parameter_2_vector[id_2]->params+parameter_2_vector[id_2]->delta;
              residual_block[i]->residual_node->computeResidual(
                          &parameter_1_vector[id_1]->candidate,
                          &parameter_2_vector[id_2]->candidate,
                          &new_residual_cost);
          }
          new_residual_cost /=2;
          cout<<"newF(X)=:"<<setprecision(12) <<new_residual_cost<<endl<<"oldF(X) is : "<<setprecision(12) <<current_cost<<"model_cost:"<<model_cost<<endl;

          double relative_decrease=(new_residual_cost-current_cost)/model_cost;
          double historical_relative_decrease=(new_residual_cost-reference_cost)/\
                  (accumulated_reference_model_cost_change+model_cost);
          r=relative_decrease<historical_relative_decrease?historical_relative_decrease:relative_decrease;
          //cout<<"miu:"<<Miu<<"  r:"<<r<<endl;
          //cout<<"relative:"<<relative_decrease<<"current:"<<current_cost<<"cost:"<<new_residual_cost<<endl;
          //cout<<"reference_cost_"<<reference_cost<<"accum"<<accumulated_reference_model_cost_change<<endl;
          //cout<<"model"<<model_cost<<std::endl;
          //cout<<"historical:"<<historical_relative_decrease<<endl;

          //cin>>test;
          //totaltime=(double)(clock()-t1)/CLOCKS_PER_SEC;
          //cout<<"\n第"<<outcount<<"循环的后处理时间为"<<totaltime<<"秒！"<<endl;
          //t1=clock();
          if (r>0.001)
          {
              Miu=min(Miu/max(1.0/3.0,1.0-pow((2*r-1.0),3)),1e16);
              v0=2;
              residual_cost = new_residual_cost;
              break;
          }else
          {
              Miu=Miu/v0;
              v0 *= 2;
          }
      }
      /*****************end the inner loop********************************/
      //summary += "    "+  std::to_string(innercount) + "   ";
      //summary += "    "+  std::to_string(new_residual_cost)+"    ";
      //summary += "    "+  std::to_string(current_cost) + "     ";
      if (!update_parameter())
      {
          cout<<"reach the parameter limit :"<<PARAMETERMIN<<endl;
          break;
      }
      //double totaltime=(double)(clock()-t1)/CLOCKS_PER_SEC;
      //cout<<"\n第"<<outcount<<"循环的更新时间为"<<totaltime<<"秒！"<<endl;
      //t1=clock();
      for (int i=0;i<residual_node_length;++i)
      {

              residual_block[i]->residual_node->computeJacobiandResidual(
                          &parameter_1_vector[residual_block[i]->parameter_a]->params,
                          &parameter_2_vector[residual_block[i]->parameter_b]->params,
                          &residual_block[i]->jacobi_parameter_1,
                          &residual_block[i]->jacobi_parameter_2,
                          &residual_block[i]->residual
                      );
      }
      //totaltime=(double)(clock()-t1)/CLOCKS_PER_SEC;
      //cout<<"\n第"<<outcount+1<<"循环的计算时间为"<<totaltime<<"秒！"<<endl;
      //t1=clock();
      //summary += "    "+  std::to_string(r) + "     ";
    }
    post_process();
}
void Problem::post_process()
{
    map<double *,int >::iterator iter;
    for (iter=parameter_1_map.begin();iter!=parameter_1_map.end();++iter)
    {
        Eigen::Map<Eigen::VectorXd>(iter->first, parameter_1_vector[iter->second]->params.rows(),1) = parameter_1_vector[iter->second]->params;
    }
    for (iter=parameter_2_map.begin();iter!=parameter_2_map.end();++iter)
    {
        Eigen::Map<Eigen::VectorXd>(iter->first, parameter_2_vector[iter->second]->params.rows(), 1) = parameter_2_vector[iter->second]->params;
    }
}
}

