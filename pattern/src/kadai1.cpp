#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>

#define eta 0.001

using namespace std;
using namespace Eigen;

MatrixXd readdata(char* name1, char* name2)
{
  MatrixXd A(2,100);
  string str;
  int i;
  double a,b;
  ifstream ifs1(name1);
  if(ifs1.fail())
    {
      cerr << "file open error!" << endl;
      exit(1);
    }
  for (i=0;i<50;i++)
    {
      getline(ifs1,str);
      sscanf(str.data(), "%lf %lf", &a, &b);
      //      cout << "a = " << a << " b = " << b << endl;
      A(0,i) = a;
      A(1,i) = b;

      cout << i+1 << " " << A(0,i) << " : " << A(1,i) << endl;
    }
  //Train2.txt
  ifstream ifs2(name2);
  if(ifs2.fail())
    {
      cerr << "file open error!" << endl;
      exit(1);
    }
  for (i=0;i<50;i++)
    {
      getline(ifs2,str);
      sscanf(str.data(), "%lf %lf", &a, &b);
      //      cout << "a = " << a << " b = " << b << endl;
      A(0,i+50) = a;
      A(1,i+50) = b;

      cout << i+51 << " " << A(0,i+50) << " : " << A(1,i+50) << endl;
    }
  return A;
 }

int main()
{
  srand((unsigned int) time(0));
  MatrixXd DATA = readdata("../data/Train1.txt", "../data/Train2.txt");
  MatrixXd w0(1,1);  //bias
  MatrixXd w(2,1);   //weight
  MatrixXd t(1,1);   //teacher signal
  MatrixXd g(1,1);
  MatrixXd dJ1(3,1);     //dJ/dw
  MatrixXd dJ2(3,1);
  int i;
  dJ1 << 10,10,10;
  dJ2 << 10,10,10;
  w0 = MatrixXd::Random(1,1);
  w = MatrixXd::Random(2,1); 
  //  w = w.array() * 0.1;  
  //  cout << w0 << endl;
  cout << w << endl;
  //  cout << w.transpose() << endl;
  //cout << w.transpose().row(0) << endl;
  //cout << DATA << endl;
  while ((dJ1.squaredNorm() + dJ2.squaredNorm()) > 0.001){
    dJ1 = MatrixXd::Zero(3,1); 
    dJ2 = MatrixXd::Zero(3,1); 
    for(i=0;i<100;i++){
      if(i<50){t << 1;
      } else {t << 0;}
      
      g = w.transpose() * DATA.col(i);
      dJ1(0,0) += g(0,0) + w0(0,0) - t(0,0);
      dJ1(1,0) += (g(0,0) + w0(0,0) - t(0,0)) * DATA(0,i);
      dJ1(2,0) += (g(0,0)  + w0(0,0) - t(0,0)) * DATA(1,i);
      cout << i+1 << "data : " << DATA(0,i) << " " << DATA(1,i) << endl;
      cout << "g : " << g << endl;
      cout << "dJ : " << dJ1(0,0) << " " << dJ1(1,0) << " " << dJ1(2,0) << endl;
    }
    /*    for(i=0;i<50;i++){
      dJ2(1,0) += (w.transpose().row(0) * DATA.col(i+50) + w0(0,1) - T(0,1)) * DATA(0,i+50);
      dJ2(2,0) += (w.transpose().row(1) * DATA.col(i+50) + w0(1,1) - T(1,1)) * DATA(1,i+50);
      //cout << i+50 << " = " << dJ2 << endl;
      
      }*/
    
      cout << "dJ1 :" << endl;
      cout << dJ1 << endl;
      /*cout << "dJ2 norm :" << endl;
      cout << dJ2 << endl;
    */
    w0(0,0) -= eta * dJ1(0,0);
    w(0,0) -= eta * dJ1(1,0);
    w(1,0) -= eta * dJ1(2,0);
    //w(0,1) = w(0,1) - eta * dJ2(0,0);
    //w(1,1) = w(1,1) - eta * dJ2(1,0);
    
    cout << w0 << " " << w(0,0) << " " << w(1,0) << endl;
  }
  
}
