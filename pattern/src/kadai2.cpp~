#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>

#define eta 0.001

using namespace std;
using namespace Eigen;

MatrixXd DATA(2,100);
MatrixXd testDATA(2,40);
MatrixXd w0(1,2);  //bias
MatrixXd w(2,2);   //weight

MatrixXd readdata(char* name1, char* name2, int N)
{
  MatrixXd A(2,N*2);
  string str;
  int i;
  double a,b;
  ifstream ifs1(name1);
  if(ifs1.fail())
    {
      cerr << "file open error!" << endl;
      exit(1);
    }
  for (i=0;i<N;i++)
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
  for (i=N;i<N*2;i++)
    {
      getline(ifs2,str);
      sscanf(str.data(), "%lf %lf", &a, &b);
      //      cout << "a = " << a << " b = " << b << endl;
      A(0,i) = a;
      A(1,i) = b;

      cout << i+1 << " " << A(0,i) << " : " << A(1,i) << endl;
    }
  return A;
 }

void training()
{
  DATA = readdata("../data/Train1.txt", "../data/Train2.txt", 50);
  MatrixXd t(2,2);   //teacher signal
  MatrixXd g(2,1);
  MatrixXd dJ(3,2);     //dJ/dw
  int i;

  //Initialize just to enter while loop
  dJ << 10,10,10,10,10,10;

  t = MatrixXd::Identity(2,2);
  w0 = MatrixXd::Random(1,2);
  w = MatrixXd::Random(2,2); 
  //  w = w.array() * 0.1;  
  //  cout << w0 << endl;
  cout << w << endl;
  //  cout << w.transpose() << endl;
  //cout << w.transpose().row(0) << endl;
  //cout << DATA << endl;
  while (dJ.squaredNorm() > 0.001){
    dJ = MatrixXd::Zero(3,2); 

    for(i=0;i<50;i++){

      g(0,0) = w.transpose().row(0) * DATA.col(i) + w0(0,0);
      dJ(0,0) += g(0,0) - t(0,0);
      dJ(1,0) += (g(0,0) - t(0,0)) * DATA(0,i);
      dJ(2,0) += (g(0,0) - t(0,0)) * DATA(1,i);

      g(1,0) = w.transpose().row(1) * DATA.col(i) + w0(0,1);
      dJ(0,1) += g(1,0) - t(1,0);
      dJ(1,1) += (g(1,0) - t(1,0)) * DATA(0,i);
      dJ(2,1) += (g(1,0) - t(1,0)) * DATA(1,i);

      cout << i+1 << "data : " << DATA(0,i) << " " << DATA(1,i) << endl;
      cout << "g : " << g(0,0) << " " << g(1,0) << endl;
      cout << "dJ : " << dJ(0,0) << " " << dJ(1,0) << " " << dJ(2,0) << endl;
      cout << "     " << dJ(0,1) << " " << dJ(1,1) << " " << dJ(2,1) << endl;

    }

    for(i=50;i<100;i++){

      g(0,0) = w.transpose().row(0) * DATA.col(i)+w0(0,0);
      dJ(0,0) += g(0,0) - t(0,1);
      dJ(1,0) += (g(0,0) - t(0,1)) * DATA(0,i);
      dJ(2,0) += (g(0,0)  - t(0,1)) * DATA(1,i);

      g(1,0) = w.transpose().row(1) * DATA.col(i) + w0(0,1);
      dJ(0,1) += g(1,0) - t(1,1);
      dJ(1,1) += (g(1,0) - t(1,1)) * DATA(0,i);
      dJ(2,1) += (g(1,0) - t(1,1)) * DATA(1,i);

      cout << i+1 << "data : " << DATA(0,i) << " " << DATA(1,i) << endl;
      cout << "g : " << g(0,0) << " " << g(1,0) << endl;
      cout << "dJ : " << dJ(0,0) << " " << dJ(1,0) << " " << dJ(2,0) << endl;
      cout << "     " << dJ(0,1) << " " << dJ(1,1) << " " << dJ(2,1) << endl;

    }
    
    cout << "dJ :" << endl;
    cout << dJ << endl;
    
    w0(0,0) -= eta * dJ(0,0);
    w(0,0) -= eta * dJ(1,0);
    w(1,0) -= eta * dJ(2,0);
    w0(0,1) -= eta * dJ(0,1);
    w(0,1) -= eta * dJ(1,1);
    w(1,1) -= eta * dJ(2,1);
    
    cout << w0(0,0) << " " << w(0,0) << " " << w(1,0) << endl;
    cout << w0(0,1) << " " << w(0,1) << " " << w(1,1) << endl;
  }
  
}


void test()
{
  MatrixXd g(2,1);
  int i,result,correct;
  double rcorrect;
  testDATA = readdata("../data/Test1.txt", "../data/Test2.txt", 20);
  correct = 0;

  ofstream ofs("../result/kadai1_result.txt");
  if(ofs.fail())
    {
      cerr << "file open error!" << endl;
      exit(1);
    }


  for (i=0;i<20;i++){
    g(0,0) = w.transpose().row(0) * testDATA.col(i) + w0(0,0);
    g(1,0) = w.transpose().row(1) * testDATA.col(i) + w0(0,1);

    if(g(0,0) > g(1,0)){
      result = 1;
      correct += 1;
      cout << i+1 << " data result : " << result << "  correct!" << endl;
      ofs << i+1 << " data result : " << result << "  correct!" << endl;
    } else {
      result = 2;
      cout << i+1 << " data result : " << result << "  false!" << endl;
      ofs << i+1 << " data result : " << result << "  false!" << endl;
    }
  }
  for (i=20;i<40;i++){
    g(0,0) = w.transpose().row(0) * testDATA.col(i) + w0(0,0);
    g(1,0) = w.transpose().row(1) * testDATA.col(i) + w0(0,1);
    
    if(g(0,0) > g(1,0)){
      result = 1;
      cout << i+1 << " data result : " << result << "  false!" << endl;
      ofs << i+1 << " data result : " << result << "  false!" << endl;
    } else {
      result = 2;
      correct += 1;
      cout << i+1 << " data result : " << result << "  correct!" << endl;
      ofs << i+1 << " data result : " << result << "  correct!" << endl;
    }
  }
  rcorrect = correct/40.0;
  cout << "correct ratio : " << rcorrect << endl;
  ofs << "correct ratio : " << rcorrect << endl;
}
 



void plot(){
  FILE *gp;
  double a, b;
  gp = popen("gnuplot -persist","w");
  fprintf(gp, "plot \"../data/Test1.txt\" using 1:2\n");
  fprintf(gp, "replot \"../data/Test2.txt\" using 1:2\n");
  fprintf(gp, "replot \"../data/Train1.txt\" using 1:2\n");
  fprintf(gp, "replot \"../data/Train2.txt\" using 1:2\n");
  if (w(1,0) != w(1,1)){
    a = (w(0,1) - w(0,0)) / (w(1,0) - w(1,1));
    b = (w0(0,1) - w0(0,0)) / (w(1,0) - w(1,1));
    fprintf(gp, "replot %lf*x + %lf\n", a, b);
  } else {
    a = (w0(0,0) - w0(0,1)) / (w(0,1) - w(0,0));
    fprintf(gp, "replot x = %lf\n",a);
  }
  fprintf(gp, "set terminal png\n");
  fprintf(gp, "set out \"../result/kadai1_result.png\"\n");
  fprintf(gp, "replot\n"); 

  
  pclose(gp);
	  

}


 

int main(){
  srand((unsigned int) time(0));
  
  training();
  test();
  plot();
  
}
