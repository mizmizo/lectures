#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>

#define eta 0.001

using namespace std;
using namespace Eigen;

MatrixXd DATA(2,100);
MatrixXd testDATA(3,1);

MatrixXd readdata_one_out(char* name1, char* name2, int N,int j)
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
      if(i!=j){
      A(0,i) = a;
      A(1,i) = b;
      //cout << i+1 << " " << A(0,i) << " : " << A(1,i) << endl;
      } else {
  	testDATA(0,0) = a;
	testDATA(1,0) = b;
	testDATA(2,0) = 1; //class label
	A(0,i) = a;
	A(1,i) = b;

	//cout << i+1 << " " << testDATA(0,0) << " : " << testDATA(1,0) << endl;
      } 
    }
  
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
      if(i!=j){
	A(0,i) = a;
	A(1,i) = b;
	//cout << i+1 << " " << A(0,i) << " : " << A(1,i) << endl;
      } else {
	testDATA(0,0) = a;
	testDATA(1,0) = b;
	testDATA(2,0) = -1; //class label
	A(0,i) = a;
	A(1,i) = b;
	//cout << i+1 << " " << testDATA(0,0) << " : " << testDATA(1,0) << endl;
      } 
    }
  return A;
}

double k_NN(int k)
{
  int j;
  int result,correct;
  double rcorrect;
  MatrixXd Dist(1,100);
  MatrixXd NN(2,k);
  int cnt1,cnt2;
  correct = 0;
  
  for (j=0;j<100;j++){
    DATA = readdata_one_out("../data/Train1.txt", "../data/Train2.txt", 50,j);
    int i,h,l;
    
    for (i=0;i<k;i++){
      NN(0,i) = 1000;
    }
    
    //calc distance
    for(i=0;i<100;i++){
      Dist(0,i) = (testDATA(0,0) - DATA(0,i)) * (testDATA(0,0) - DATA(0,i)) + (testDATA(1,0) - DATA(1,i)) * (testDATA(1,0) - DATA(1,i));
    }
    
    for(i=0;i<100;i++){
      if(i != j){
	//cout << NN << endl;
	//cout << Dist(0,i) << endl;
	for(h=0;h<k;h++){
	  if(Dist(0,i) < NN(0,h)){
	    for(l=k-1;l>h;l--){
	      NN(0,l) = NN(0,l-1);
	      NN(1,l) = NN(1,l-1);
	    }
	    NN(0,h) = Dist(0,i);
	    if(i<50){
	      NN(1,h) = 1;
	    } else {
	      NN(1,h) = -1;
	    }
	    break;
	  }
	}
      }
    }
    
    cnt1 = 0;
    cnt2 = 0;
    for(i=0;i<k;i++){
      if(NN(1,i) == 1){
	cnt1++;
      } else if(NN(1,i) == -1){
	cnt2++;
      }
    }
    
    if((cnt1>cnt2 && j<50) || (cnt1<cnt2 && j>=50)){
      correct += 1;
      cout << j+1 << " data : " << testDATA(0,0) << " " << testDATA(1,0) << endl;
      cout <<"class1 " << cnt1 << " class2 " << cnt2 << "  correct!" << endl;
    } else {
      cout << j+1 << " data : " << testDATA(0,0) << " " << testDATA(1,0) << endl;
      cout <<"class1 " << cnt1 << " class2 " << cnt2 << "  false!" << endl;
    }  
  }

  rcorrect = correct/100.0;
  return rcorrect;
}


void plot(double* r){
  int i;
  ofstream ofs("../result/kadai3.txt");
  if(ofs.fail())
    {
      cerr << "file open error!" << endl;
      exit(1);
    }

  for(i=0;i<10;i++){
    ofs << i+1 << " " << r[i] << endl;
  }

  FILE *gp;
  gp = popen("gnuplot -persist","w");
  fprintf(gp, "plot \"../result/kadai3.txt\" using 1:2 with lines\n");
  fprintf(gp, "set yr[0:1]\n");
  fprintf(gp, "replot\n");
  fprintf(gp, "set terminal png\n");
  fprintf(gp, "set out \"../result/kadai3_result.png\"\n");
  fprintf(gp, "replot\n"); 

  
  pclose(gp);
}


 

int main(){
  srand((unsigned int) time(0));
  int k;
  double r[10];
  for(k=1;k<=10;k++){
    r[k-1] = k_NN(k);
  }
  for(k=0;k<10;k++){
  cout << r[k] <<  endl;
  }

   plot(r);
  
}
