#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>

#define eta 0.001

using namespace std;
using namespace Eigen;

MatrixXd DATA(2,100);
MatrixXd testDATA(2,40);

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
      A(0,i) = a;
      A(1,i) = b;
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
      A(0,i) = a;
      A(1,i) = b;
            
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
  
    DATA = readdata("../data/Train1.txt", "../data/Train2.txt", 50);
    testDATA = readdata("../data/Test1.txt", "../data/Test2.txt", 20);
    int i,h,l;

    //test
    for (j=0;j<40;j++){
      
      for (i=0;i<k;i++){
	NN(0,i) = 1000;
      }
      
      //calc distance
      for(i=0;i<100;i++){
	Dist(0,i) = (testDATA(0,j) - DATA(0,i)) * (testDATA(0,j) - DATA(0,i)) + (testDATA(1,j) - DATA(1,i)) * (testDATA(1,j) - DATA(1,i));
      }
      
      for(i=0;i<100;i++){
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
      
      cnt1 = 0;
      cnt2 = 0;
      for(i=0;i<k;i++){
	if(NN(1,i) == 1){
	  cnt1++;
	} else if(NN(1,i) == -1){
	  cnt2++;
	}
      }
      
      if((cnt1>cnt2 && j<20) || (cnt1<cnt2 && j>=20)){
	correct += 1;
	cout << j+1 << " data : " << testDATA(0,j) << " " << testDATA(1,j) << endl;
	cout <<"class1 " << cnt1 << " class2 " << cnt2 << "  correct!" << endl;
      } else {
	cout << j+1 << " data : " << testDATA(0,j) << " " << testDATA(1,j) << endl;
	cout <<"class1 " << cnt1 << " class2 " << cnt2 << "  false!" << endl;
      }  
    }

  rcorrect = correct/40.0;
  return rcorrect;
}


int main(){
  double r = k_NN(3);
  
  cout << r <<  endl;
}
