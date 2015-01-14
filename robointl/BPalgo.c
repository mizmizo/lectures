/*==================================================
 This program makes three layer feedforward neural network model 
and enhance it by BP Algorithm.
neural network is consist of follow variables.
the value of neurons in input layer             : vector x1
the value of neurons in middle layer            : vector x2
the value of neurons in  output layer           : vector y
synaptic weight between input and middle layer  : vector w1
synaptic weight between middle and output layer : vector w2
====================================================*/

/*-- include --*/
#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

/*-- define --*/
#define N 400   //image size
#define n_type 6  //the number of image type
#define d 25     //noise in input images
#define alpha 1.0 //gain
#define eta 0.05 //learning rate
#define mu 0.5 //inertia factor
#define num_image 5 //the number of images of each letter
#define num_learn 30 //the number of learning

/*-- grobal variables --*/
double w1[N][N];
double w2[N][n_type];
double delta_w2[N][n_type];
double Erms = 0.0;


/*--------------------------------------------------
read_img function 
 - read 20x20 input image "name" and convert into gray image
   then put image data into q  
----------------------------------------------------*/ 
int read_img(double *q,char *name){
  IplImage *img;
  IplImage *gray;
  int x,y;
  unsigned int p;
  int noise;

  img = cvLoadImage(name, CV_LOAD_IMAGE_COLOR);
  if(img == NULL){
    fprintf(stderr, "couldn't read image!");
    return 1;
  }

  gray = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
  cvCvtColor(img, gray, CV_BGR2GRAY);

  for(y = 0; y < gray->height; y++){
    for(x = 0; x < gray->width; x++){
      p = (int)(unsigned char)gray->imageData[gray->widthStep*y + x];
      noise = rand() % 100;
      if(noise < d){
	p = ((rand() % 255) + 1);
      }
      q[y*gray->width + x] = p/255.0;
    }
  }
  cvReleaseImage(&img);
  cvReleaseImage(&gray);
  return 0;

}


/*--------------------------------------------------
init_w function
 - Initialize w1 and w2 by random score (-0.5 ~ 0.5)
--------------------------------------------------*/
void init_w(){
  int i,j,k;
  for (i = 0; i < N; i++){
    for (j = 0; j < N; j++){
      w1[i][j] = ((double)rand() / RAND_MAX) - 0.5;
    }
    for(k = 0; k < n_type; k++){
      w2[i][k] = ((double)rand() / RAND_MAX) - 0.5;
      delta_w2[i][k] = 0.0;
    }
  }
}


/*--------------------------------------------------
calc_y function
 - calculate x1, x2, y 
--------------------------------------------------*/
void calc_y(double *q, double *x1, double *x2, double *y){
  int i,j;

  //input layer
  for (i = 0; i < N; i++){
    x1[i] = q[i];
  }

  //middle layer
  for (i = 0; i < N; i++){
    for (j = 0; j < N; j++){
      if (j == 0){
	x2[i] = x1[j] * w1[j][i];
      } else {
	x2[i] += x1[j] * w1[j][i];
      }
    }
    x2[i] = 1.0 / (1.0 + exp(-alpha * x2[i]));
  }

  //output layer
  for (i = 0; i < n_type; i++){
    for (j = 0; j < N; j++){
      if(j == 0){
	y[i] = x2[j] * w2[j][i];
      } else {
       	y[i] += x2[j] * w2[j][i];
      }
    }
    y[i] = 1.0 / (1.0 + exp(-alpha * y[i]));
  }

  printf("y   = ");
  for (i = 0; i < n_type; i++){
    printf("%4.3lf ", y[i]);
  }
  printf("\n");
}


/*--------------------------------------------------
calc_err function
 - calculate error 
--------------------------------------------------*/
void calc_err(double *y, double *y_ans, double *err){
  int i;
  for (i = 0; i < n_type; i++){
    err[i] = y[i] - y_ans[i];
    Erms += err[i] * err[i];
  }
  
  printf("err = ");
  for (i = 0; i < n_type; i++){
    printf("%4.3lf ", err[i]);
  }
  printf("\n");
}


/*--------------------------------------------------
calc_new_w function
 - update w1 and w2
--------------------------------------------------*/
void calc_new_w(double *x1, double *x2, double *y, double *err){
  int i,j;
  double sigmaout[n_type];
  double deltaout[n_type];

  //sigmaout
  for (i = 0; i < n_type; i++){
    sigmaout[i] = alpha * y[i] * (1.0 - y[i]);
  }
  //deltaout
  for (i = 0; i < n_type; i++){
    deltaout[i] = err[i] * sigmaout[i];
  }

  //w2
  for (j = 0; j < n_type; j++){
    for (i = 0; i < N; i++){
      w2[i][j] -= eta * x2[i] * deltaout[j] + mu * delta_w2[i][j];
delta_w2[i][j] = eta * x2[i] * deltaout[j] + mu * delta_w2[i][j];

    }
  }  
}


/*--------------------------------------------------
BPalgorithm function
 - operate BPalgorithm 
--------------------------------------------------*/
void BPalgorithm(char* mark, int n){
  double q[N];
  char name[80];
  double x1[N];
  double x2[N];
  int r;
  double y[n_type];
  double y_ans[n_type] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double err[n_type];

  switch(*mark){
  case 'A':
    y_ans[0] = 1.0;
    break;
  case 'B':
    y_ans[1] = 1.0;
    break;
  case 'C':
    y_ans[2] = 1.0;
    break;
  case 'D':
    y_ans[3] = 1.0;
    break;
  case 'E':
    y_ans[4] = 1.0;
    break;
  case 'F':
    y_ans[5] = 1.0;
    break;
  default:
    printf("no answer!\n");
    break;
  }
  
  sprintf(name,"./images/is%s/%s%d.jpg",mark, mark, n);
  if((r = read_img(q,name)) == 0){
    calc_y(q, x1, x2, y);
    calc_err(y, y_ans, err);
    calc_new_w(x1, x2, y, err);
  }
}


/*--------------------------------------------------
max_y function
 - return the address in which the maximum value in y is. 
--------------------------------------------------*/
int max_y(double *y){
  int i;
  double buf = 0.0;
  int max = 0;
  for (i = 0; i < n_type; i++){
    if(y[i] > buf){
      buf = y[i];
      max = i;
    }
  }
  return max;
}

/*--------------------------------------------------
recognize_test function
 - recognize a image of letter and report the result. 
--------------------------------------------------*/
int recognize_test(char* mark, int n, FILE *fp){
  double q[N];
  char name[80];
  double x1[N];
  double x2[N];
  int r,i;
  double y[n_type];
  double y_ans[n_type] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double err[n_type];
  int ans;
  
  switch(*mark){
  case 'A':
    y_ans[0] = 1.0;
    ans = 0;
    break;
  case 'B':
    y_ans[1] = 1.0;
    ans = 1;
    break;
  case 'C':
    y_ans[2] = 1.0;
    ans = 2;
    break;
  case 'D':
    y_ans[3] = 1.0;
    ans = 3;
    break;
  case 'E':
    y_ans[4] = 1.0;
    ans = 4;
    break;
  case 'F':
    y_ans[5] = 1.0;
    ans = 5;
    break;
  default:
    printf("no answer!\n");
    ans = 6;
    break;
  }
  
  sprintf(name,"./images/is%s/%s%d.jpg",mark, mark, n);
  if((r = read_img(q,name)) == 0){
    calc_y(q, x1, x2, y);
    calc_err(y, y_ans, err);
    fprintf(fp, "%s%d.jpg  y = ", mark, n);
    for (i = 0; i < n_type; i++){
      fprintf(fp, "%4.3lf ", y[i]);
    }
    fprintf(fp, " err = ");
    for (i = 0; i < n_type; i++){
      fprintf(fp, "%4.3lf ", err[i]);
    }

    if(ans == max_y(y)){
      fprintf(fp, " correct\n");
      return 1;
    } else {
      fprintf(fp, "error\n");
      return 0;
    }
  }
}


/*--------------------------------------------------
test_report function
 - test the efficiency of neural network and report the result in a file. 
--------------------------------------------------*/
void test_report(){
  int n, correct;
  FILE *fp;
  char name[80];

  sprintf(name, "./result/d-%d_eta-%4.3lf_mu-%4.3lf_learn-%d.txt", d, eta, mu, num_learn);
  if ((fp = fopen(name, "w")) == NULL){
    printf("result file open error!\n");
    exit(0);
  }
  fprintf(fp, "d = %d\neta = %4.3lf\nmu = %4.3lf\nnum_learn = %d\n",d,eta,mu,num_learn);

  correct = 0;
  Erms = 0;
    for (n = 1; n <= num_image; n++){
      printf("%d image\n",n);
      correct += recognize_test("A",n,fp);
      correct += recognize_test("B",n,fp);
      correct += recognize_test("C",n,fp);
      correct += recognize_test("D",n,fp);
      correct += recognize_test("E",n,fp);
      correct += recognize_test("F",n,fp);
    }
    Erms = sqrt(Erms)/(n_type * num_image * num_learn * n_type);
    fprintf(fp, "Erms = %lf\n", Erms);
    fprintf(fp, "correct answers rate = %d / %d\n", correct, n_type * num_image);
  fclose(fp);

    printf("correct answers rate = %d / %d\n", correct, n_type * num_image);
    printf("report finished !\n");

}


void main (){
  int l,n;
  double Erms_;

  srand((unsigned)time(NULL));

  //initialize
  init_w(); 

  //learning
  for (l = 0; l < num_learn; l++){
    printf("\n%d learng\n",(l+1));
    for (n = 1; n <= num_image; n++){
      printf("%d image\n",n);
      BPalgorithm("A",n);
      BPalgorithm("B",n);
      BPalgorithm("C",n);
      BPalgorithm("D",n);
      BPalgorithm("E",n);
      BPalgorithm("F",n);
    }
  }

  //test and report  
  test_report();
}
