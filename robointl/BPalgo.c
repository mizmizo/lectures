
/*-- This program makes neural network model by BP Algorithm --*/

#include <stdio.h>
#include <cv.h>
#include <highgui.h>

#define N 400   //image size
#define n_type 6  //the number of image type
#define d 0     //noise in input images
#define alpha 1.0 //gain
#define eta 0.01 //learning rate
#define num_image 1000

//grobal variables
double w1[N][N];
double w2[N][n_type];


/*-- read 20x20 input image and convert into gray image --*/ 
int read_img(double *q,char *name){
  IplImage *img;
  IplImage *gray;
  int x,y;
  unsigned int p;
  img = cvLoadImage(name, CV_LOAD_IMAGE_COLOR);
  if(img == NULL){
    fprintf(stderr, "couldn't read image!");
    return 0;
  }

  gray = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
  cvCvtColor(img, gray, CV_BGR2GRAY);

  for(y = 0; y < gray->height; y++){
    for(x = 0; x < gray->width; x++){
      p = (int)(unsigned char)gray->imageData[gray->widthStep*y + x];
      q[y*gray->width + x] = p/255.0;
    }
  }
  cvReleaseImage(&img);
  cvReleaseImage(&gray);
  return 1;

}

void init_w(){
  int i,j,k;
  for (i = 0; i < N; i++){
    for (j = 0; j < N; j++){
      if(i == j){
	w1[i][j] = 1.0;
      }else{
	w1[i][j] = 0;
      }
    }
    for(k = 0; k < 6; k++){
      w2[i][k] = 1.0/400;
    }
  }
}

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
  }

  printf("y = ");
  for (i = 0; i < n_type; i++){
    printf("%4.3lf ", y[i]);
  }
  printf("\n");

}

void calc_err(double *y, double *y_ans, double *err){
  int i;
  for (i = 0; i < n_type; i++){
    err[i] = y[i] - y_ans[i];
  }

  printf("err = ");
  for (i = 0; i < n_type; i++){
    printf("%4.3lf ", err[i]);
  }
  printf("\n\n");
}

void calc_new_w(double *x1, double *x2, double *y, double *err){
  int i,j;
  double sigmaout[n_type];
  double sigmamiddle[N];
  double deltaout[n_type];
  double deltamiddle[N];
  //sigmaout
  for (i = 0; i < n_type; i++){
    sigmaout[i] = alpha * y[i] * (1.0 - y[i]);
  }
  //sigmamiddle
  for (i = 0; i < N; i++){
    sigmamiddle[i] = alpha * x2[i] * (1.0 - x2[i]);
  }
  //deltaout
  for (i = 0; i < n_type; i++){
    deltaout[i] = err[i] * sigmaout[i];
  }
  //deltamiddle
  for ( i = 0; i < N; i++){
    for (j = 0; j < n_type; j++){
      if(j == 0){
	deltamiddle[i] = w2[i][j] * deltaout[j] * sigmamiddle[i];
      } else {
 	deltamiddle[i] += w2[i][j] * deltaout[j] * sigmamiddle[i];
      }
    }
  }

  //w2
  for (j = 0; j < n_type; j++){
    for (i = 0; i < N; i++){
      w2[i][j] -= eta * x2[i] * deltaout[j];
    }
  }

  //w1
  for (j = 0; j < N; j++){
    for (i = 0; i < N; i++){
      w1[i][j] -= eta * x1[i] * deltamiddle[j];
    }
  }

}

void BPalgorithm(int n){
  double q[N];
  char name[50];
  int i;
  double x1[N];
  double x2[N];
  double y[n_type];
  double y_ans[n_type] = {1,0,0,0,0,0};
  double err[n_type];

  sprintf(name,"./images/isA/A%d.jpg",n);
  read_img(q,name);
  /* for (i = 0; i < 400; i++){
    printf("%3.2lf ", q[i]);
    if((i+1)%20 == 0)printf("\n\n");
    }*/

  calc_y(q, x1, x2, y);
  calc_err(y, y_ans, err);
  calc_new_w(x1, x2, y, err);
  /*
  for (i = 0; i < 400; i++){
    printf("%3.2lf ", x1[i]);
    if((i+1)%20 == 0)printf("\n\n");
  }
  printf("\n");
  for (i = 0; i < 400; i++){
    printf("%3.2lf ", x2[i]);
    if((i+1)%20 == 0)printf("\n\n");
  }
  */
}


void main (){
  int n;
  n = 1;
  init_w();
  for (n = 1; n <= num_image; n++){
    BPalgorithm(1);
  }

  //write to file
}
