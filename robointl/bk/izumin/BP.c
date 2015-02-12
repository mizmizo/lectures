#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cv.h>

#define NUM 150     //NUM回学習させる

#define IN 400
#define OUT 5
#define MID 400

typedef struct _Neuro {
  double  Xe[NUM][IN];                // 入力層教師データ(n番めのベクトル)
  double  TO[OUT];               // 出力層教師データ 
  double  mid[MID];                 // 中間層データ
  double  A;                          // 慣性項係数(0.1～1.0～)Hとの関係で決まる　過大：発散
  double  H;                          // 学習係数(0.01～0,5)過大：発散/過小：長時間
  double  valIn[IN];                // 入力層ノード値
  double  valMid[MID];              // 中間層ノード値(中間層の出力値)
  double  valOut[OUT];                // 出力層ノード値(出力層の出力値)
  double  errMid[MID];                // 中間層エラー
  double  errOut[OUT];                // 出力層エラー
  double  Wij[MID][IN];             // 結合係数(入力層→中間層)
  double  incWij[MID][IN];        // Wij増分
  double  Wki[OUT][MID];            // 結合係数(中間層→出力層)
  double  incWki[OUT][MID];      // Wki増分
} Neuro;

Neuro nn, mm;

nn->TO[5] = {1.0, 0.0, 0.0, 0.0, 0.0};

/*-- read 20x20 input image and convert into gray image --*/ 
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

/* -0.5～0.5の乱数を発生させる */
double my_rand(){
  double r;
  srand((unsigned)time(NULL));
  r = rand()/RAND_MAX - 0.5;
  return r;
}

/* sigmoid関数 */
double sigmoid(double x){
  double y;
  y = 1.0 / (1.0 + exp(-x));
  return y;
}

/* sigmoid関数の微分 */
double d_sigmoid(double x){
  double y;
  y = (1.0 - sigmoid(x)) * sigmoid(x);
  return y;
}

/* 変数の初期化 */
void init (Neuro *pN){
  int j, i, k, n;
  pN -> A = 0.6;           //慣性項係数
  pN -> H = 0.03;          //学習係数
  

  
  for (i = 0; i < MID; i++) {   // 係数マトリクス初期化
        for (j = 0; j < IN ; j++){
            pN->Wij[i][j] = (my_rand() - 0.5);  // -0.5～0.5
	    //pN->incInMid[i][j] = 0.0;
        }
    }
    for (k = 0; k < OUT; k++) {
        for (i = 0; i < MID ; i++){
            pN->Wki[k][i] = (my_rand() - 0.5) / 5;  // -0.1～0.1
            //pN->incMidOut[k][i] = 0.0;
        }
    }
}


/* nR番目のデータについて、入力されたベクトルについて出力を計算する */
double update_net (Neuro *pN, int nR){
  int j, i, k;
  double norm_y;
      //===== ①中間層ユニットの出力計算  =====//
    for (i = 0; i < MID; i++) {
      double v = 0;
      // double v = pN->wWij[i][IN] * 1.0;          // しきい値
        for (j = 0; j < IN; j++)
	  v += pN->Wij[i][j] * pN->Xe[nR][j];
        pN->valMid[i+1] = sigmoid(v);
    }
    //===== ②出力層ユニットの出力計算 =====//
    for (k = 0; k < OUT; k++) {
      double w = 0;
      // double v = pN->Wkj[nO][MID] * 1.0;        // しきい値
        for (i = 0; i < MID+1; i++)
            w += pN->Wki[k][i] * pN->valMid[i];
        pN->valOut[k] = sigmoid(w);
    }
  
}





//nR番めのデータについてBP学習
double back_prop(Neuro *pN, int nR){
  int j, i, k;
  //===== ③教師データと出力層ユニットの出力を比較し、誤差計算 =====//
  for (k = 0; k < OUT; k++) {// 出力層誤差計算
    pN->errOut[k] =(pN->valOut[k] - pN->TO[k]) * (1.0 - pN->valOut[k]) * pN->valOut[k];
  }
  for (i = 0; i < MID + 1; i++){ // 中間層誤差計算
    pN->errMid[i] = 0.0;
    for (k = 0; k < OUT; k++)
      pN->errMid[i] += pN->errOut[k] * pN->Wki[k][i];
    pN->errMid[i] *= (1.0 - pN->valMid[i]) * pN->valMid[i];
  }
  //==== ④中間層－出力層間の結合荷重の修正 =====//
  for (k = 0; k < OUT; k++) {
    for (i = 0; i < MID + 1; i++) {
      pN->incWki[k][i]  = pN->A * pN->incWki[k][i] - pN->H * pN->errOut[k] * pN->valMid[i];
      pN->Wki[k][i] += pN->incWki[k][i];
    }
  }
  //==== ⑤入力層－中間層間の結合荷重の修正 =====//
  for (i = 0; i < MID; i++) {
    for (j = 0; j < IN + 1; j++) {
      pN->incWij[i][j]  = pN->A * pN->incWij[i][j] - pN-> H *pN->errMid[i] * pN->Xe[nR][j];
      pN->Wij[i][j] += pN->incWij[i][j];
    }
  }
}




//========================== ［学習フェーズ］==================================//
void learn(Neuro *pN) {
    int     n, nT, nO;          // 学習回数
    int     maxLearn = 300;     // 学習打ち切り回数
    double  errMax;             // エラーの最大値
    int     nErr;               // エラーの大きいノード数
    double  err;               //nErrの割合
    double  eps = 0.03;

    init(pN);   // 結合係数の初期化
    for (n = 0; n < maxLearn; n++) {
        errMax = 0.0;
        nErr = 0;
        for (nT = 0; nT < NUM; nT += 2) {
            // 偶数番目のデータを学習データとする
          //  pN->valIn[IN] = 1.0;        // 閾値用
            update_net(pN, nT);
            back_prop(pN, nT);
            for (nO = 0; nO < OUT; nO++) {      // 出力層最大誤差
                double e = fabs(pN->valOut[nO] - pN->TO[nO]);
                if (e >= 0.5) nErr++;
                if (e > errMax) errMax = e;
            }
        }
        err = (double)nErr / OUT;
        printf("%d %.5f %.3f %d\n", n, errMax, err, nErr);
        if (errMax < eps || err < 0.01) break;
    }
    printf("学習終了\n");
}

//========================== ［認識フェーズ］==================================//
void recognize(Neuro *pN){
  update_net(pN);

 //norm_y
  norm_y = 0;
  for (i = 0; i < OUT; i++){
    norm_y += y[i] * y[i];
  }
  norm_y = sqrt(norm_y);

  for (i = 0; i < OUT; i++){
    //y[i] = y[i]/norm_y;
  }
  printf("y = ");
  for (i = 0; i < OUT; i++){
    printf("%lf ", y[i]);
  }
  printf(" norm = %lf\n",norm_y);

  return norm_y;
  
}


void main() {
  
  int a;
  init(nn);
    //ファイル読み込み
  for (a = 0; a < NUM; a++){
    sprintf(name,"./Pictures/A/%d.jpg",a);
    if((r = read_img(*(nn->Xe[0][0]),name)) == 1){
      return 0;
    } else {
      // 学習を行う
      learn(&nn);
    }
  }

  //認識する
  sprintf(name, "./Pictures/test/A.jpg");
  init(mm);
 if((r = read_img(*(mm->Xe[0][0]),name)) == 1){
      return 0;
    } else {
      // 学習を行う
   update_net(&nn,1);
   
    }
  }  

      /*    
    // 認識を行う
    recognize(&nn);
    */
}
