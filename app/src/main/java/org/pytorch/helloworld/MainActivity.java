package org.pytorch.helloworld;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Paint;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.graphics.Matrix;
import android.widget.ImageView;
import android.widget.TextView;
import android.graphics.Canvas;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    Bitmap bitmap = null;
    Module module = null;
    try {
      // creating bitmap from packaged into app android asset 'image.jpg',
      // app/src/main/assets/image.jpg
      bitmap = BitmapFactory.decodeStream(getAssets().open("ym.jpg"));
      // loading serialized torchscript module from packaged into app android asset model.pt,
      // app/src/model/assets/model.pt
      module = Module.load(FaceUtils.assetFilePath(this, "mbv2.pt"));
      System.out.println("Succeed load model ...");
    } catch (IOException e) {
      Log.e("Pytorch ClS", "Error reading assets", e);
      finish();
    }

    //************************* resize and padding to 640*640 *********************//
    Bitmap mergebitmap= FaceUtils.PreProcessing(bitmap);

    // preparing input tensor
//    float[] face_mean = new float[]{104.0f, 117.0f, 123.0f};
    float[] face_mean = new float[]{116.0f, 117.0f, 111.0f};   //offset to {104.0f, 117.0f, 123.0f}
    float[] face_std = new float[]{1.0f, 1.0f, 1.0f};
    final Tensor inputTensor = FaceUtils.bitmapToFloat32Tensor(mergebitmap,
            face_mean, face_std);

    System.out.println(inputTensor.getDataAsFloatArray().length);
//    System.out.println("inputTensor " +inputTensor.numel());
    // running the model
    long startTime = System.currentTimeMillis();
    final IValue[] outputTensor = module.forward(IValue.from(inputTensor)).toTuple();
    long endTime = System.currentTimeMillis();
    System.out.println("程序运行时间：" + (endTime - startTime) / 2 + "ms");
    long infertime = (endTime - startTime) / 2;
    System.out.println("Out put length : " + outputTensor.length);
    //*************************** bbox ******************************//
    IValue bbox = outputTensor[0];
    Tensor boxt = bbox.toTensor();
    float[] facebox = boxt.getDataAsFloatArray();

    IValue cls = outputTensor[1];
    Tensor clst = cls.toTensor();
    float[] facecls = clst.getDataAsFloatArray();

//    IValue ldm = outputTensor[2];
//    Tensor ldmt = ldm.toTensor();
//    float[] faceldm = ldmt.getDataAsFloatArray();

    System.out.println("face box length : " + facebox.length);
    System.out.println("face cls length : " + facecls.length);
//    System.out.println("face landmark length : " + faceldm.length);
    int imw = mergebitmap.getWidth();
    int imh = mergebitmap.getHeight();

    double fmw1 = Math.ceil(((float) imw) / 16.0f );
    double fmh1 = Math.ceil(((float) imh) / 16.0f );
    double fmw2 = Math.ceil(((float) imw) / 32.0f );
    double fmh2 = Math.ceil(((float) imh) / 32.0f );
    double fmw3 = Math.ceil(((float) imw) / 64.0f );
    double fmh3 = Math.ceil(((float) imh) / 64.0f );

    int totalnum = 2*(((int)fmh1)*((int)fmw1)+((int)fmh2)*((int)fmw2)+((int)fmh3)*((int)fmw3));
    float maxcls = 0.0f;
    float maxx = 0.0f;
    float maxy = 0.0f;
    float maxw = 0.0f;
    float maxh = 0.0f;
    float[] faceconf = new float[totalnum];
    int[] faceidx = new int[totalnum];
    int clsnum = 0;
    int maxidx = 0;
    facebbox[] fbbox = new facebbox[totalnum];
    for (int i = 0; i < facebox.length; i = i+4) {
      int clsidx = (int) (i/4);
      float clsconf = facecls[2 *clsidx+1];
      facebbox boxtmp = new facebbox();
      boxtmp.score = clsconf;
      boxtmp.x1 = facebox[i];
      boxtmp.y1 = facebox[i + 1];
      boxtmp.x2 = facebox[i + 2];
      boxtmp.y2 = facebox[i + 3];
      fbbox[clsidx] = boxtmp;
      if(clsconf > 0.2){
        faceconf[clsnum] = clsconf;
        faceidx[clsnum] = clsidx;
        clsnum +=1;
      }
      if (clsconf > maxcls) {
        maxcls = clsconf;
        maxidx = clsidx;
      }
    }
    System.out.println("MAX :"+maxcls);
    System.out.println("MAX idx:"+maxidx);
//
//    // ************************************* PriorBox ***********************************//
    facebbox[] Anchors = FaceUtils.getAnchors(imw,imh);

//    //***************************** decode bbox from anchor and pred *********************************//
//     ************************************ 按照置信度降序排列 **************************************** //
    float[] PredConfOr = new float[clsnum];
    int[] PredBoxX = new int[clsnum];
    int[] PredBoxY = new int[clsnum];
    int[] PredBoxW = new int[clsnum];
    int[] PredBoxH = new int[clsnum];

    Bitmap bitmap2 = mergebitmap.copy(Bitmap.Config.ARGB_8888, true);

    Canvas canvas = new Canvas(bitmap2);
    Paint paint = new Paint();
    paint.setColor(Color.RED);
    paint.setStyle(Paint.Style.STROKE);
    paint.setStrokeWidth(3);

    for(int k=0;k<clsnum;k++) {
      for (int j = 0; j < clsnum - k - 1; j++) {
        if (faceconf[j] < faceconf[j + 1]) {
          float tmp = faceconf[j];
          faceconf[j] = faceconf[j + 1];
          faceconf[j + 1] = tmp;
          int idxtmp = faceidx[j];
          faceidx[j] = faceidx[j+1];
          faceidx[j+1] = idxtmp;

        }
      }
    }
    for(int k=0;k<1;k++) {
      int idxtmp = faceidx[k];
      double ax = Anchors[faceidx[k]].x1;
      double ay = Anchors[faceidx[k]].y1;
      double aw = Anchors[faceidx[k]].x2;
      double ah = Anchors[faceidx[k]].y2;

      double bboxx = ax + maxx * 0.1 * aw;
      double bboxy = ay + maxy * 0.1 * ah;
      double bboxw = aw * Math.exp(maxw * 0.2);
      double bboxh = ah * Math.exp(maxh * 0.2);
      float boxconf = faceconf[faceidx[k]];

      bboxx = bboxx - bboxw / 2;
      bboxy = bboxy - bboxh / 2;
      bboxw = bboxw + bboxx;
      bboxh = bboxh + bboxy;

      PredBoxX[k] = (int) Math.round( bboxx*640 );
      PredBoxY[k] = (int) Math.round( bboxy*640 );
      PredBoxW[k] = (int) Math.round( bboxw*640 );
      PredBoxH[k] = (int) Math.round( bboxh*640 );
      canvas.drawRect(PredBoxX[k], PredBoxY[k],PredBoxW[k], PredBoxH[k], paint);
    }
    //***************************** NMS ***********************************//
    // TODO //
//    int[] ConfIdx = new int[clsnum];
//    float[] BoxX = new float[clsnum];
//    float[] BoxY = new float[clsnum];
//    float[] BoxW = new float[clsnum];
//    float[] BoxH = new float[clsnum];
//    for (int k=0;k<clsnum;k++){
//      for (int j=0;j<clsnum-k-1;j++){
//        if (faceconf[j]<faceconf[j+1]){
//          float conf_tmp = faceconf[j];
//          faceconf[j] = faceconf[j+1];
//          faceconf[j+1] = conf_tmp;
//
//          int idx_tmp = faceidx[j];
//          faceidx[j] = faceidx[j+1];
//          faceidx[j+1] = idx_tmp;
//        }
//      }
//    }
//    int[] bboxx1 = new int[clsnum];
//    int[] bboxy1 = new int[clsnum];
//    int[] bboxx2 = new int[clsnum];
//    int[] bboxy2 = new int[clsnum];
//    int boxnum = 0;
//    for (int k=0;k<clsnum;k++){
//      for (int j=k+1;j<clsnum;j++){
//        float iou = FaceUtils.intersectionOverUnion(PredBoxX[k],PredBoxY[k],PredBoxW[k],PredBoxH[k],PredBoxX[j],PredBoxY[j],PredBoxW[j],PredBoxH[j]);
//        if(iou > 0.4){}
//      }
//
//    }


    ImageView imageView = findViewById(R.id.image);
    imageView.setImageBitmap(bitmap2);
    String className = "置信度:"+maxcls+"\n时间 : " + infertime + " ms";
    // showing className on UI
    TextView textView = findViewById(R.id.text);
    textView.setText(className);
  }

  public static float max(float [] array){
    float max=0;
    int i=0;
    for(i=0;i<array.length;i++){
      if(array[i]>max){
        max=array[i];
      }
    }
    return max;
  }


}
