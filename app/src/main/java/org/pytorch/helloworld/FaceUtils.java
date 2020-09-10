package org.pytorch.helloworld;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;

import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;

public class FaceUtils {
    public static Tensor bitmapToFloat32Tensor(
            final Bitmap bitmap, final float[] normMeanRGB, final float normStdRGB[]) {
        return bitmapToFloat32Tensor(
                bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), normMeanRGB, normStdRGB);
    }
    public static Tensor bitmapToFloat32Tensor(
            final Bitmap bitmap,
            int x,
            int y,
            int width,
            int height,
            float[] normMeanRGB,
            float[] normStdRGB) {

        final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * width * height);
        bitmapToFloatBuffer(bitmap, x, y, width, height, normMeanRGB, normStdRGB, floatBuffer, 0);
        return Tensor.fromBlob(floatBuffer, new long[]{1, 3, height, width});
    }
    public static void bitmapToFloatBuffer(
            final Bitmap bitmap,
            final int x,
            final int y,
            final int width,
            final int height,
            final float[] normMeanRGB,
            final float[] normStdRGB,
            final FloatBuffer outBuffer,
            final int outBufferOffset) {
        checkOutBufferCapacity(outBuffer, outBufferOffset, width, height);

        final int pixelsCount = height * width;
        final int[] pixels = new int[pixelsCount];
        bitmap.getPixels(pixels, 0, width, x, y, width, height);
        final int offset_g = pixelsCount;
        final int offset_b = 2 * pixelsCount;
        for (int i = 0; i < pixelsCount; i++) {
            final int c = pixels[i];
            float r = ((c >> 16) & 0xff) / 1.0f;
            float g = ((c >> 8) & 0xff) / 1.0f;
            float b = ((c) & 0xff) / 1.0f;
//      System.out.print(" "+r+" ;"+g+" ;"+b);
            float rF = (r - normMeanRGB[0]) / normStdRGB[0];
            float gF = (g - normMeanRGB[1]) / normStdRGB[1];
            float bF = (b - normMeanRGB[2]) / normStdRGB[2];
            outBuffer.put(outBufferOffset + i, rF);
            outBuffer.put(outBufferOffset + offset_g + i, gF);
            outBuffer.put(outBufferOffset + offset_b + i, bF);
        }
    }

    private static void checkOutBufferCapacity(FloatBuffer outBuffer, int outBufferOffset, int tensorWidth, int tensorHeight) {
        if (outBufferOffset + 3 * tensorWidth * tensorHeight > outBuffer.capacity()) {
            throw new IllegalStateException("Buffer underflow");
        }
    }

    public static Bitmap PreProcessing(Bitmap bitmap)
    {
        int imgw = bitmap.getWidth();
        int imgh = bitmap.getHeight();
        float scaleHeight;
        float scaleWidth;
        if(imgw > imgh){
            scaleHeight = 640.0f / ((float) imgw);
            scaleWidth = 640.0f / ((float) imgw) ;
        }else {
            scaleWidth = 640.0f / ((float) imgh);
            scaleHeight = 640.0f / ((float) imgh)  ;
        }
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);
        // 得到新的图片
        Bitmap newbm = Bitmap.createBitmap(bitmap, 0, 0, imgw, imgh, matrix,
                true);

        int imgmin = Math.min(newbm.getWidth(),newbm.getHeight());
        int padsize = (int) (640 - imgmin)/2;
        Bitmap mergebitmap = Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888);
        if(newbm.getHeight()>newbm.getWidth()) {
            Bitmap bgBitmap1 = Bitmap.createBitmap(padsize, 640, Bitmap.Config.ARGB_8888);//创建一个新空白位图
            Bitmap bgBitmap2 = Bitmap.createBitmap(640 - padsize, 640, Bitmap.Config.ARGB_8888);
            Canvas canvasadd = new Canvas(mergebitmap);
            canvasadd.drawBitmap(bgBitmap1, 0, 0, null);
            canvasadd.drawBitmap(newbm, padsize, 0, null);
            canvasadd.drawBitmap(bgBitmap2, padsize + newbm.getWidth(), 0, null);
        }else if (newbm.getHeight()<newbm.getWidth()){
            Bitmap bgBitmap1 = Bitmap.createBitmap(640, padsize, Bitmap.Config.ARGB_8888);//创建一个新空白位图
            Bitmap bgBitmap2 = Bitmap.createBitmap(640, 640 - padsize, Bitmap.Config.ARGB_8888);
            Canvas canvasadd = new Canvas(mergebitmap);
            canvasadd.drawBitmap(bgBitmap1, 0, 0, null);
            canvasadd.drawBitmap(newbm, 0, padsize, null);
            canvasadd.drawBitmap(bgBitmap2, 0, padsize + newbm.getWidth(),  null);

        }else {
            mergebitmap = newbm;
        }

        return mergebitmap;
    }
    public static float intersectionOverUnion(int rect1x,int rect1y,int rect1w,int rect1h,int rect2x,int rect2y,int rect2w,int rect2h){
        int leftColumnMax = Math.max(rect1x, rect2x);
        int rightColumnMin = Math.min(rect1w,rect2w);
        int upRowMax = Math.max(rect1y, rect2y);
        int downRowMin = Math.min(rect1h,rect2h);

        if (leftColumnMax>=rightColumnMin || downRowMin<=upRowMax){
            return 0;
        }
        int s1 = (rect1w-rect1x)*(rect1h-rect1y);
        int s2 = (rect2w-rect2x)*(rect2h-rect2y);
        float sCross = (downRowMin-upRowMax)*(rightColumnMin-leftColumnMax);
        return sCross/(s1+s2-sCross);
    }

    public static facebbox[] getAnchors(int imw, int imh){
        int num = 0;
        double fmw1 = Math.ceil(((float) imw) / 16.0f );
        double fmh1 = Math.ceil(((float) imh) / 16.0f );
        double fmw2 = Math.ceil(((float) imw) / 32.0f );
        double fmh2 = Math.ceil(((float) imh) / 32.0f );
        double fmw3 = Math.ceil(((float) imw) / 64.0f );
        double fmh3 = Math.ceil(((float) imh) / 64.0f );

        int totalnum = 2*(((int)fmh1)*((int)fmw1)+((int)fmh2)*((int)fmw2)+((int)fmh3)*((int)fmw3));
        facebbox[] Anchors = new facebbox[totalnum];
        for (int k = 0; k < fmh1; k++) {
            for (int j = 0; j < fmw1; j++) {
                double s_kx = 16.0 / ((double) imw);
                double s_ky = 16.0 / ((double) imh);
                double dense_cx = (j + 0.5) * 16 / ((double) imw);
                double dense_cy = (k + 0.5) * 16 / ((double) imh);
                facebbox boxtmp = new facebbox();
                boxtmp.x1 = (float) dense_cx;
                boxtmp.y1 = (float) dense_cy;
                boxtmp.x2 = (float) s_kx;
                boxtmp.y2 = (float) s_ky;
                Anchors[num] =boxtmp;
                num += 1;

                double s_kx2 = 32.0 / ((double) imw);
                double s_ky2 = 32.0 / ((double) imh);
                double dense_cx2 = (j + 0.5) * 16 / ((double) imw);
                double dense_cy2 = (k + 0.5) * 16 / ((double) imh);
                facebbox boxtmp2 = new facebbox();
                boxtmp2.x1 = (float) dense_cx2;
                boxtmp2.y1 = (float) dense_cy2;
                boxtmp2.x2 = (float) s_kx2;
                boxtmp2.y2 = (float) s_ky2;
                Anchors[num] = boxtmp2;
                num += 1;
            }
        }

        for (int k = 0; k < fmh2; k++) {
            for (int j = 0; j < fmw2; j++) {
                double s_kx = 64.0 / ((double) imw);
                double s_ky = 64.0 / ((double) imh);
                double dense_cx = (j + 0.5) * 32 / ((double) imw);
                double dense_cy = (k + 0.5) * 32 / ((double) imh);
                facebbox boxtmp = new facebbox();
                boxtmp.x1 = (float) dense_cx;
                boxtmp.y1 = (float) dense_cy;
                boxtmp.x2 = (float) s_kx;
                boxtmp.y2 = (float) s_ky;
                Anchors[num] =boxtmp;
                num += 1;

                double s_kx2 = 128.0 / ((double) imw);
                double s_ky2 = 128.0 / ((double) imh);
                double dense_cx2 = (j + 0.5) * 32 / ((double) imw);
                double dense_cy2 = (k + 0.5) * 32 / ((double) imh);
                facebbox boxtmp2 = new facebbox();
                boxtmp2.x1 = (float) dense_cx2;
                boxtmp2.y1 = (float) dense_cy2;
                boxtmp2.x2 = (float) s_kx2;
                boxtmp2.y2 = (float) s_ky2;
                Anchors[num] = boxtmp2;
                num += 1;
            }
        }

        for (int k = 0; k < fmh3; k++) {
            for (int j = 0; j < fmw3; j++) {
                double s_kx = 256.0 / ((double) imw);
                double s_ky = 256.0 / ((double) imh);
                double dense_cx = (j + 0.5) * 64 / ((double) imw);
                double dense_cy = (k + 0.5) * 64 / ((double) imh);
                facebbox boxtmp = new facebbox();
                boxtmp.x1 = (float) dense_cx;
                boxtmp.y1 = (float) dense_cy;
                boxtmp.x2 = (float) s_kx;
                boxtmp.y2 = (float) s_ky;
                Anchors[num] =boxtmp;
                num += 1;

                double s_kx2 = 512.0 / ((double) imw);
                double s_ky2 = 512.0 / ((double) imh);
                double dense_cx2 = (j + 0.5) * 64 / ((double) imw);
                double dense_cy2 = (k + 0.5) * 64 / ((double) imh);
                facebbox boxtmp2 = new facebbox();
                boxtmp2.x1 = (float) dense_cx2;
                boxtmp2.y1 = (float) dense_cy2;
                boxtmp2.x2 = (float) s_kx2;
                boxtmp2.y2 = (float) s_ky2;
                Anchors[num] = boxtmp2;
                num += 1;
            }
        }
        return Anchors;
    }


    /**
     * Copies specified asset to the file in /files app directory and returns this file absolute path.
     *
     * @return absolute file path
     */
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

}
