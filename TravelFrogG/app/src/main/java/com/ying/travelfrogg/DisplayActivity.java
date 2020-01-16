package com.ying.travelfrogg;

import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;

import java.io.FileOutputStream;
import java.io.InputStream;

public class DisplayActivity extends AppCompatActivity {

    private static final int PERMISSION_REQUEST_CODE = 1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_display);

        ImageView view = findViewById(R.id.imageViewRand);

        try {
            InputStream ims = getAssets().open("heds/(1).jpg");
            // load image as Drawable
            Drawable d = Drawable.createFromStream(ims, null);
            // set image to ImageView
            view.setImageDrawable(d);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void goSeeResult(View view) {
        // ask for permission
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.M) {

            if (checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                    == PackageManager.PERMISSION_DENIED) {

                Log.d("permission", "permission denied to WRITE_EXTERNAL_STORAGE - requesting it");
                String[] permissions = {Manifest.permission.WRITE_EXTERNAL_STORAGE};

                requestPermissions(permissions, PERMISSION_REQUEST_CODE);
            }

            if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE)
                    == PackageManager.PERMISSION_DENIED) {

                Log.d("permission", "permission denied to READ_EXTERNAL_STORAGE - requesting it");
                String[] permissions = {Manifest.permission.READ_EXTERNAL_STORAGE};

                requestPermissions(permissions, PERMISSION_REQUEST_CODE);
            }
        }

        View drawView = findViewById(R.id.imageViewRand);

        Log.d("SAVE", "saving an image...");

        Bitmap bitmap = viewToBitmap(drawView);

        try {
//            String path = Environment.getExternalStorageDirectory().toString();
//
//            File file = new File(path, "sample.png");
//            FileOutputStream outputStream = new FileOutputStream(file);
//            bitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream);
//            Log.d("SAVE", "file is compressed, size is " + bitmap.getByteCount());
//            outputStream.close();

            // ===========================================================================
            // Write file
            String filename = "sample.png";
            FileOutputStream stream = this.openFileOutput(filename, Context.MODE_PRIVATE);
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);

            // Finish write
            stream.close();
            bitmap.recycle();

            // Go to generation activity
            Intent intent = new Intent(this, GenerateActivity.class);
            intent.putExtra("image", filename);
            startActivity(intent);

        } catch(Exception e) {
            e.printStackTrace();
        }
    }

    public Bitmap viewToBitmap(View view) {
        Bitmap bitmap = Bitmap.createBitmap(view.getWidth(), view.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bitmap);
        view.draw(canvas);
        return bitmap;
    }
}
