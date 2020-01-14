package com.ying.travelfrogg;

import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

import java.io.File;
//import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

public class DrawActivity extends AppCompatActivity {

    private static final int PERMISSION_REQUEST_CODE = 1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_draw);

        // Get the Intent that started this activity and extract the string
        Intent intent = getIntent();
        String message = intent.getStringExtra(NameActivity.EXTRA_MESSAGE);

        // Capture the layout's TextView and set the string as its text
        TextView textView = findViewById(R.id.textView2);
        textView.setText("为你的" + message + "画一个旅游景点吧！");
    }

    public void savePaintAndGo(View view) {
        // ask for permission
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.M) {

            if (checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                    == PackageManager.PERMISSION_DENIED) {

                Log.d("permission", "permission denied to WRITE_EXTERNAL_STORAGE - requesting it");
                String[] permissions = {Manifest.permission.WRITE_EXTERNAL_STORAGE};

                requestPermissions(permissions, PERMISSION_REQUEST_CODE);
            }
        }

        View drawView = findViewById(R.id.drawingView);

        Log.d("SAVE", "saving an image...");

        Bitmap bitmap = viewToBitmap(drawView);

        try {
            // TODO: save bitmap to app specific space

            String path = Environment.getExternalStorageDirectory().toString();

            File file = new File(path, "sample.png");
            FileOutputStream outputStream = new FileOutputStream(file);
//            System.out.println(outputStream);
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream);
            Log.d("SAVE", "file is compressed, size is " + bitmap.getByteCount());
            outputStream.close();

//            MediaStore.Images.Media.insertImage(getContentResolver(),
//                    file.getAbsolutePath(), file.getName(), file.getName());
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
