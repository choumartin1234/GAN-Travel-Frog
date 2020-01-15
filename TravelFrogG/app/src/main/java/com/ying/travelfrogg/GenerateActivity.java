package com.ying.travelfrogg;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;

import com.ying.travelfrogg.tflite.Generator;
import com.ying.travelfrogg.tflite.Generator.Model;
import com.ying.travelfrogg.tflite.Generator.Device;

import java.io.FileInputStream;
import java.io.IOException;

public class GenerateActivity extends AppCompatActivity {

    private Generator generator;

    private Model model = Model.QUANTIZED;     // default as QUANTIZED
    private Device device = Device.CPU;        // default as CPU
    private int numThreads = 4;

    Bitmap inputImage;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_generate);

        /** Decode input image */
        String filename = getIntent().getStringExtra("image");

        try {
            FileInputStream is = this.openFileInput(filename);
            inputImage = BitmapFactory.decodeStream(is);
            is.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

        Log.d("INPUT", "input image decoded, bitmap size " + inputImage.getByteCount());

        generateImage();
    }

    private void generateImage() {
        /** Generating generator */
        try {
            generator = Generator.create(this, model, device, numThreads);
        } catch (IOException e) {
            e.printStackTrace();
        }

        if (generator != null) {
            ImageView genImageView = findViewById(R.id.genImageView);

            Bitmap outputImage = generator.generateImage(inputImage);

//            Log.d("GEN", "generated bitmap with byte count " + outputImage.getByteCount());

            genImageView.setImageBitmap(outputImage);
        }
    }
}
