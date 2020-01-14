package com.ying.travelfrogg;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;

import com.ying.travelfrogg.tflite.Generator.Model;
import com.ying.travelfrogg.tflite.Generator.Device;

public class GenerateActivity extends AppCompatActivity {

    private Model model = Model.QUANTIZED;     // default as QUANTIZED
    private Device device = Device.CPU;        // default as CPU
    private int numThreads = -1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_generate);
    }
}
