package com.ying.travelfrogg.tflite;

import android.app.Activity;

import java.io.IOException;

public class GeneratorFloat extends Generator {
    public GeneratorFloat(Activity activity, Device device, int numThreads)
            throws IOException {
        super(activity, device, numThreads);
    }

    @Override
    protected String getModelPath() {
        // see download.gradle
        return "pix2pix_v1_1.0.tflite";
    }
}
