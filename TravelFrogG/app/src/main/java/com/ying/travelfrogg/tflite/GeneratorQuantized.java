package com.ying.travelfrogg.tflite;

import android.app.Activity;

import java.io.IOException;

public class GeneratorQuantized extends Generator {

    public GeneratorQuantized(Activity activity, Device device, int numThreads)
            throws IOException {
        super(activity, device, numThreads);
    }

    @Override
    protected String getModelPath() {
        // from download.gradle or asset
        return "pix2pix_v1_1.0_quant.tflite";
    }
}
