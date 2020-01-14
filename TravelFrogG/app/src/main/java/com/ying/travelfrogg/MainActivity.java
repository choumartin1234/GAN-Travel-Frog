package com.ying.travelfrogg;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    public void enterNameActivity(View view) {
        // Do something in response to button
        Intent intent = new Intent(this, NameActivity.class);
        startActivity(intent);
    }
}
