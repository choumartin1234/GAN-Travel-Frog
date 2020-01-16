package com.ying.travelfrogg;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

public class TravelActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_travel);

        // Get the Intent that started this activity and extract the string
        Intent intent = getIntent();
        String message = intent.getStringExtra(NameActivity.EXTRA_MESSAGE);

        // Capture the layout's TextView and set the string as its text
        TextView textViewMouse = findViewById(R.id.textViewMouseAsk);
        textViewMouse.setText(message + "，这一次你要去什么地方旅游呀？");
    }

    public void seePicture(View view) {
        Intent intent = new Intent(this, DisplayActivity.class);
        startActivity(intent);
    }
}
