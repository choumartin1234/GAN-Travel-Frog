package com.ying.travelfrogg;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;

public class ChooseActivity extends AppCompatActivity {

    public static final String EXTRA_MESSAGE = "com.ying.travelfrogg.MESSAGE";

    private String name;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_choose);

        // Get the Intent that started this activity and extract the string
        Intent intent = getIntent();
        name = intent.getStringExtra(NameActivity.EXTRA_MESSAGE);

        // Capture the layout's TextView and set the string as its text
        TextView textView = findViewById(R.id.textView4);
        textView.setText("为你的" + name + "选择旅游的模式...");
    }

    public void goToTravelMode(View view) {
        Intent intent = new Intent(this, TravelActivity.class);
        intent.putExtra(EXTRA_MESSAGE, name);
        startActivity(intent);
    }

    public void goToDarkMode(View view) {
        Intent intent = new Intent(this, DrawActivity.class);
        intent.putExtra(EXTRA_MESSAGE, name);
        startActivity(intent);
    }
}
