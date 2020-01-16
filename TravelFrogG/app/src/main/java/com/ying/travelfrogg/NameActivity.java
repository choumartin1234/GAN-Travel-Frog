package com.ying.travelfrogg;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.EditText;

public class NameActivity extends AppCompatActivity {

    public static final String EXTRA_MESSAGE = "com.ying.travelfrogg.MESSAGE";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_name);
    }

    public void sendName(View view) {
        // Do something in response to button
        Intent intent = new Intent(this, ChooseActivity.class);
        EditText editText = findViewById(R.id.editText);
        String message = editText.getText().toString();
        intent.putExtra(EXTRA_MESSAGE, message);
        startActivity(intent);
    }
}
