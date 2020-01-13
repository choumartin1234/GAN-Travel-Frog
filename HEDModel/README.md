### Edge Detection Based on Pretrained HED model

```bash
python edge.py [args]
```

optional arguments:
  -h, --help          ： show this help message and exit
  --input INPUT  ： Path to image or video. Skip to capture frames from camera
  --prototxt PROTOTXT ： Path to deploy.prototxt
  --caffemodel CAFFEMODEL ：Path to hed_pretrained_bsds.caffemodel
  --width WIDTH      ：Resize input image to a specific width
  --height HEIGHT   ：Resize input image to a specific height
  --savefile SAVEFILE  ： Specifies the output video path
  --all ALL             ：to Parse all jpg files in current  dir

Command to run the edge detection model on all .jpg files in current dir (with same W, H)

```bash
python edge_detector.py --all true --prototxt deploy.prototxt --caffemodel hed_pretrained_bsds.caffemodel --width [W] --height [H]
```

Command to run the edge detection model on one image : 

```python
python edge_detector.py --input [ImagePath] --prototxt deploy.prototxt --caffemodel hed_pretrained_bsds.caffemodel --width [W] --height [H]
```



#### Reference

Modified based on: https://github.com/ashukid/hed-edge-detector,

Add Function to parse all images in a dir