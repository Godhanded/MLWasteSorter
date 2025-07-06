# Waste Classification Model (Biological, Recyclabale, Trash)

## steps to run 

- create virtual environment
```
python -m venv venv
```

- enter virtual environment
```
source venv/bin/activate

or

venv/Scripts/activate
```

- install dependencies
```
pip install tensorflow scipy numpy pillow opencv-python  
```

- run the desired scripts
```
python sorter.py
```

```
python imgchk.py
```

**note**: preprocessor.py trains the model, mobilenetv2_trashident_model.h5 is the trained model
sorter.py runs the model with live webcam, imgchk.py runs the model but with local pictures

## WebApp
To run the web application, follow these steps:
1. Install Flask and Flask-SocketIO:
   ```
   pip install Flask Flask-SocketIO
   ```
2. see readme in waste_classifier_main folder or click [here](https://github.com/Godhanded/MLWasteSorter/tree/master/waste_classifier_main)



**Link to images used for training**: https://www.kaggle.com/datasets/mostafaabla/garbage-classification