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

**Link to images used for training**: https://www.kaggle.com/datasets/mostafaabla/garbage-classification