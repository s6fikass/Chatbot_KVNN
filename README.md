# Chatbot_KVNN

to run the model in training mode:

```
python3 pytourch_main.py
```

To evaluate a certain model :

```
python3 pytourch_main.py -tm < model path > -vv 1
```

Every update in textdata.py needs reprocessing and deleting files in ```data/samples```


#Arguments to use:
    -e  => Number of Epochs to Run ,default=1500
    
    -es => Size of the embedding   ,default=200
    
    -b  =>  batch size to use      ,default=10, type=int)
    
    -tm => Location of trained model,default=None
   
    -vv => evaluate  model   , default=False
    
    -cuda => to use cuda  , default=False