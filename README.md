**Project Title:**
face-mask-detector
**Datasets:**
img folder --sample(100 images for evaluation)
           |     --cloth    (25 images)  
           |     --n95      (25 images)
           |     --nomask   (25 images)
           |     --surgical (25 images)
            --train(1600 images for train)
           |     --cloth    (400 images)  
           |     --n95      (400 images)
           |     --nomask   (400 images)
           |     --surgical (400 images)
**Source code:**
a)training
masks_detector_train.py 
Running this code to transform the images of the dataset in uniform, use the built CNN to train
the model, and will generate the evaluation results used by the random split test set. And it will
save the trained model in model.pkl.
b)application
masks_detector.py
Running this code to load the saved trained model to evaluate with the sample data(100 images).
**Trained model:**
model.pkl
**Project report:**



