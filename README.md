<pre>
**Project Title:** 
face-mask-detector
**Datasets:**
img folder --gender(200 images for bias evaluation)
             |--female(100 images for bias evaluation)
               |--cloth    (25 images)  
               |--n95      (25 images)
               |--nomask   (25 images)
               |--surgical (25 images)
             |--male(100 images for bias evaluation)
               |--cloth    (25 images)  
               |--n95      (25 images)
               |--nomask   (25 images)
               |--surgical (25 images)
            --race(200 images for bias evaluation)
             |--visible majority(100 images for bias evaluation)
                |--cloth    (25 images)  
                |--n95      (25 images)
                |--nomask   (25 images)
                |--surgical (25 images)
             |--visible minority(100 images for bias evaluation)
               |--cloth    (25 images)  
               |--n95      (25 images)
               |--nomask   (25 images)
               |--surgical (25 images)
            --train(1750 images for train)
             |--cloth    (444 images)  
             |--n95      (437 images)
             |--nomask   (424 images)
             |--surgical (445 images)
            --sample(100 images for evaluation)
             |--cloth    (25 images)  
             |--n95      (25 images)
             |--nomask   (25 images)
             |--surgical (25 images)
            --train(1600 images for train)
             |--cloth    (400 images)  
             |--n95      (400 images)
             |--nomask   (400 images)
             |--surgical (400 images)

Phase I
**Source code:**

a)training
FILE:masks_detector_train.py 
Comments:Running this code to transform the images of the dataset in uniform, use the built CNN to train
         the model, and will generate the evaluation results used by the random split test set. And it will
         save the trained model in model.pkl.

b)application
FILE:masks_detector.py
Comments:Running this code to load the saved trained model in phase I to evaluate with the sample data(100 images).

**Trained model:**
FILE:model.pkl
Comments:Save the trained model in this file.


Phase II
**Source code:**

a)training
FILE:masks_detector_train_p2.py 
Comments:Running this code to transform the images of the dataset in uniform, use the improved CNN to train
         the model, and will generate the evaluation results used by the random split test set. And it will
         save the trained model in model2.pkl.

b)application
FILE:masks_detector_p2.py
Comments:Running this code to load the saved trained model in phase II to evaluate with the sample data(100 images).

**Trained model:**
FILE:model2.pkl
FILE：model2+.pkl
Comments:model2 is saved after improved CNN architecture.
         model2+ is saved after update the dataset to eliminate the bias.
**Project report:**
File:
</pre>



