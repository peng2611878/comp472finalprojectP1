**Project Title:**<br />
face-mask-detector<br />
**Datasets:**<br />
img folder --sample(100 images for evaluation)<br />
           |     --cloth    (25 images)  <br />
           |     --n95      (25 images)<br />
           |     --nomask   (25 images)<br />
           |     --surgical (25 images)<br />
            --train(1600 images for train)<br />
           |     --cloth    (400 images)  <br />
           |     --n95      (400 images)<br />
           |     --nomask   (400 images)<br />
           |     --surgical (400 images)<br />
**Source code:**<br />
a)training<br />
masks_detector_train.py <br />
Running this code to transform the images of the dataset in uniform, use the built CNN to train<br />
the model, and will generate the evaluation results used by the random split test set. And it will<br />
save the trained model in model.pkl.<br />
b)application<br />
masks_detector.py<br />
Running this code to load the saved trained model to evaluate with the sample data(100 images).<br />
**Trained model:**<br />
model.pkl<br />
**Project report:**<br />



