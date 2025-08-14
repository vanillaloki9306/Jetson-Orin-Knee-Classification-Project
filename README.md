# Jetson-Orin-Knee-Classification-Project
Learning AI training with knee classification utilizing x-ray images.

![osteoporosis-test9](https://github.com/user-attachments/assets/c24888f9-10aa-41fe-9af0-b793e2ea6cf3)
Image of osteoporosis x-ray scan  labelled by AI
![osteopenia-test10](https://github.com/user-attachments/assets/4f5eb501-3b46-4982-801f-09f3ae51d5d7)
Image of osteopenia x-ray scan labelled by AI
![normal-test20](https://github.com/user-attachments/assets/b4792bbb-dfd4-4527-af77-033d84460cb7)
Image of a normal knee x-ray scan labelled by AI


## The Algorithm

The code follows a two-stage logic, where the first stage differentiates between normal/abnormal knee scans, and the second stage decides whether it is osteopenia or osteoporosis. The first stage has been trained on the resnet152 network and the second stage has been trained on the densenet121 network. Both models were trained using all 3 classes, to prepare for a situation of error, and also ensure no training bias. When the user submits an input image, the code would first use the resnet152 model to confirm whether or not it is normal or abnormal (it will predict it's actual condition but it wouldn't have the deciding factor). If the confidence that the photo is normal exceeds 60%, the code stops and confirms that the photo is of a normal knee. However, if the model's confidence is less than 60%, the code calls for the second model to detect what type of knee condition it may be. In the situation of the first model having an error, the second model is still trained with normal knee data as a second layer of defense. 
## Running this project

1. Find or create a dataset with classes and labels. Make sure it is sorted by test/train/val. In my case, I used a dataset organized by Fuyad Hasan Bhoyan on kaggle https://www.kaggle.com/datasets/fuyadhasanbhoyan/knee-osteoarthritis-classification-224224.
2. Unzip the dataset into jetson-inference. It should be found in jetson-inference/python/training/classification/data
3. Open the docker from jetson-inference and navigate back into the python/training/classification directory. There, begin training. In my case, I used the command 
"python3 train.py \
  --model-dir=models/knee-classification-3 \
  --arch=densenet121 \
  --epochs=90 \
  --batch-size=8 \
  --workers=0 \
  --pretrained data/knee-classification-3"
4. Repeat this twice, with any model deemed reasonable. I used resnet152 for my stage 1 model and densenet121 for my stage 2 model.
5. Export the models and include their paths as an argument in a python file.
6. Test the models with a few photos.
8. Make sure to include pytorch.

