#!/usr/bin/python3


import jetson_inference
import jetson_utils
import argparse



parser = argparse.ArgumentParser()

parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("out_filename", type=str, help="filename of intended output" )


# normal vs abnormal model

parser.add_argument("--model1", type=str,
    default="/home/ethan/jetson-inference/python/training/classification/models/knee-classification/resnet152.onnx", help="path to retrained ONNX model")

parser.add_argument("--labels1", type=str,
    default="/home/ethan/jetson-inference/python/training/classification/models/knee-classification/labels.txt", help="path to labels.txt file")


# osteopenia vs osteoporosis model 

parser.add_argument("--model2", type=str,
    default="/home/ethan/jetson-inference/python/training/classification/models/knee-classification-3/densenet121.onnx", help="path to retrained ONNX model")

parser.add_argument("--labels2", type=str,
    default="/home/ethan/jetson-inference/python/training/classification/models/knee-classification-abnormal/labels.txt", help="path to labels.txt file")



parser.add_argument("--input_blob", type=str, default="input_0")
parser.add_argument("--output_blob", type=str, default="output_0")

opt = parser.parse_args()


img = jetson_utils.loadImage(opt.filename)
net1 = jetson_inference.imageNet(model=opt.model1,
    labels=opt.labels1,
    input_blob=opt.input_blob,
    output_blob=opt.output_blob)
class_idx1, confidence1 = net1.Classify(img)
class_desc1 = net1.GetClassDesc(class_idx1)    
print("Stage-1 model:", opt.model1)
print("Stage-1 labels file:", opt.labels1)
print("Stage-1 classes:")
for i in range(net1.GetNumClasses()):
      print(f"  {i}: {net1.GetClassDesc(i)}")


    
# logic to split between the normal vs abnormal models so when the stage 1 model doesn't detect normal, the stage 2 model will activate

stage1_label = net1.GetClassDesc(class_idx1).strip().lower()
print(f"[DEBUG] Stage‑1 prediction: '{stage1_label}' @ {confidence1:.2%}")

NORMAL_INDEX = 0  

if class_idx1 == NORMAL_INDEX and confidence1 >= 0.60:
    final_class = class_desc1
    final_confidence = confidence1
    final_idx = class_idx1
else:
    net2 = jetson_inference.imageNet(model=opt.model2,
        labels=opt.labels2,
        input_blob=opt.input_blob,
        output_blob=opt.output_blob)
    

    class_idx2, confidence2 = net2.Classify(img)
    class_desc2 = net2.GetClassDesc(class_idx2)

    print("Stage-2 model:", opt.model2)
    print("Stage-2 labels file:", opt.labels2)
    print("Stage‑2 classes:")
    for i in range(net2.GetNumClasses()):
        print(f"{i}: {net2.GetClassDesc(i)}")

    final_idx = class_idx2
    final_class = str(class_desc2)
    final_confidence = confidence2

    print(f"[DEBUG] Index 0 label: {net2.GetClassDesc(0)}")




image_height = img.height

font_size = int(image_height*0.1)
font = jetson_utils.cudaFont(size=font_size)

font.OverlayText(
    img,
    text=f"{final_class} {final_idx} {final_confidence*100:.1f}%",
    x=0,
    y=5,
    color=(255, 255, 0, 255),
    background=(0, 0, 0, 160,)
)
jetson_utils.saveImage(opt.out_filename, img)
print(f"Successfully classified image to {opt.out_filename}")
print(f"image is recognized as {final_class} class # {final_idx} with {final_confidence*100}% confidence")
