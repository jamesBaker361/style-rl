import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from datasets import load_dataset,Dataset
from sklearn.preprocessing import StandardScaler
import joblib  # for saving

parser=argparse.ArgumentParser()

parser.add_argument("--project_name",type=str,default="evaluation-creative")
parser.add_argument("--src_dataset",type=str,default="jlbaker361/league-tagged-clip")
parser.add_argument("--dest_dataset",type=str,default="jlbaker361/league-clip-classification")



def main(args):
    accelerator=Accelerator(log_with="wandb")
    
    data=load_dataset(args.src_dataset,split="train")
    all_tags=set(data["tag"])
    all_champions=set(data["champion"])

    plane_dict={
        "weight_SVC":[],
        "bias_SVC":[],
        "weight_SGD":[],
        "bias_SGD":[],
        "type":[], #tag or character
        "label":[], #the tag or character in question
        "positives":[] #how many positive examples there were
    }
    X=[row["embedding"][0] for row in data]

    scaler =StandardScaler()
    X=scaler.fit_transform(X)
    model_dict={
            "SVC":LinearSVC,
            "SGD":SGDClassifier
        }
    for tag in all_tags:

        
        y=[1 if row["tag"]==tag else 0 for row in data]

        classes = np.unique(y)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        class_weight = {c: w for c, w in zip(classes, weights)}

        
        for key,model in model_dict.items():
            # Train linear SVM
            clf = model(class_weight=class_weight, max_iter=20000,verbose=1)
            clf.fit(X, y)

            w = clf.coef_[0]   # normal vector to hyperplane
            b = clf.intercept_[0]   # bias term

            plane_dict[f"weight_{key}"].append(w)
            plane_dict[f"bias_{key}"].append(b)
        plane_dict["type"].append("tag")
        plane_dict["label"].append(tag)
        plane_dict["positives"].append(sum(y))

    for champion in all_champions:

        
        y=[1 if row["champion"]==champion else 0 for row in data]

        classes = np.unique(y)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        class_weight = {c: w for c, w in zip(classes, weights)}

        for key,model in model_dict.items():
            # Train linear SVM
            clf = model(class_weight=class_weight, max_iter=20000,verbose=1)
            clf.fit(X, y)

            w = clf.coef_[0]   # normal vector to hyperplane
            b = clf.intercept_[0]   # bias term

            plane_dict[f"weight_{key}"].append(w)
            plane_dict[f"bias_{key}"].append(b)
        plane_dict["type"].append("champion")
        plane_dict["label"].append(champion)
        plane_dict["positives"].append(sum(y))
    '''print("Hyperplane: w·x + b = 0")
    print("w =", w)
    print("b =", b)'''
    Dataset.from_dict(plane_dict).push_to_hub(args.dest_dataset)


if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")