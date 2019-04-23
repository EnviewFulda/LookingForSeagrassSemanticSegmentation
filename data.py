# dataset provider for classifications

import os 
import cv2
import numpy as np
from PIL import Image
import sys 
import json
from time import sleep
from IPython.display import display 


class Data:
    
    # loads the all file names or the serialized numpy object
    def loadDataset(self):
        self.pathImages = {
            "train": self.config["path"],
            "trainLabel": self.config["path"],
            "test" : self.config["path"],
            "testLabel": self.config["path"],
            "validation": self.config["path"],
            "validationLabel": self.config["path"]
        }

        if self.config["preProcessedPath"] != "":
            self.pathImages["train"] += self.config["preProcessedPath"]+self.config["images"]
            self.pathImages["trainLabel"] += self.config["preProcessedPath"]+self.config["labels"]
        else:
            self.pathImages["train"] += self.config["images"]
            self.pathImages["trainLabel"] += self.config["labels"]
            self.pathImages["test"] += self.config["images"]
            self.pathImages["testLabel"] += self.config["labels"]
            self.pathImages["validation"] += self.config["images"]
            self.pathImages["validationLabel"] += self.config["labels"]

        if not self.config["serializedObject"]:
            
            if self.config["name"] == "Seagrass":       
                jsonData = json.load(open(self.config["path"]+"train.json"))
                trainData = list(map(lambda i:os.path.basename(i["image"]) if i["depth"] <= float(self.config["depth"]) else None, jsonData))
                labelData = list(map(lambda i:os.path.basename(i["ground-truth"]) if i["depth"] <= float(self.config["depth"]) else None, jsonData))

                jsonData = json.load(open(self.config["path"]+"test.json"))
                testData = list(map(lambda i:os.path.basename(i["image"]) if i["depth"] <= float(self.config["depth"]) else None, jsonData))
                testLabelData = list(map(lambda i:os.path.basename(i["ground-truth"]) if i["depth"] <= float(self.config["depth"]) else None, jsonData))

                jsonData = json.load(open(self.config["path"]+"validate.json"))
                validateData = list(map(lambda i:os.path.basename(i["image"]) if i["depth"] <= float(self.config["depth"]) else None, jsonData))
                validateLabelData = list(map(lambda i:os.path.basename(i["ground-truth"]) if i["depth"] <= float(self.config["depth"]) else None, jsonData))
                
                self.imageData = {
                    "train": list(filter(lambda i:i != None, trainData)),
                    "trainLabel": list(filter(lambda i:i != None, labelData)),
                    "test": list(filter(lambda i:i != None, testData)),
                    "testLabel": list(filter(lambda i:i != None, testLabelData)),
                    "validation": list(filter(lambda i:i != None, validateData)),
                    "validationLabel": list(filter(lambda i:i != None, validateLabelData))
                }
                
                    
                
            else:
                # sort data because os.listdir selects files in arbitrary order
                trainDataFiles = os.listdir(self.pathImages["train"])
                trainLabelDataFiles = os.listdir(self.pathImages["trainLabel"])
                trainDataFiles.sort()
                trainLabelDataFiles.sort()


                trainElements = int(self.config["trainSize"]*self.config["size"])
                testElements = int(self.config["testSize"]*self.config["size"])

                self.imageData = {
                    "train": trainDataFiles[:trainElements],
                    "trainLabel": trainLabelDataFiles[:trainElements],
                    "test": trainDataFiles[trainElements:trainElements+testElements],
                    "testLabel": trainLabelDataFiles[trainElements:trainElements+testElements],
                    "validation": trainDataFiles[trainElements+testElements if testElements > 0 else trainElements:],
                    "validationLabel": trainLabelDataFiles[trainElements+testElements if testElements > 0 else trainElements:],
                }

                
            self.config["trainSize"] = len(self.imageData["train"])
            self.config["testSize"] = len(self.imageData["test"])
            self.config["validationSize"] = len(self.imageData["validation"])

            print("trainSize: ", self.config["trainSize"], " Testsize: ", self.config["testSize"], "Validationsize: ", self.config["validationSize"])
            
        else:

            self.imageData = np.load(self.config["path"]+self.config["fileName"]+".npy")
            self.config["trainSize"] = len(self.imageData.item().get("train"))
            self.config["testSize"] = len(self.imageData.item().get("test"))
            self.config["validationSize"] = len(self.imageData.item().get("validation"))
            print("Finished loading dataset...")


    # string configPath - path of the json file which describes the dataset
    def __init__(self, configPath):

        try:
            self.config = json.load(open(configPath))    
        except:
            raise "Wrong path for data config file given!"

    
        self.loadDataset()

    # gets a value from the config file with its given name
    def getConfig(self, name):
        return self.config[name]

    
    def getImageTuple(self, imageFilename, labelFilename):
        
        img = cv2.imread(self.pathImages["train"]+imageFilename.decode())
        labelImg = cv2.imread(self.pathImages["trainLabel"]+labelFilename.decode())
        if self.config["downsize"]:
            img = cv2.resize(img, (self.config["x"], self.config["y"]), interpolation=cv2.INTER_AREA)
            labelImg = cv2.resize(labelImg, (self.config["x"], self.config["y"]), interpolation=cv2.INTER_AREA)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        labelImg = cv2.cvtColor(labelImg, cv2.COLOR_BGR2RGB)

        
        
        # exterminate conversion errors by opencv
        labelImg[(labelImg <= 127).all(-1)] = [0,0,0]
        labelImg[(labelImg >= 128).all(-1)] = [255,255,255]
       
        for rgbIdx, rgbV in enumerate(self.config["ClassToRGB"]):
            labelImg[(labelImg == rgbV).all(-1)] = rgbIdx

            
        labelImg = labelImg[:,:,0]
        img = ((img - img.mean()) / img.std()).astype(np.float32)

        return img, labelImg