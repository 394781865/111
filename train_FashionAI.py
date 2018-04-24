import os
import model as modellib
import FashionAI


DATA_DIR = "./data"

annotations_path = {
    "val_train": os.path.join(DATA_DIR, "train/Annotations/val_train.csv"),
    "train": os.path.join(DATA_DIR, "train/Annotations/train.csv"),
    "val": os.path.join(DATA_DIR, "train/Annotations/val.csv"),
}

FINE_TUNE = False

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

config = FashionAI.FashionConfig()
config.display()

# Training dataset
# load person keypoints dataset
train_csv_path = annotations_path["val_train"]
train_dataset_keypoints = FashionAI.FashionDataset(train_csv_path, DATA_DIR+"/train/")
train_dataset_keypoints.load_fashions()
train_dataset_keypoints.prepare()


#Validation dataset
val_csv_path = annotations_path["val"]
val_dataset_keypoints = FashionAI.FashionDataset(val_csv_path, DATA_DIR+"/val/")
val_dataset_keypoints.load_fashions()
val_dataset_keypoints.prepare()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)

if FINE_TUNE:
    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn__0004.h5")
    # # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH)
else:
    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                                               "mrcnn_bbox", "mrcnn_mask"])

print("Loading weights from ", COCO_MODEL_PATH)

if not FINE_TUNE:
    model.train(train_dataset_keypoints, val_dataset_keypoints,
                learning_rate=config.LEARNING_RATE,
                epochs=5,
                layers='all')

model.train(train_dataset_keypoints, val_dataset_keypoints,
            learning_rate=config.LEARNING_RATE/10,
            epochs=5,
            layers='all')

model.train(train_dataset_keypoints, val_dataset_keypoints,
            learning_rate=config.LEARNING_RATE / 100,
            epochs=100,
            layers='all')
