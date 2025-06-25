import dataUtils # Utils made for this project
import config # Config File
import os

def organise_label_data():
    testLabels = config.CITYSCAPES_ROOT_DIR + "gtFine_trainvaltest/gtFine/test"
    trainLabels = config.CITYSCAPES_ROOT_DIR + "gtFine_trainvaltest/gtFine/train"
    valLabels = config.CITYSCAPES_ROOT_DIR + "gtFine_trainvaltest/gtFine/val"
    
    trainOut = config.DATA_ROOT_DIR + "/CS_trainLabels"
    valOut = config.DATA_ROOT_DIR + "/CS_valLabels"

    # Move train labels (we are joining train and test labels)
    print("ABOUT TO GET TEST LABELS:")
    dataUtils.move_files_filtered(
        src_dir=testLabels,
        out_dir=trainOut,
        filter=r'.*instanceIds\.png$',
    )
    
    print("ABOUT TO GET TRAIN LABELS:")
    dataUtils.move_files_filtered(
        src_dir=trainLabels,
        out_dir=trainOut,
        filter=r'.*instanceIds\.png$',    
    )

    # Move val labels
    print("ABOUT TO GET VAL LABELS:")
    dataUtils.move_files_filtered(
        src_dir=valLabels,
        out_dir=valOut,
        filter=r'.*instanceIds\.png$',
    ) 

#TODO: There are 2 train labels with no corresponding image

if __name__ == "__main__":
    # Get the labels we are going to use from cityscapes dataset
    
    organise_label_data()
