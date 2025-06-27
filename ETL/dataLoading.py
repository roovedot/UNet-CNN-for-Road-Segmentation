import dataUtils # Utils made for this project
import UNet.ETL.dataConfig as dataConfig # Config File
import os

def organise_label_data():
    testLabels = dataConfig.CITYSCAPES_ROOT_DIR + "gtFine_trainvaltest/gtFine/test"
    trainLabels = dataConfig.CITYSCAPES_ROOT_DIR + "gtFine_trainvaltest/gtFine/train"
    valLabels = dataConfig.CITYSCAPES_ROOT_DIR + "gtFine_trainvaltest/gtFine/val"
    
    trainOut = dataConfig.DATA_ROOT_DIR + "/CS_trainLabels"
    valOut = dataConfig.DATA_ROOT_DIR + "/CS_valLabels"

    # Move train labels (we are joining train and test labels)
    print("ABOUT TO GET TEST LABELS:")
    dataUtils.move_files_filtered(
        src_dir=testLabels,
        out_dir=trainOut,
        filter=r'.*labelIds\.png$',
    )
    
    print("ABOUT TO GET TRAIN LABELS:")
    dataUtils.move_files_filtered(
        src_dir=trainLabels,
        out_dir=trainOut,
        filter=r'.*labelIds\.png$',    
    )

    # Move val labels
    print("ABOUT TO GET VAL LABELS:")
    dataUtils.move_files_filtered(
        src_dir=valLabels,
        out_dir=valOut,
        filter=r'.*labelIds\.png$',
    ) 

if __name__ == "__main__":
    # Get the labels we are going to use from cityscapes dataset
    
    organise_label_data()
