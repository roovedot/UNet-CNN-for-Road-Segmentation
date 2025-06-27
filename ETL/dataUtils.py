'''
Methods to help organise the data in cityscapes.
I'll try to make it as modular and universal as possible, but 
 you will have to adapt it to your needs.
'''

import os
import shutil
import re
import UNet.ETL.dataConfig as dataConfig

def move_file(file_path, out_dir):
            print(f"Moving file: {file_path} to {out_dir}")
            try:
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
                    print(f"Created output directory {out_dir}.")
                shutil.move(file_path, out_dir)
            except Exception as e:
                print(f"Error moving file {file_path}: {e}")

def move_files_filtered(src_dir, out_dir, filter = None, dont_ask = False):
    r"""
    Moves files from src_dir to out_dir that match the filter "string" 
    
    :param filter: Regular Expression to filter files by. eg. <<< ".*gtFine_labelIds\.png$" (full name of file ending in gtFine_labelIds.png). >>> Recommend testing in https://regex101.com/ before using.
    :param src_dir: Directory to recursively search for files
    :param out_dir: Directory to move files to
    :param dont_ask: If True, will not ask for confirmation before moving files

    :return: None
    """

    cancelled = False

    ### FILTER CONFIG ###

    use_regex = True
    no_filter = False

    print("Filter Config:")
    if filter is None:
        no_filter = True
        print("No filter provided. Moving all files from source directory to output directory.")
    else:
        try:
            # Compile the regular expression filter previously to prevent compiling for each file.
            regFilter = re.compile(filter)  
            print(f"Using regex filter: {filter}")
        except re.error as e:
            print("Error in filter: ", e)
            print("Resorting to string-matching filter")
            use_regex = False
    print("----------------------------------------------------------------------------------")   

    ### PATH CONFIG ###

    print("Path Config:")
    # Check directories
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source directory {src_dir} does not exist.")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        print(f"Created output directory {out_dir}.")
    else:
        print(f"Output directory {out_dir} already exists. Files will be moved into it.")
    
    # Ask for confirmation before proceeding 
    if not dont_ask:
        userResponse = input("Do you want to proceed? (y/n): ").strip().lower()
        if userResponse not in ("y", "yes", "Y"):
            print("Operation cancelled by user.")
            cancelled = True


    ### MOVE FILES ###

    if not cancelled:

        print(f"Moving files from {src_dir} to {out_dir}. If you don't see any 'Moving file' messages, means no files in {src_dir} matched the filter.")
        print("--------------------------------------------------------------------------------")

        # Crawl src_dir recursively
        for dirpath, dirnames, fileNames in os.walk(src_dir):

            # No filter -> move all files
            if no_filter: 
                for fileName in fileNames:
                    file_path = os.path.join(dirpath, fileName)
                    move_file(file_path, out_dir)

            # Filter
            else: 
                # Use regex filter
                if use_regex: 
                    for fileName in fileNames:
                        if regFilter.search(fileName):
                            file_path = os.path.join(dirpath, fileName)
                            move_file(file_path, out_dir)

                # Use string-matching filter
                else: 
                    for fileName in fileNames: # For each file in src_dir (recursively)
                        if filter in fileName: 
                            file_path = os.path.join(dirpath, fileName)
                            move_file(file_path, out_dir)

        print("--------------------------------------------------------------------------------")
        print(f"Finished moving files from {src_dir} to {out_dir}.")

def test_move_files_filtered():
    """
    Test function for move_files_filtered.
    """
    src_dir = "/home/roovedot/sucio/dir1"
    out_dir = "/home/roovedot/sucio/dir2"
    filter = r'\.txt'  # Example filter to move all PNG files

    move_files_filtered(src_dir, out_dir, filter, dont_ask=True)

def purge_images(
    labels_dir = dataConfig.DATA_ROOT_DIR + "CS_trainLabels",
    images_dir = dataConfig.DATA_ROOT_DIR + "images/train_extended",
    out_dir = dataConfig.DATA_ROOT_DIR + "images/train"
    ):
    """
    Selects only images that have a corresponding label file
    """
    
    # Match label files and save the fileId eg. (aachen_000001_000019)
    filter = re.compile(r'^(?P<fileId>.*)_gtFine_labelIds.png$')

    # Crawl the labels directory
    for dirpath, dirnames, fileNames in os.walk(labels_dir):
        for fileName in fileNames:
            m = filter.match(fileName)
            if m: # Make sure the fileName matches the filter 

                # save fileId
                fileId = m.group('fileId') 
                print(f"Processing fileId: {fileId}") #Debug

                # Build corresponding image file path
                imgPath = os.path.join(images_dir, fileId + "_leftImg8bit.png")
                if os.path.exists(imgPath):
                    print(f"Moving image: {fileId} to {out_dir}") #Debug
                    move_file(imgPath, out_dir)
                else:
                    print(f"Image file {imgPath} does not exist. Skipping.")
            else:
                print(f"File {fileName} does not match the filter. Skipping.")



# Main (just for testing and debugging purposes)
if __name__ == "__main__":
    # Uncomment to test the function
    # test_move_files_filtered()

    r'''Reuse images from another proyect
    move_files_filtered(
        src_dir="/media/roovedot/PHILIPS/yolo-cityscapes/dataYolo11Structure/train/images",
        out_dir="/media/roovedot/PHILIPS/cityscapesRootFolder/images/train",
        filter=r'.*\.png$')
    '''

    # Figuring out how regex groups work
    '''
    stringPrueba = "aachen_000001_000019_gtFine_labelIds.png"

    filtro= r'^(?P<fileId>.*)_gtFine_labelIds.png$'
    compilado = re.compile(filtro)
    m = compilado.match(stringPrueba)
    print(m.groupdict(0))
    print("groups: "+m.group('fileId'))
    '''

    # Uncomment to purge images
    # purge_images()