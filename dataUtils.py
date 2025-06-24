'''
Methods to help organise the data in cityscapes.
I'll try to make it as modular and universal as possible, but 
 you will have to adapt it to your needs.
'''

import os
import shutil
import re

def move_files_filtered(src_dir, out_dir, filter = None):
    r"""
    Moves files from src_dir to out_dir that match the filter "string" 
    
    :param filter: Regular Expression to filter files by. eg. <<< ".*gtFine_labelIds\.png$" (full name of file ending in gtFine_labelIds.png). >>> Recommend testing in https://regex101.com/ before using.
    :param src_dir: Directory to recursively search for files
    :param out_dir: Directory to move files to

    :return: None
    """

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
    else:
        print(f"Output directory {out_dir} already exists. Files will be moved into it.")
    print(f"Moving files from {src_dir} to {out_dir}. If you don't see any 'Moving file' messages, means no files in {src_dir} matched the filter.")
    print("--------------------------------------------------------------------------------")

    ### MOVE FILES ###

    # Sub-function to move a single file
    def move_file(file_path, out_dir):
        print(f"Moving file: {file_path} to {out_dir}")
        try:
            shutil.move(file_path, out_dir)
        except Exception as e:
            print(f"Error moving file {file_path}: {e}")

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
    src_dir = "/home/roovedot/sucio/dir2"
    out_dir = "/home/roovedot/sucio/dir1"
    filter = "\.txt"  # Example filter to move all PNG files

    move_files_filtered(src_dir, out_dir, filter)

if __name__ == "__main__":
    # Uncomment to test the function
    # test_move_files_filtered()

    pass