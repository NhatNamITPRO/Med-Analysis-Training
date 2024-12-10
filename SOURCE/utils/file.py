import os
import gzip

def get_all_path_by_ext(root, extension, match=""):
    """
    Retrieve all file paths in the root directory and subdirectories with a specific extension
    and optionally filter by filenames containing a specific string.
    
    Parameters:
        root (str): The root directory to search.
        extension (str): The file extension to search for (e.g., '.gz').
        match (str, optional): A string that the filename must contain. Defaults to "" (no filter).

    Returns:
        list: A list of file paths that match the given extension and match criteria.
    """
    # Ensure the extension has a dot at the start
    if not extension.startswith("."):
        extension = "." + extension
    
    file_paths = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(extension) and (match in filename):
                file_path = os.path.join(dirpath, filename).replace("\\","/")
                file_paths.append(file_path)
    
    return file_paths

def decompress_gz(file_path, output_dir):
    """
    Decompress a .gz file to the specified output directory.

    Parameters:
        file_path (str): The path to the .gz file.
        output_dir (str): The directory where the decompressed file should be saved.

    Returns:
        str: The path to the decompressed file.

    Raises:
        FileNotFoundError: If the .gz file does not exist.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, os.path.basename(file_path[:-3]))  

    try:
        with gzip.open(file_path, 'rb') as f_in:
            with open(output_file_path, 'wb') as f_out:
                f_out.write(f_in.read())
    except Exception as e:
        raise IOError(f"Error decompressing {file_path}: {e}")
    
    return output_file_path