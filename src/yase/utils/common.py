import os


def create_folder_if_not_exist(path_folder: str, mode: int = 0o777) -> None:
    """Creates a folder if it does not already exist.

    This function checks if a folder at the specified path exists, and if not, 
    it creates the folder with the specified permissions (mode).

    Args:
        path_folder (str): The path where the folder should be created.
        mode (int, optional): The permissions mode for the new folder. Defaults to 0o777.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified path is invalid or inaccessible.
        PermissionError: If the user does not have sufficient permissions to create the folder.
        Exception: For other unexpected errors that may occur.
    """
    try:
        if not os.path.exists(path_folder):
            os.makedirs(name=path_folder, mode=mode)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"The specified path is invalid or inaccessible: {path_folder}") from exc
    except PermissionError as exc:
        raise PermissionError(f"Insufficient permissions to create folder at: {path_folder}") from exc
    except Exception as exc:
        raise Exception(f"An unexpected error occurred: {exc}") from exc
