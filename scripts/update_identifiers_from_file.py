import logging
from sys import argv

from autofit.tools.update_identifiers import update_identifiers_from_file

if __name__ == "__main__":
    try:
        directory, map_file = argv[1:]
        update_identifiers_from_file(
            output_directory=directory,
            map_filename=map_file
        )
    except Exception as e:
        logging.debug(e)
        print(
            "Usage: update_identifiers_from_file.py directory map_file.yaml"
        )
