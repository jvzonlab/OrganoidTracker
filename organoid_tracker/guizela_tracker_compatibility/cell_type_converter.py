# Map of file_name.p to CELL_TYPE
FILE_TO_CELL_TYPE = {
    "enterocyte.p": "ENTEROCYTE",
    "enteroendocrine.p": "ENTEROENDOCRINE",
    "goblet.p": "GOBLET",
    "paneth.p": "PANETH",
    "stemcell.p": "STEM",
    "tuft.p": "TUFT",
    "young.p": "WGA_PLUS"
}

# Map of CELL_TYPE to file_name.p
CELL_TYPE_TO_FILE = dict(((cell_type, file_name) for file_name, cell_type in FILE_TO_CELL_TYPE.items()))

