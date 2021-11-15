import csv

# Dataset constants
CARDBOARD_SIZE = 403
GLASS_SIZE = 501
METAL_SIZE = 410
PAPER_SIZE = 594
PLASTIC_SIZE = 482
TRASH_SIZE = 137

CARDBOARD_PATH = './data/garbage_classification/cardboard/cardboard'
GLASS_PATH = './data/garbage_classification/glass/glass'
METAL_PATH = './data/garbage_classification/metal/metal'
PAPER_PATH = './data/garbage_classification/paper/paper'
PLASTIC_PATH = './data/garbage_classification/plastic/plastic'
TRASH_PATH = './data/garbage_classification/trash/trash'

METAL_EXCLUDED = [5, 11, 15, 16, 25, 40, 42, 44, 46, 50, 52, 57, 61, 72, 74, 77, 94, 114, 115, 118, 120, 122, 141, 144, 161, 164, 165, 166, 169, 170, 171, 172,
                  173, 175, 176, 177, 196, 220, 221, 224, 232, 250, 251, 257, 258, 266, 270, 290, 309, 312, 315, 319, 328, 342, 345, 352, 353, 356, 357, 374, 380, 388, 390, 406]

# Empty path list (only titles)
path_list = [('path', 'type', 'set')]

# Generate list
for i in range(1, CARDBOARD_SIZE+1):
    path_list.append((CARDBOARD_PATH + str(i) +
                     '.jpg', 'papier', 'undefined'))

for i in range(1, PAPER_SIZE+1):
    path_list.append((PAPER_PATH + str(i) +
                     '.jpg', 'papier', 'undefined'))

for i in range(1, GLASS_SIZE+1):
    path_list.append((GLASS_PATH + str(i) +
                     '.jpg', 'glas', 'undefined'))

for i in range(1, PLASTIC_SIZE+1):
    path_list.append((PLASTIC_PATH + str(i) +
                     '.jpg', 'pmd', 'undefined'))

for i in range(1, METAL_SIZE+1):
    type = 'pmd'
    if (i in METAL_EXCLUDED):
        type = 'restafval'
    path_list.append((METAL_PATH + str(i) +
                     '.jpg', type, 'undefined'))

for i in range(1, TRASH_SIZE+1):
    path_list.append((TRASH_PATH + str(i) +
                     '.jpg', 'restafval', 'undefined'))

# Save to csv
with open('./data/path_list.csv', mode='w', newline='') as list_file:
    writer = csv.writer(list_file, delimiter=',',
                        quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerows(path_list)
