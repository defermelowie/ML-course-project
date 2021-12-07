import csv
import random
import itertools

#####################
# Dataset constants #
#####################

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

##################
# Populate lists #
##################

# Define empty lists
papier_list = []
glas_list = []
pmd_list = []
restafval_list = []

# Populate lists
for i in range(1, CARDBOARD_SIZE+1):
    papier_list.append({
        'path': CARDBOARD_PATH + str(i) + '.jpg',
        'type': 'papier',
        'set': 'undefined'
    })

for i in range(1, PAPER_SIZE+1):
    papier_list.append({
        'path': PAPER_PATH + str(i) + '.jpg',
        'type': 'papier',
        'set': 'undefined'
    })

for i in range(1, GLASS_SIZE+1):
    glas_list.append({
        'path': GLASS_PATH + str(i) + '.jpg',
        'type': 'glas',
        'set': 'undefined'
    })

for i in range(1, PLASTIC_SIZE+1):
    pmd_list.append({
        'path': PLASTIC_PATH + str(i) + '.jpg',
        'type': 'pmd',
        'set': 'undefined'
    })

for i in range(1, METAL_SIZE+1):
    if (i in METAL_EXCLUDED):
        restafval_list.append({
            'path': METAL_PATH + str(i) + '.jpg',
            'type': 'restafval',
            'set': 'undefined'
        })
    else:
        pmd_list.append({
            'path': METAL_PATH + str(i) + '.jpg',
            'type': 'pmd',
            'set': 'undefined'
        })

for i in range(1, TRASH_SIZE+1):
    restafval_list.append({
        'path': TRASH_PATH + str(i) + '.jpg',
        'type': 'restafval',
        'set': 'undefined'
    })

##########################################
# Split data into sets (test & CV/train) #
##########################################
K = 5

# Generate set labels
labels = itertools.cycle(['test'] + [f'cv{nr}' for nr in range(0, K-1)])

# Shuffle lists to achieve random sets
random.shuffle(papier_list)
random.shuffle(glas_list)
random.shuffle(pmd_list)
random.shuffle(restafval_list)

# Assign labels to data
for item in papier_list:
    item['set'] = next(labels)

for item in glas_list:
    item['set'] = next(labels)

for item in pmd_list:
    item['set'] = next(labels)

for item in restafval_list:
    item['set'] = next(labels)

###############
# Save to csv #
###############

# Combine all lists
path_list = papier_list + glas_list + pmd_list + restafval_list

with open('./data/path_list.csv', mode='w', newline='') as fd:
    writer = csv.DictWriter(fd, dict.keys(path_list[0]))
    writer.writeheader()
    writer.writerows(path_list)

#    writer = csv.writer(list_file, delimiter=',',
#                        quotechar='"', quoting=csv.QUOTE_MINIMAL)
#    writer.writerows(path_list)
