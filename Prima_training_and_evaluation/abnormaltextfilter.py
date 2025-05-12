import csv
from tqdm import tqdm

# This function finds abnormal summarized reports from a csv of summarized reports
def getabnormallist(fname):

    reader = csv.reader(open(fname))

    words = ['subdural','subgaleal','subarachnoid','hematoma','contusion',
        'tumor','neoplasm','metastasis','oma','vasogenic',
        'itis','abscess','osis',
        'stroke','hemorrhage','aneurysm','infarct','arteriovenous','avm','cavernoma','fistula',
        'dementia','parkin','hunting','alzheim',
        'malformation','agen','dysgen','dysmorph','chiari','ectopia','hydroceph','ventriculomeg',
        'cyst','cavity','postsurg','edema','catheter','cranio','shunt','biopsy']

    excludew = ['labral','biceps','acromion','menis','humer','tendon']

    excludes = []
    normals = []
    abnormals = []
    wordhit = {}
    for word in words+excludew:
        wordhit[word] = 0


    for row in tqdm(reader):
        normal = True
        exclude = False
        for word in words:
            if word in row[1].lower():
                normal = False
                wordhit[word] += 1
        for word in excludew:
            if word in row[1].lower():
                exclude = True
                wordhit[word] += 1
        if exclude:
            excludes.append(row[0])
        elif normal:
            normals.append(row[0])
        else:
            abnormals.append(row[0])
        

    return abnormals

