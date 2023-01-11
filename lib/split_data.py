import os
import random


'''random.seed(41)

list_val = sorted(random.sample(range(7500), 3000))

label_data = open('/media/sean/Sean/PIPNet/data/WFLW_dvs_v3_SS/train_label_r.txt','w+')
unlabel_data = open('/media/sean/Sean/PIPNet/data/WFLW_dvs_v3_SS/train_unlabel_r.txt','w+')

label_path = os.path.join('/media/sean/Sean/PIPNet/data', 'WFLW_dvs_v3_SS', 'train.txt')
with open(label_path, 'r') as f:
    labels = f.readlines()
labels = [x.strip() for x in labels]

labels_new = []
i = 0
for label in labels:
    image_name = label + '\n'

    if i in list_val:
        print('Save ', image_name, 'to label file')
        label_data.write(image_name)
    else:
        print('Save ', image_name, 'to unlabel file')
        unlabel_data.write(image_name)
    i += 1

label_data.close()
unlabel_data.close()'''

###################
with open('/media/sean/Sean/PIPNet/data/WFLW_dvs_v3_SS/train_label_r.txt', 'r') as f:
    labels = f.readlines()
labels = [x.strip() for x in labels]
print('label numbers : ', len(labels))

with open('/media/sean/Sean/PIPNet/data/WFLW_dvs_v3_SS/train_unlabel_r.txt', 'r') as f:
    labels = f.readlines()
labels = [x.strip() for x in labels]
print('unlabel numbers : ', len(labels))
