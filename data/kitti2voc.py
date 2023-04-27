import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count
from pascal_voc_writer import Writer

KITTI_LABEL_NAMES = [
    'Car', 
    'Van', 
    'Truck',
    'Pedestrian', 
    'Person_sitting', 
    'Cyclist', 
    'Tram',
    'Misc',
    'DontCare']

def kitti2voc(kitti_dataset_dir,voc_dataset_dir,train_ratio=0.8,class_names=KITTI_LABEL_NAMES):

    """
        @NOTE: convert kitti dataset to voc dataset
        
        @param (str) kitti_dataset_dir: path to kitti format dataset
        @param (str) voc_dataset_dir: path to save voc format dataset
        @param (double) train_ratio: train to val ratio, default set to 0.8
        @param (list) class_names: class names to convert
        
        @return (None)
    """
  
    # kitti format dataset init
    kitti_image_dir = os.path.join(kitti_dataset_dir,'training','image_2')
    kitti_label_dir = os.path.join(kitti_dataset_dir,'training','label_2')
    print(kitti_image_dir,kitti_label_dir)

    # voc format dataset init
    voc_image_dir = os.path.join(voc_dataset_dir, 'JPEGImages')
    if not os.path.exists(voc_image_dir):
        os.makedirs(voc_image_dir)
    voc_annotation_dir = os.path.join(voc_dataset_dir, "Annotations")
    if not os.path.exists(voc_annotation_dir):
        os.makedirs(voc_annotation_dir)
    voc_imagesets_main_dir = os.path.join(voc_dataset_dir, "ImageSets", "Main")
    if not os.path.exists(voc_imagesets_main_dir):
        os.makedirs(voc_imagesets_main_dir)

    # init kitti image paths and kitti label paths
    kitti_image_paths = []
    kitti_label_paths = []
    voc_image_paths = []
    voc_annotation_paths = []
    for txt_name in os.listdir(kitti_label_dir):
        txt_label_path = os.path.join(kitti_label_dir,txt_name)
        #print(txt_label_path)
        #print(is_conitain_object(txt_label_path,class_names))
        if is_conitain_object(txt_label_path,class_names):
            name,ext = os.path.splitext(txt_name)
            kitti_image_paths.append(os.path.join(kitti_image_dir,name+".png"))
            kitti_label_paths.append(os.path.join(kitti_label_dir,txt_name))
            voc_image_paths.append(os.path.join(voc_image_dir,name+".jpg"))
            voc_annotation_paths.append(os.path.join(voc_annotation_dir,name+".xml"))
    kitti_image_paths = np.array(kitti_image_paths)
    kitti_label_paths = np.array(kitti_label_paths)
    voc_image_paths = np.array(voc_image_paths)
    voc_annotation_paths = np.array(voc_annotation_paths)
    print(len(kitti_image_paths),len(kitti_label_paths),len(voc_image_paths),len(voc_annotation_paths))

    # dataset shuffle
    size = len(kitti_label_paths)
    random_index = np.random.permutation(size)
    kitti_image_paths = kitti_image_paths[random_index]
    kitti_label_paths = kitti_label_paths[random_index]
    voc_image_paths = voc_image_paths[random_index]
    voc_annotation_paths = voc_annotation_paths[random_index]

    # split train and val dataset (10-fold cross-validation)
    train_size = int(size*train_ratio)  # train : val = 8 : 2
    train_kitti_image_paths = kitti_image_paths[0:train_size]
    val_kitti_image_paths = kitti_image_paths[train_size:]
    for choice,kitti_image_paths in [('train',train_kitti_image_paths),
                               ('val',val_kitti_image_paths),
                               ('trainval',kitti_image_paths)]:
        voc_txt_path = os.path.join(voc_imagesets_main_dir,choice+".txt")
        with open(voc_txt_path,'w') as f:
            for image_path in kitti_image_paths:
                dir,image_name = os.path.split(image_path)
                name,ext = os.path.splitext(image_name)
                f.write(name+"\n")

    # convert Kitti to VOC suing multi-processing
    batch_size = size // (cpu_count()-1)
    pool = Pool(processes=cpu_count()-1)
    for start in np.arange(0,size,batch_size):
        end = int(np.min([start+batch_size,size]))
        batch_kitti_image_paths = kitti_image_paths[start:end]
        batch_kitti_label_paths = kitti_label_paths[start:end]
        batch_voc_image_paths = voc_image_paths[start:end]
        batch_voc_annotation_paths = voc_annotation_paths[start:end]
        pool.apply_async(batch_image_label_process,error_callback=print_error,
                         args=(batch_kitti_image_paths,batch_kitti_label_paths,
                               batch_voc_image_paths,batch_voc_annotation_paths,class_names))
    pool.close()
    pool.join()

def is_conitain_object(kitti_txt_path,class_names):
    """
        @NOTE: check for existance of labels in calss_names

        @param (str) kitti_txt_path: kitti txt label path
        @param (list) class_names: class names to be converted

        @return (bool) True: contain, False: not contain
    """

    with open(kitti_txt_path,'r') as f:
        for line in f.readlines():
            strs = line.strip().split(" ")
            if strs[0] not in class_names:
                continue
            else:
                return True
    return False

def print_error(value):
    print("error: ", value)

def single_image_label_process(kitti_image_path,kitti_txt_path,voc_image_path,voc_annotation_path,class_names):
    """
        @NOTE: convert single image and label to VOC dataset format

        @param (str) kitti_image_path: path to kitti image
        @param (str) kitti_txt_path: path to kitti label
        @param (str) voc_image_path: path to voc image
        @param (str) voc_annotation_path: path to voc annotation
        @param (list) class_names: class names to be converted

        @return (None)
    """
    image = cv2.imread(kitti_image_path)
    cv2.imwrite(voc_image_path,image)
    h, w, c = np.shape(image)
    dir,image_name = os.path.split(kitti_image_path)
    writer = Writer(image_name,w,h)

    # txt to xml
    with open(kitti_txt_path,'r') as f:       
        for line in f.readlines():
            strs = line.strip().split(" ")
            if strs[0] not in class_names:
                continue
            else:
                xmin = max(int(float(strs[4]))-1,0)
                ymin = max(int(float(strs[5]))-1,0)
                xmax = min(int(float(strs[6]))+1,w)
                ymax = min(int(float(strs[7]))+1,h)
                label = strs[0]
                if label in ['Person_sitting',"Pedestrian"]:
                    label = 'pedestrian'
                elif label in ['Cyclist']:
                    label = 'cyclist'
                elif label in ['Car','Van','Truck','Tram']:
                    label = 'car'
                writer.addObject(label,xmin,ymin,xmax,ymax)
        writer.save(voc_annotation_path)

def batch_image_label_process(batch_kitti_image_paths,batch_kitti_txt_paths,
                              batch_voc_image_paths,batch_voc_annotation_paths,class_names):
    size = len(batch_kitti_image_paths)
    for i in tqdm(np.arange(size)):
        kitti_image_path = batch_kitti_image_paths[i]
        kitti_txt_path = batch_kitti_txt_paths[i]
        voc_image_path = batch_voc_image_paths[i]
        voc_annotation_path = batch_voc_annotation_paths[i]
        single_image_label_process(kitti_image_path,kitti_txt_path,
                                   voc_image_path,voc_annotation_path,class_names)

if __name__ == '__main__':
    print("======= Convert KITTI to VOC ========")
    kitti_dataset_dir = os.path.abspath("/data/ziruiw3/KITTIdevkit")
    voc_dataset_dir = os.path.abspath("/data/ziruiw3/KITTI2VOC")
    train_ratio = 0.8
    class_names=['Person_sitting',"Pedestrian",'Cyclist',"Truck","Car","Tram","Van"]
    kitti2voc(kitti_dataset_dir,voc_dataset_dir,train_ratio,class_names)
    print("======= Finished ======")