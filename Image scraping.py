import urllib.request
import cv2
import numpy as np
import os

def resize_image():
    img = cv2.imread(r"C:\Users\itzme\Desktop\Hemant\Ambulance.jpg")
    resized_image = cv2.resize(img, (30, 60))
    cv2.imwrite(r"C:\Users\itzme\Desktop\Hemant\Ambulance.jpg",resized_image)
def store_raw_images():
    neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02701002'   
    
    #neg_images_link = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02701002'   
    neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()
    pic_num = 1
    
    if not os.path.exists('pos'):
        os.makedirs('pos')
        
    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i, "pos/"+str(pic_num)+".jpg")
            img = cv2.imread("pos/"+str(pic_num)+".jpg",cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite("pos/"+str(pic_num)+".jpg",resized_image)
            pic_num += 1
            
        except Exception as e:
            print(str(e))

def find_uglies():
    match = False
    for file_type in ['pos']:
        for img in os.listdir(file_type):
            for ugly in os.listdir('uglies'):
                try:
                    current_image_path = str(file_type)+'/'+str(img)
                    ugly = cv2.imread('uglies/'+str(ugly))
                    question = cv2.imread(current_image_path)
                    if ugly.shape == question.shape and not(np.bitwise_xor(ugly,question).any()):
                        print('That is one ugly pic! Deleting!')
                        print(current_image_path)
                        os.remove(current_image_path)
                except Exception as e:
                    print(str(e))
def create_pos_n_neg():
    for file_type in ['pos']:        
        for img in os.listdir(file_type):
            if file_type == 'pos':
                line = file_type+'/'+img+' 1 0 0 50 50\n'
                with open('info.dat','a') as f:
                    f.write(line)
            elif file_type == 'pos':
                line = file_type+'/'+img+'\n'
                with open('bg.txt','a') as f:
                    f.write(line)
                    
if __name__=='__main__':
    resize_image()
    store_raw_images()
    find_uglies()
    create_pos_n_neg()
