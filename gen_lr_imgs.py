import cv2
from glob import glob
from tqdm import tqdm
import random
from threadpool import ThreadPool, makeRequests

all_list = glob('data/ffhq-512/*') + glob('data/celebahq-512/*')
target_size = 128
# there are 3 quality ranges for each img
quality_ranges = [(15, 75)]
output_path = 'data/lr-128'


def saving(path):
    assert '.jpg' in path
    img = cv2.imread(path)
    img = cv2.resize(img, (target_size, target_size))
    for qr in quality_ranges:
        quality = int(random.random() * (qr[1] - qr[0]) + qr[0])
        cv2.imwrite(output_path + '/' + path.split('/')[-1], img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), quality]) #.replace('.jpg', '_q%d.jpg' % quality)


with tqdm(total=len(all_list), desc='Resizing images') as pbar:
    def callback(req, x):
        pbar.update()

    t_pool = ThreadPool(12)
    requests = makeRequests(saving, all_list, callback=callback)
    for req in requests:
        t_pool.putRequest(req)
    t_pool.wait()
