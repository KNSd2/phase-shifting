import os
import re
import glob
import cv2
import numpy as np
import plotly.offline as po
import plotly.graph_objs as go

class PhaseShifting:
    def __init__(self, width, height, step, gamma, output_dir, black_thr, white_thr, filter_size, input_prefix, config_file):
        self.WIDTH = width
        self.HEIGHT = height
        self.STEP = step
        self.GAMMA = gamma
        self.OUTPUTDIR = output_dir
        self.BLACKTHR = black_thr
        self.WHITETHR = white_thr
        self.FILTER = filter_size
        self.INPUTPRE = input_prefix
        self.CONFIG_FILE = config_file

    def generate(self):
        GC_STEP = int(self.STEP / 2)

        if not os.path.exists(self.OUTPUTDIR):
            os.mkdir(self.OUTPUTDIR)

        imgs = []

        print('Generating sinusoidal patterns ...')
        angle_vel = np.array((6, 4)) * np.pi / self.STEP
        xs = np.array(range(self.WIDTH))
        for i in range(1, 3):
            for phs in range(1, 4):
                vec = 0.5 * (np.cos(xs * angle_vel[i - 1] + np.pi * (phs - 2) * 2 / 3) + 1)
                vec = 255 * (vec ** self.GAMMA)
                vec = np.round(vec)
                img = np.zeros((self.HEIGHT, self.WIDTH), np.uint8)
                for y in range(self.HEIGHT):
                    img[y, :] = vec
                imgs.append(img)

        ys = np.array(range(self.HEIGHT))
        for i in range(1, 3):
            for phs in range(1, 4):
                vec = 0.5 * (np.cos(ys * angle_vel[i - 1] + np.pi * (phs - 2) * 2 / 3) + 1)
                vec = 255 * (vec ** self.GAMMA)
                vec = np.round(vec)
                img = np.zeros((self.HEIGHT, self.WIDTH), np.uint8)
                for x in range(self.WIDTH):
                    img[:, x] = vec
                imgs.append(img)

        print('Generating graycode patterns ...')
        gc_height = int((self.HEIGHT - 1) / GC_STEP) + 1
        gc_width = int((self.WIDTH - 1) / GC_STEP) + 1

        graycode = cv2.structured_light_GrayCodePattern.create(gc_width, gc_height)
        patterns = graycode.generate()[1]
        for pat in patterns:
            img = np.zeros((self.HEIGHT, self.WIDTH), np.uint8)
            for y in range(self.HEIGHT):
                for x in range(self.WIDTH):
                    img[y, x] = pat[int(y / GC_STEP), int(x / GC_STEP)]
            imgs.append(img)
        imgs.append(255 * np.ones((self.HEIGHT, self.WIDTH), np.uint8))  # white
        imgs.append(np.zeros((self.HEIGHT, self.WIDTH), np.uint8))  # black

        for i, img in enumerate(imgs):
            cv2.imwrite(self.OUTPUTDIR + '/pat' + str(i).zfill(2) + '.png', img)

        print('Saving config file ...')
        fs = cv2.FileStorage(self.OUTPUTDIR + '/config.xml', cv2.FILE_STORAGE_WRITE)
        fs.write('disp_width', self.WIDTH)
        fs.write('disp_height', self.HEIGHT)
        fs.write('step', self.STEP)
        fs.release()

        print('Done')

    def decode(self):
        fs = cv2.FileStorage(self.CONFIG_FILE, cv2.FILE_STORAGE_READ)
        DISP_WIDTH = int(fs.getNode('disp_width').real())
        DISP_HEIGHT = int(fs.getNode('disp_height').real())
        STEP = int(fs.getNode('step').real())
        GC_STEP = int(STEP / 2)
        fs.release()

        gc_width = int((DISP_WIDTH - 1) / GC_STEP) + 1
        gc_height = int((DISP_HEIGHT - 1) / GC_STEP) + 1
        graycode = cv2.structured_light_GrayCodePattern.create(gc_width, gc_height)
        graycode.setBlackThreshold(self.BLACKTHR)
        graycode.setWhiteThreshold(self.WHITETHR)

        print('Loading images ...')
        re_num = re.compile(r'(\d+)')

        def numerical_sort(text):
            return int(re_num.split(text)[-2])

        filenames = sorted(glob.glob(self.INPUTPRE + '\pat*.png'), key=numerical_sort)
        if len(filenames) != graycode.getNumberOfPatternImages() + 14:
            print('Number of images is not right (right number is ' + str(graycode.getNumberOfPatternImages() + 14) + ')')
            return

        imgs = []
        for f in filenames:
            imgs.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE))
        ps_imgs = imgs[0:12]
        gc_imgs = imgs[12:]
        black = gc_imgs.pop()
        white = gc_imgs.pop()
        CAM_WIDTH = white.shape[1]
        CAM_HEIGHT = white.shape[0]

        print('Decoding images ...')

        def decode_ps(pimgs):
            pimg1 = pimgs[0].astype(np.float32)
            pimg2 = pimgs[1].astype(np.float32)
            pimg3 = pimgs[2].astype(np.float32)
            return np.arctan2(np.sqrt(3) * (pimg1 - pimg3), 2 * pimg2 - pimg1 - pimg3)

        ps_map_x1 = decode_ps(ps_imgs[0:3])
        ps_map_x2 = decode_ps(ps_imgs[3:6])
        ps_map_y1 = decode_ps(ps_imgs[6:9])
        ps_map_y2 = decode_ps(ps_imgs[9:12])

        gc_map = np.zeros((CAM_HEIGHT, CAM_WIDTH, 2), np.int16)
        mask = np.zeros((CAM_HEIGHT, CAM_WIDTH), np.uint8)
        for y in range(CAM_HEIGHT):
            for x in range(CAM_WIDTH):
                if int(white[y, x]) - int(black[y, x]) <= self.BLACKTHR:
                    continue
                err, proj_pix = graycode.getProjPixel(gc_imgs, x, y)
                if not err:
                    gc_map[y, x, :] = np.array(proj_pix)
                    mask[y, x] = 255

        if self.FILTER != 0:
            print('Applying smoothing filter ...')
            ext_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((self.FILTER * 2 + 1, self.FILTER * 2 + 1)))
            for y in range(CAM_HEIGHT):
                for x in range(CAM_WIDTH):
                    if mask[y, x] == 0 and ext_mask[y, x] != 0:
                        sum_x = 0
                        sum_y = 0
                        cnt = 0
                        for dy in range(-self.FILTER, self.FILTER + 1):
                            for dx in range(-self.FILTER, self.FILTER + 1):
                                ty = y + dy
                                tx = x + dx
                                if ((dy != 0 or dx != 0) and ty >= 0 and ty < CAM_HEIGHT and tx >= 0 and tx < CAM_WIDTH and mask[ty, tx] != 0):
                                    sum_x += gc_map[ty, tx, 0]
                                    sum_y += gc_map[ty, tx, 1]
                                    cnt += 1
                        if cnt != 0:
                            gc_map[y, x, 0] = np.round(sum_x / cnt)
                            gc_map[y, x, 1] = np.round(sum_y / cnt)

            mask = ext_mask

        def decode_pixel(gc, ps1, ps2):
            dif = None
            if ps1 > ps2:
                if ps1 - ps2 > np.pi * 4 / 3:
                    dif = (ps2 - ps1) + 2 * np.pi
                else:
                    dif = ps1 - ps2
            else:
                if ps2 - ps1 > np.pi * 4 / 3:
                    dif = (ps1 - ps2) + 2 * np.pi
                else:
                    dif = ps2 - ps1

            p = None
            if gc % 2 == 0:
                p = ps1
                if dif > np.pi / 6 and p < 0:
                    p = p + 2 * np.pi
                if dif > np.pi / 2 and p < np.pi * 7 / 6:
                    p = p + 2 * np.pi
            else:
                p = ps1
                if dif > np.pi * 5 / 6 and p > 0:
                    p = p - 2 * np.pi
                if dif < np.pi / 2 and p < np.pi / 6:
                    p = p + 2 * np.pi
                p = p + np.pi
            return gc * GC_STEP + self.STEP * p / 3 / 2 / np.pi

        print('Decoding each pixel ...')
        viz = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), np.uint8)
        res_list = []
        for y in range(CAM_HEIGHT):
            for x in range(CAM_WIDTH):
                if mask[y, x] != 0:
                    est_x = decode_pixel(gc_map[y, x, 0], ps_map_x1[y, x], ps_map_x2[y, x])
                    est_y = decode_pixel(gc_map[y, x, 1], ps_map_y1[y, x], ps_map_y2[y, x])
                    viz[y, x, :] = (est_x, est_y, 128)
                    res_list.append((y, x, est_y, est_x))

        print('Exporting result ...')
        if not os.path.exists(self.OUTPUTDIR):
            os.mkdir(self.OUTPUTDIR)
        cv2.imwrite(self.OUTPUTDIR + '/vizualized.png', viz)
        with open(self.OUTPUTDIR + '/camera2display.csv', mode='w') as f:
            f.write('camera_y, camera_x, display_y, display_x\n')
            for (cam_y, cam_x, disp_y, disp_x) in res_list:
                f.write(str(cam_y) + ', ' + str(cam_x) + ', ' + str(disp_y) + ', ' + str(disp_x) + '\n')

        print('Done')


if __name__ == '__main__':
    # Set the parameters here
    width = 1920
    height = 1080
    step = 32
    gamma = 1.0
    output_dir = 'output_test'
    black_thr = 5
    white_thr = 40
    filter_size = 0
    input_prefix = r"sample_data\object1"
    config_file = r"sample_data\object1\config.xml"

    ps = PhaseShifting(width, height, step, gamma, output_dir, black_thr, white_thr, filter_size, input_prefix, config_file)
    ps.generate()
    ps.decode()