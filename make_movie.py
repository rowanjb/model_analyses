# Does what it says on the tin
import os
import cv2 
import re

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def generate_video(fig_dir,run):
    figs = sorted(os.listdir(fig_dir))#,key=numericalSort)
    frame = cv2.imread(fig_dir + figs[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter('plume_' + run + '.mp4', 
                            fourcc=fourcc, 
                            fps=3, 
                            frameSize=(width, height))
    for fig in figs:
        img = cv2.imread(fig_dir+fig)
        video.write(img)
    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    #generate_video('./figs_mrb_002/','mrb_002')
    #generate_video('./figs_mrb_003/','mrb_003')
    #generate_video('./figs_mrb_004/','mrb_004')
    #generate_video('./figs_mrb_005/','mrb_005')
    #generate_video('./figs_mrb_006/','mrb_006')
    #generate_video('./figs_mrb_007/','mrb_007')
    generate_video('./figs_mrb_008/','mrb_008')
    
