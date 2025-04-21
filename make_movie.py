# Does what it says on the tin
import os
import cv2 
import re

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def generate_video(run, var):
    figs_dir = 'figures/figs2D_'+run+'_'+var+'/'
    figs = sorted(os.listdir(figs_dir))#,key=numericalSort)
    frame = cv2.imread(figs_dir + figs[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter('plume2D_' + run + '_'+var+'.mp4', 
                            fourcc=fourcc, 
                            fps=3, 
                            frameSize=(width, height))
    for fig in figs:
        img = cv2.imread(figs_dir+fig)
        video.write(img)
    cv2.destroyAllWindows()
    video.release()
    print(run + '_' + var + ' video saved')

if __name__ == "__main__":
    #generate_video('mrb_033', 'rho_theta')
    generate_video('mrb_049', 'T')
    generate_video('mrb_049', 'S')
    generate_video('mrb_049', 'quiver')
    generate_video('mrb_049', 'rho_theta')