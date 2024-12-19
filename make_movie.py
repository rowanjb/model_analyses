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
    generate_video('/albedo/home/robrow001/MITgcm/verification/tutorial_deep_convection_cp/figs_run_parallel/','parallel')
    generate_video('/albedo/home/robrow001/MITgcm/verification/tutorial_deep_convection_cp/figs_run_40m/','40m')
    generate_video('/albedo/home/robrow001/MITgcm/verification/tutorial_deep_convection_cp/figs_run_80m/','80m')
    generate_video('/albedo/home/robrow001/MITgcm/verification/tutorial_deep_convection_cp/figs_run_160m/','160m')