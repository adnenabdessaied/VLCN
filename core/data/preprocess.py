import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import skvideo.io as skv
import torch
import pickle
from PIL import Image
import tqdm
import numpy as np
from model.C3D import C3D
import json
from torchvision.models import vgg19
import torchvision.transforms as transforms
import torch.nn as nn
import argparse


def _select_frames(path, frame_num):
    """Select representative frames for video.
    Ignore some frames both at begin and end of video.
    Args:
        path: Path of video.
    Returns:
        frames: list of frames.
    """
    frames = list()
    video_data = skv.vread(path)
    total_frames = video_data.shape[0]
    # Ignore some frame at begin and end.
    for i in np.linspace(0, total_frames, frame_num + 2)[1:frame_num + 1]:
        frame_data = video_data[int(i)]
        img = Image.fromarray(frame_data)
        img = img.resize((224, 224), Image.BILINEAR)
        frame_data = np.array(img)
        frames.append(frame_data)
    return frames

def _select_clips(path, clip_num):
    """Select self.batch_size clips for video. Each clip has 16 frames.
    Args:
        path: Path of video.
    Returns:
        clips: list of clips.
    """
    clips = list()
    # video_info = skvideo.io.ffprobe(path)
    video_data = skv.vread(path)
    total_frames = video_data.shape[0]
    height = video_data[1]
    width = video_data.shape[2]
    for i in np.linspace(0, total_frames, clip_num + 2)[1:clip_num + 1]:
        # Select center frame first, then include surrounding frames
        clip_start = int(i) - 8
        clip_end = int(i) + 8
        if clip_start < 0:
            clip_end = clip_end - clip_start
            clip_start = 0
        if clip_end > total_frames:
            clip_start = clip_start - (clip_end - total_frames)
            clip_end = total_frames
        clip = video_data[clip_start:clip_end]
        new_clip = []
        for j in range(16):
            frame_data = clip[j]
            img = Image.fromarray(frame_data)
            img = img.resize((112, 112), Image.BILINEAR)
            frame_data = np.array(img) * 1.0
            # frame_data -= self.mean[j]
            new_clip.append(frame_data)
        clips.append(new_clip)
    return clips
    
def preprocess_videos(video_dir, frame_num, clip_num):
    frames_dir = os.path.join(os.path.dirname(video_dir), 'frames')
    os.mkdir(frames_dir)

    clips_dir = os.path.join(os.path.dirname(video_dir), 'clips')
    os.mkdir(clips_dir)

    for video_name in tqdm.tqdm(os.listdir(video_dir)):
        video_path = os.path.join(video_dir, video_name)
        frames = _select_frames(video_path, frame_num)
        clips = _select_clips(video_path, clip_num)

        with open(os.path.join(frames_dir, video_name.split('.')[0] + '.pkl'), "wb") as f:
            pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(clips_dir, video_name.split('.')[0] + '.pkl'), "wb") as f:
            pickle.dump(clips, f, protocol=pickle.HIGHEST_PROTOCOL)


def generate_video_features(path_frames, path_clips, c3d_path):
    device = torch.device('cuda:0')
    frame_feat_dir = os.path.join(os.path.dirname(path_frames), 'frame_feat')
    os.makedirs(frame_feat_dir, exist_ok=True)

    clip_feat_dir = os.path.join(os.path.dirname(path_frames), 'clip_feat')
    os.makedirs(clip_feat_dir, exist_ok=True)

    cnn = vgg19(pretrained=True)
    in_features = cnn.classifier[-1].in_features
    cnn.classifier = nn.Sequential(
        *list(cnn.classifier.children())[:-1])    # remove last fc layer
    cnn.to(device).eval()
    c3d = C3D()
    c3d.load_state_dict(torch.load(c3d_path))
    c3d.to(device).eval()
    transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406),
                                                        (0.229, 0.224, 0.225))])
    for vid_name in tqdm.tqdm(os.listdir(path_frames)):
        frame_path = os.path.join(path_frames, vid_name)
        clip_path = os.path.join(path_clips, vid_name)

        frames = pickle.load(open(frame_path, 'rb'))
        clips = pickle.load(open(clip_path, 'rb'))

        frames = [transform(f) for f in frames]
        frame_feat = []
        clip_feat = []

        for frame in frames:
            with torch.no_grad():
                feat = cnn(frame.unsqueeze(0).to(device))
            frame_feat.append(feat)
        for clip in clips:
            # clip has shape (c x f x h x w)
            clip = torch.from_numpy(np.float32(np.array(clip)))
            clip = clip.transpose(3, 0)
            clip = clip.transpose(3, 1)
            clip = clip.transpose(3, 2).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = c3d(clip)
            clip_feat.append(feat)
        frame_feat = torch.cat(frame_feat, dim=0)
        clip_feat = torch.cat(clip_feat, dim=0)

        torch.save(frame_feat, os.path.join(frame_feat_dir, vid_name.split('.')[0] + '.pt'))
        torch.save(clip_feat, os.path.join(clip_feat_dir, vid_name.split('.')[0] + '.pt'))

def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='Preprocessing Args')

    parser.add_argument('--RAW_VID_PATH', dest='RAW_VID_PATH',
                      help='The path to the raw videos',
                      required=True,
                      type=str)

    parser.add_argument('--FRAMES_OUTPUT_DIR', dest='FRAMES_OUTPUT_DIR',
                      help='The directory where the processed frames and their features will be stored',
                      required=True,
                      type=str)

    parser.add_argument('--CLIPS_OUTPUT_DIR', dest='FRAMES_OUTPUT_DIR',
                      help='The directory where the processed frames and their features will be stored',
                      required=True,
                      type=str)

    parser.add_argument('--C3D_PATH', dest='C3D_PATH',
                      help='Pretrained C3D path',
                      required=True,
                      type=str)

    parser.add_argument('--NUM_SAMPLES', dest='NUM_SAMPLES',
               help='The number of frames/clips to be sampled from the video',
               default=20,
               type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    preprocess_videos(args.RAW_VID_PATH, args.NUM_SAMPLES, args.NUM_SAMPLES)
    frames_dir = os.path.join(os.path.dirname(args.RAW_VID_PATH), 'frames')
    clips_dir = os.path.join(os.path.dirname(args.RAW_VID_PATH), 'clips')
    generate_video_features(frames_dir, clips_dir)
