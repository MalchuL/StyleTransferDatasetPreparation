from __future__ import print_function
import os
from pathlib import Path
import argparse
import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

STATS_FILE_PATH = 'testvideo.stats.csv'
FORMAT_FILES = ['.mkv','.avi','.mp4']

parser = argparse.ArgumentParser()

parser.add_argument('--start_time', default=0, type=int, help='get frames starts from this second')
parser.add_argument('--end_time', default=None, type=int, help='get frames ends to this second')
parser.add_argument('--video_folder', required=True, help='path to folder with videos')
parser.add_argument('--output_folder', required=True, help='path to folder with frames')
parser.add_argument('--num_images', default=5, type=int, help='num frames from scene')

args = parser.parse_args()







def main(file_path, video_name, start_time, end_time, output_folder, num_images):

    # Create a video_manager point to video file testvideo.mp4. Note that multiple
    # videos can be appended by simply specifying more file paths in the list
    # passed to the VideoManager constructor. Note that appending multiple videos
    # requires that they all have the same frame size, and optionally, framerate.
    video_manager = VideoManager([file_path])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    # Add ContentDetector algorithm (constructor takes detector options like threshold).
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()

    try:
        # If stats file exists, load it.
        if os.path.exists(STATS_FILE_PATH):
            # Read stats from CSV file opened in read mode:
            with open(STATS_FILE_PATH, 'r') as stats_file:
                stats_manager.load_from_csv(stats_file, base_timecode)

        if start_time is not None:
            start_time = base_timecode + start_time    # 00:06:20.667
        if end_time is not None:
            end_time = base_timecode + end_time    # 00:20:00:000

        #end_time = None  # last frame

        # Set video_manager duration to read frames from 00:00:00 to 00:00:20.
        video_manager.set_duration(start_time=start_time, end_time=end_time)

        # Set downscale factor to improve processing speed (no args means default).
        video_manager.set_downscale_factor()

        # Start video_manager.
        video_manager.start()

        # Perform scene detection on video_manager.
        scene_manager.detect_scenes(frame_source=video_manager)

        # Obtain list of detected scenes.
        scene_list = scene_manager.get_scene_list(base_timecode)
        # Like FrameTimecodes, each scene in the scene_list can be sorted if the
        # list of scenes becomes unsorted.

        print('List of scenes obtained:')
        for i, scene in enumerate(scene_list):
            print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
                i+1,
                scene[0].get_timecode(), scene[0].get_frames(),
                scene[1].get_timecode(), scene[1].get_frames(),))

        # We only write to the stats file if a save is required:
        if stats_manager.is_save_required():
            with open(STATS_FILE_PATH, 'w') as stats_file:
                stats_manager.save_to_csv(stats_file, base_timecode)

        # Save images into folder 3 images means, start, end and middle
        print('Start saving')
        os.makedirs(output_folder, exist_ok=True)
        print(scene_list)
        images_was_saved = scenedetect.scene_manager.save_images(scene_list, video_manager, num_images=num_images, output_dir=output_folder, show_progress=True)
        print(images_was_saved)
    except Exception as e:
        print('error', e)

    finally:
        video_manager.release()

if __name__ == "__main__":
    full_path = Path(args.video_folder)
    filenames = []
    for ext in FORMAT_FILES:
        filenames.extend(full_path.rglob('*' + ext))

    full_paths = list(map(str, filenames))

    for i, path in enumerate(full_paths):
        main(path, "VIDEO" + str(i), args.start_time, args.end_time, args.output_folder, args.num_images)