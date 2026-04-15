"""
This file contains the functions to load videos.
"""
import os
import torch
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

try:
    import av
    AV_AVAILABLE = True
except ImportError:
    av = None
    AV_AVAILABLE = False

try:
    from video_reader import PyVideoReader  # pip install video-reader-rs
    PYVIDEO_READER_AVAILABLE = True
except ImportError:
    PyVideoReader = None
    PYVIDEO_READER_AVAILABLE = False

try:
    from decord import VideoReader
    DECORD_AVAILABLE = True
except ImportError:
    VideoReader = None
    DECORD_AVAILABLE = False


def _build_frame_indices(total_frames, actual_fps, fps, num_frames,
                         sample_all_frames, seg_start_time=None,
                         seg_end_time=None):
    duration = float(total_frames) / actual_fps if actual_fps > 0 else 0.0
    if seg_start_time is not None and seg_end_time is not None:
        duration = seg_end_time - seg_start_time
        seg_start_frame = int(seg_start_time * actual_fps)
        seg_end_frame = int(seg_end_time * actual_fps)
    else:
        seg_start_frame = 0
        seg_end_frame = total_frames - 1

    if fps is not None:
        total_frames_to_take = max(1, int(duration * fps))
        frame_indices = np.linspace(
            seg_start_frame, seg_end_frame, total_frames_to_take, dtype=int)
    else:
        frame_indices = list(range(seg_start_frame, seg_end_frame + 1))

    if not sample_all_frames:
        available_frames = len(frame_indices)
        if num_frames > available_frames:
            print(f"Warning: num_frames ({num_frames}) is greater than "
                  f"available frames ({available_frames})")
            num_frames = available_frames
        sample_indices = np.linspace(
            0, len(frame_indices) - 1, num_frames, dtype=int)
        frame_indices = [frame_indices[i] for i in sample_indices]
        frame_id_dict = {i: idx for i, idx in enumerate(sample_indices)}
    else:
        frame_id_dict = None

    return frame_indices, frame_id_dict


def _load_video_cv2(vid_path, return_tensor=False, device=None, use_float=False,
                    num_frames=8, sample_all_frames=False, fps=None,
                    seg_start_time=None, seg_end_time=None):
    if not CV2_AVAILABLE:
        return False, None, None

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        return False, None, None

    actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if actual_fps <= 0:
        actual_fps = 30.0

    all_frames = []
    target_size = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if target_size is None:
            target_size = (frame.shape[1], frame.shape[0])
        elif (frame.shape[1], frame.shape[0]) != target_size:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(frame)
    cap.release()

    if not all_frames:
        return False, None, None

    frame_indices, frame_id_dict = _build_frame_indices(
        len(all_frames), actual_fps, fps, num_frames, sample_all_frames,
        seg_start_time=seg_start_time, seg_end_time=seg_end_time)
    max_index = len(all_frames) - 1
    frame_indices = [min(max(int(idx), 0), max_index) for idx in frame_indices]
    frames = np.stack([all_frames[idx] for idx in frame_indices])

    if use_float:
        frames = frames.astype(np.float32)

    frames = frames[None]
    frames = np.transpose(frames, (0, 1, 4, 2, 3))

    if return_tensor:
        frames = torch.from_numpy(frames)
        if torch.isnan(frames).any() or torch.isinf(frames).any():
            raise ValueError("Frames contain NaNs or Infs")
        if device is not None:
            frames = frames.to(device)
    return True, frames, frame_id_dict


def load_video_pyvideo_reader(vid_path, return_tensor=False, device=None,
                              use_float=False, num_frames=8,
                              sample_all_frames=False, fps=None,
                              seg_start_time=None, seg_end_time=None):
    '''
    load video from file with regular interval sampling using Decord.
    Args:
        vid_path: path to video file
        return_tensor: if True, return torch tensor, otherwise numpy array
        device: device to load tensor to
        use_float: if True, convert frames to float32, otherwise keep uint8
        num_frames: number of frames to sample (default=8)
        sample_all_frames: if True, return all frames without subsampling
        fps: if set, load video at this frame rate
        seg_start_time: if set, load video from this time, in seconds
        seg_end_time: if set, load video to this time, in seconds
    Returns:
        frames: (B, T, C, H, W) numpy array or tensor, where T = num_frames
        frame_id_dict: dictionary mapping sampled frame indices to original  indices
    '''
    print(f"Processing {vid_path}...")
    assert os.path.exists(vid_path), f"Video file {vid_path} does not exist"


    if PYVIDEO_READER_AVAILABLE:
        vr = PyVideoReader(vid_path)
        total_frames = len(vr)
        duration = float(vr.get_info()['duration'])
        actual_fps = float(vr.get_info()['fps'])
        read_batch = vr.get_batch
    elif DECORD_AVAILABLE:
        vr = VideoReader(vid_path, num_threads=1)
        total_frames = len(vr)
        actual_fps = float(vr.get_avg_fps())
        duration = float(total_frames) / actual_fps if actual_fps > 0 else 0.0
        read_batch = lambda idxs: vr.get_batch(idxs).asnumpy()
    elif AV_AVAILABLE:
        return load_video(
            vid_path,
            return_tensor=return_tensor,
            device=device,
            use_float=use_float,
            num_frames=num_frames,
            sample_all_frames=sample_all_frames,
            fps=fps,
        )
    else:
        raise ImportError(
            "No video backend available. Install video-reader-rs, decord, or av."
        )

    frame_indices, frame_id_dict = _build_frame_indices(
        total_frames, actual_fps, fps, num_frames, sample_all_frames,
        seg_start_time=seg_start_time, seg_end_time=seg_end_time)

    # Read frames
    try:
        frames = read_batch(frame_indices)  # T,H,W,C
    except Exception as exc:
        print(f"Primary video reader failed for {vid_path}: {exc}")
        return _load_video_cv2(
            vid_path,
            return_tensor=return_tensor,
            device=device,
            use_float=use_float,
            num_frames=num_frames,
            sample_all_frames=sample_all_frames,
            fps=fps,
            seg_start_time=seg_start_time,
            seg_end_time=seg_end_time,
        )

    # Convert to float if needed
    if use_float:
        frames = frames.astype(np.float32)

    # Add batch dimension and rearrange to B,T,C,H,W
    frames = frames[None]  # B,T,H,W,C
    frames = np.transpose(frames, (0, 1, 4, 2, 3))  # B,T,C,H,W

    if return_tensor:
        frames = torch.from_numpy(frames)
        if torch.isnan(frames).any() or torch.isinf(frames).any():
            raise ValueError("Frames contain NaNs or Infs")
        if device is not None:
            frames = frames.to(device)
    return True, frames, frame_id_dict


def load_video(vid_path, return_tensor=False, device=None, use_float=False,
               num_frames=8, sample_all_frames=False, fps=None):
    '''
    load video from webm file with regular interval sampling.
    Args:
        vid_path: path to video file
        return_tensor: if True, return torch tensor, otherwise numpy array
        device: device to load tensor to
        num_frames: number of frames to sample (default=8)
        sample_all_frames: if True, return all frames without subsampling
        fps: if set, load video at this frame rate
    Returns:
        frames: (B, T, C, H, W) numpy array or tensor, where T = num_frames
    '''
    print(f"Processing {vid_path}...FPS: {fps}")
    assert os.path.exists(vid_path), f"Video file {vid_path} does not exist"

    if not AV_AVAILABLE:
        if DECORD_AVAILABLE:
            return load_video_pyvideo_reader(
                vid_path,
                return_tensor=return_tensor,
                device=device,
                use_float=use_float,
                num_frames=num_frames,
                sample_all_frames=sample_all_frames,
                fps=fps,
            )
        return False, None, None

    # Option 2: Using PyAV
    # pylint: disable=broad-exception-caught
    try:
        container = av.open(vid_path)
    except (OSError, ValueError, Exception):
        return False, None, None

    # Get video stream
    stream = container.streams.video[0]
    original_fps = float(stream.average_rate)

    frames = []
    if fps is not None:
        # Calculate frame interval based on desired fps
        interval = int(round(original_fps / fps))
        frame_count = 0
        for frame in container.decode(video=0):
            if frame_count % interval == 0:
                # Convert to RGB numpy array
                frame = frame.to_ndarray(format='rgb24')
                if use_float:
                    frame = frame.astype(np.float32)
                frames.append(frame)
            frame_count += 1
    else:
        # Original behavior without fps control
        for frame in container.decode(video=0):
            frame = frame.to_ndarray(format='rgb24')
            if use_float:
                frame = frame.astype(np.float32)
            frames.append(frame)

    total_frames = len(frames)
    container.close()

    # Stack frames into a single array and rearrange dimensions
    frames = np.stack(frames)[None]  # B,T,H,W,C
    frames = np.transpose(frames, (0, 1, 4, 2, 3))  # B,T,C,H,W
    frame_id_dict = None

    if not sample_all_frames:
        # Ensure num_frames does not exceed total_frames
        if num_frames > total_frames:
            print(f"Warning: num_frames ({num_frames}) is greater than "
                  f"total_frames ({total_frames}). Adjusting num_frames to total_frames.")
            num_frames = total_frames  # Set num_frames to total_frames
        # Calculate indices for uniformly sampled frames
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        frames = frames[:, frame_indices]  # (1, T, C, H, W)
        frame_id_dict = {i: frame_indices[i] for i in range(num_frames)}

    if return_tensor:
        frames = torch.from_numpy(frames).to(device)

    return True, frames, frame_id_dict
