
import av
from io import BytesIO
from typing import Optional, Any
from PIL import Image
import numpy as np
import cv2
import filetype

async def video_parser(video_data: bytes, llm: Any = None, fps: float = 1) -> str:
    """Extract frames from a video at specified fps and use LLM to describe the content."""
    if not isinstance(video_data, bytes):
        raise TypeError(f"Expected bytes for video_data, got {type(video_data)}")

    if not video_data:
        raise ValueError("video_data is empty")

    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    # llm is optional in signature but required logically if we want description
    # if llm is None:
    #    raise ValueError("LLM instance is  required for video parsing")

    # Validate video format
    kind = filetype.guess(video_data)
    if not kind or kind.mime not in {"video/mp4", "video/avi", "video/mpeg", "video/quicktime", "video/x-matroska"}:
        # Relaxed check or log warning. Some containers might be guessed differently.
        # But keeping user logic mostly:
        if kind and kind.mime not in {"video/mp4", "video/avi", "video/mpeg"}:
             pass # pass for now to allow trying AV generic open

    # Load video from bytes using pyav
    video_stream_obj = BytesIO(video_data)
    try:
        container = av.open(video_stream_obj)
    except Exception as e:
        raise ValueError(f"Failed to initialize AV container: {str(e)}")

    # Get video stream
    # video_stream = next((s for s in container.streams if s.type == 'video'), None)
    # PyAV streams are accessed slightly differently or filtered. 
    # container.streams.video[0] is common
    if not container.streams.video:
        raise ValueError("No video stream found in container")
    
    video_stream = container.streams.video[0]

    # Get video properties
    # fps might be in average_rate
    video_fps = float(video_stream.average_rate or video_stream.rate or 24.0)
    if video_fps <= 0:
        video_fps = 24.0 # Fallback

    # total_frames
    total_frames = video_stream.frames
    if not total_frames:
         # Estimate
         duration = float(video_stream.duration * video_stream.time_base) if video_stream.duration else 0
         total_frames = int(duration * video_fps)

    frames = []
    
    # Logic for extraction
    # If fps=1, pick middle frame of each second
    if fps == 1:
        num_seconds = max(1, int(total_frames / video_fps))
        for second in range(num_seconds):
            target_time = second + 0.5
            # Seek
            # PyAV seek is to timestamp (in time_base units)
            timestamp = int(target_time / video_stream.time_base)
            container.seek(timestamp, stream=video_stream)
            
            # Decode frames until we find the one closest to target? 
            # Simplified: decode one frame after seek
            try:
                for frame in container.decode(video_stream):
                    frame_np = frame.to_ndarray(format='bgr24')
                    # Encode to jpg in memory and reload to PIL (User logic)
                    # changing Image.from_bytes to Image.open(BytesIO)
                    _, frame_bytes = cv2.imencode('.jpg', frame_np)
                    frames.append(Image.open(BytesIO(frame_bytes.tobytes())))
                    break # Just one frame per second
            except:
                continue
    else:
        # Uniform sampling
        frame_interval = max(1, int(video_fps / fps))
        count = 0
        for frame in container.decode(video_stream):
            if count % frame_interval == 0:
                 frame_np = frame.to_ndarray(format='bgr24')
                 _, frame_bytes = cv2.imencode('.jpg', frame_np)
                 frames.append(Image.open(BytesIO(frame_bytes.tobytes())))
            count += 1

    container.close()

    if not frames:
        return "No frames extracted from video."

    if llm is None:
        return f"Headers parsed. extracted {len(frames)} frames. No LLM provided for description."

    # Use LLM to describe the video
    prompt = "Describe what the video shows based on the provided frames."
    print("""Using LLM to process video frames...""")
    
    # Use LangChain invoke/ainvoke with vision content format
    if hasattr(llm, "invoke") or hasattr(llm, "ainvoke"):
        # Construct message
        from langchain_core.messages import HumanMessage
        import base64
        
        # Limit frames and resize to avoid "Request Entity Too Large" error
        MAX_FRAMES = 5
        MAX_SIZE = (512, 512)
        
        content = [{"type": "text", "text": prompt}]
        for i, img in enumerate(frames[:MAX_FRAMES]):
            # Resize image maintaining aspect ratio
            img.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)
            
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=70) # Lower quality to reduce size
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
            
        message = HumanMessage(content=content)
        
        try:
            if hasattr(llm, "ainvoke"):
                resp = await llm.ainvoke([message])
            else:
                resp = llm.invoke([message])
            return resp.content.strip()
        except Exception as e:
            return f"Error invoking LLM on video frames: {e}"

    return "LLM does not support video description generation."
