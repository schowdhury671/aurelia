import os
import cv2
import re
import json
import textwrap
import pathlib
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from pytube import YouTube
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from skimage.metrics import structural_similarity as ssim
import PIL.Image

# Generative AI imports
from google.colab import userdata
from IPython.display import Markdown, Video
import google.generativeai as genai
import argparse
parser = argparse.ArgumentParser()

# Configure Gemini API
# GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
#genai.configure(api_key="")

# Constants
# DATA_PATH = ''
# AUDIO_DATA_PATH = ''

# Utility
def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# --- Main Function to run Gemini ---
def run_gemini(vid, question_text, cropped_audio_path):
    try:
        cap = cv2.VideoCapture(vid)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_directory = 'selected_frames'
        os.makedirs(output_directory, exist_ok=True)

        selected_frames = []
        previous_frame = None
        threshold = 0.6

        for frame_idx in tqdm(range(n_frames), desc="Processing Frames"):
            ret, img = cap.read()
            if not ret:
                break

            b, g, r = cv2.split(img)

            if previous_frame is not None:
                ssim_b, _ = ssim(previous_frame[0], b, full=True)
                ssim_g, _ = ssim(previous_frame[1], g, full=True)
                ssim_r, _ = ssim(previous_frame[2], r, full=True)
                similarity_index = (ssim_b + ssim_g + ssim_r) / 3

                if similarity_index < threshold:
                    selected_frames.append(img)
                    vid_ytid = os.path.splitext(os.path.basename(vid))[0]
                    frame_filename = os.path.join(output_directory, f"{vid_ytid}_frame_{frame_idx:04d}.png")
                    print("writing image frame:", frame_filename)
                    cv2.imwrite(frame_filename, img)

            previous_frame = cv2.split(img)

        cap.release()
        print(f"\nTotal key frames: {len(selected_frames)}")

        images = []
        for img_file in os.listdir('selected_frames'):
            if vid_ytid in img_file:
                img_path = os.path.join('selected_frames', img_file)
                print("loading frame:", img_path)
                images.append(PIL.Image.open(img_path))

        if not images:
            print("No key frames selected.")
            return None

        chosen_frames = images[:min(5, len(images))]
        audio_file = genai.upload_file(path=cropped_audio_path)

        prompt = question_text
        model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        response = model.generate_content([prompt, *chosen_frames, audio_file])
        return response.text

    except Exception as e:
        print("Couldn't process video:", str(e))
        return None

# --- Audio Extraction ---
def extract_audio(input_video_path, output_audio_path):
    video_clip = VideoFileClip(input_video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_audio_path)
    video_clip.close()
    audio_clip.close()
    print("Audio extracted successfully.")

# --- GPT-based Summarizer ---
def run_gpt_summarizer(reasoning_steps, reasoning_answer, final_answer, question):
    instruct_prompt = """Given the reasoning steps, the answer to the reasoning steps, and the final response
    for the question, generate a detailed caption that describes the content of the video and the audio.
    Discard any world knowledge not present in the video/audio context."""

    prompt = f"""Reasoning: {reasoning_steps}
    Reasoning answer: {reasoning_answer}
    Final answer: {final_answer}
    Question: {question}"""
    MODEL = "gpt-4o"

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": instruct_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

# --- Feedback Loop ---
def feedback(score, reasoning_steps, question, video, audio):
#     instruction = f"""The reasoning steps you previously generated: '{reasoning_steps}' for the question: '{question}'
# received a score of {score}/10, which suggests they might be inadequate.
# Now, given the video, audio, and the question, generate the correct reasoning steps.

# Follow this format:
# Task_1: Detailed reasoning steps.
# Task_2: Detailed answers to Task 1 steps.
# Task_3: Final answer."""
    instruction = (
    f"The previously generated reasoning steps:\n'{reasoning_steps}'\n"
    f"were evaluated for the question:\n'{question}'\n"
    f"and received a score of {score} out of 10. This indicates that the reasoning may be incomplete, inaccurate, or not sufficiently grounded in the provided video and audio context.\n\n"
    
    "Your task now is to re-evaluate the question using both the video and audio content. Carefully generate a new, accurate reasoning process that is grounded solely in the visual and auditory evidence.\n\n"
    
    "Please follow the exact format below in your response:\n\n"
    "Task_1: [Step-by-step reasoning to solve the question without giving the final answer. Be thorough and logically structured.]\n"
    "Task_2: [Answer each reasoning step from Task 1 in detail. Provide grounded and contextual insights.]\n"
    "Task_3: [Provide the final answer to the question.]\n\n"
    
    "Return your response strictly as a Python dictionary in this format:\n"
    "{\n"
    "  'Task_1': <Your detailed reasoning>,\n"
    "  'Task_2': <Your step-by-step answers>,\n"
    "  'Task_3': <Final answer>\n"
    "}"
)


    return run_gemini(video, instruction, audio)



if __name__=="__main__":

    # Adding arguments
    parser.add_argument("--save_path",default="reason_data.json", help = "path to json file for saving reasoning data")
    parser.add_argument("--video_path", default="videos", help = "path to video folder")
    parser.add_argument("--audio_path", default="audios", help = "path to audio folder")
    parser.add_argument("--query", default="", help = "represents the query to be answered")
    parser.add_argument("--max_tries", default=5, help = "max tries to generate each data point")
    parser.add_argument("--score_threshold", default=6, help = "threshold for goodness of reasoning data generated")

    # Read arguments from command line
    args = parser.parse_args()

    num_tries=0

    reason_steps = None

    while num_tries < args.max_tries:
        try:
            if num_tries > 0:
                
                # Generate new reasoning using feedback loop
                reason_steps = feedback(score, reasoning_steps, args.query, args.video_path, args.audio_path)

            # First attempt uses original reasoning
            if num_tries == 0:
                question_text = "Given the video, audio and the question: {} \n Task 1: generate detailed reasoning steps \
                for solving the given question without revealing the answer. \n Task 2: provide detailed answers to each of \
                these above reasoning steps generated in Task 1. \n Task 3: provide a final answer for the question. \n \
                Your output should be in the form of a dictionary which looks like: \
                Task_1: Task 1 answers, Task_2: Task 2 answers, Task_3: Task 3 answers.".format(args.query)
                reason_steps = run_gemini(args.video_path, question_text, args.audio_path)

            # Parse Gemini output into dictionary
            reason_steps_cleaned = ast.literal_eval(
                "{" + re.search(r"{\s*(.+?)\s*}", reason_steps.replace('\n', '')).group(1) + "}"
            )

            reasoning_steps = reason_steps_cleaned['Task_1']
            reasoning_answer = reason_steps_cleaned['Task_2']
            final_answer = reason_steps_cleaned['Task_3']

            # Generate a caption from reasoning
            summarized_response = run_gpt_summarizer(
                reasoning_steps, reasoning_answer, final_answer, args.query
            )

            # Scoring prompt
            feedback_prompt = (
                "Given the video and audio inputs, rate the following caption with a single integer between 1 and 10 "
                "(1 = lowest similarity, 10 = highest similarity), based only on its similarity to the inputs.\n"
                "Respond strictly in the following JSON format:\n"
                '{"score": <number>}\n\n'
                f"Caption: {summarized_response}"
            )

            # Ask Gemini for a score
            feedback_response_raw = run_gemini(args.video_path, feedback_prompt, args.audio_path)

            # Parse JSON score safely
            if isinstance(feedback_response_raw, str):
                feedback_response = json.loads(feedback_response_raw)
            else:
                feedback_response = feedback_response_raw

            score = int(feedback_response.get("score", 0))

            # Check if score meets threshold
            if score >= args.score_threshold:
                print(f"Accepted caption with score: {score}")
                response_data = {
                    "query": args.query,
                    "video_path": args.video_path,
                    "audio_path": args.audio_path,
                    "reasoning_steps": reasoning_steps
                }

                with open(args.save_path, "w") as json_file:
                    json.dump(response_data, json_file, indent=2)
                break
            else:
                print(f"Score {score} < threshold {args.score_threshold}, retrying...")
                num_tries += 1

        except Exception as e:
            print(f"Error on try {num_tries + 1}/{args.max_tries}: {str(e)}")
            num_tries += 1




