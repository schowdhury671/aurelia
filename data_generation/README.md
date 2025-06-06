## Generate Reasoning Data

### Install necessary requirements
```
Python>=3.10
pip install openai==0.28.0
pip install google-generativeai
pip install -q -U pytube moviepy
apt-get install -y ffmpeg
```

Provide api keys for the openAI and Gemini, and add them to the environment variables.
```
export OPENAI_API_KEY=""
export GOOGLE_API_KEY=""
```

### Run the data generation pipeline
```
python gen_data.py --save_path "reason_data.json" \
                   --video_path "sample.mp4" \
                   --audio_path "sample.mp3" \
                   --query "What is the most popular food of the country where the loudest instrument originates from?" \
                   --max_tries 5
```