import requests
import streamlit as st
import elevenlabs
import os
from dotenv import load_dotenv

IMAGE_TO_TEXT_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
GENERATE_STORY_URL = "https://api-inference.huggingface.co/models/google/gemma-2b-it"
SPEECH_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"

load_dotenv()

acess_token = f"Bearer {os.getenv('HUGGING_FACE_API_TOKEN')}"
headers = {"Authorization": acess_token}

eleven_labs = os.getenv("ELEVEN_LABS_API_TOKEN")


# Make the Image into text
def image2Text(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(IMAGE_TO_TEXT_URL, headers=headers, data=data)
    result = response.json()
    # print(result[0]["generated_text"])
    return result[0]["generated_text"]

# Generate a story from the text generated
def generate_story(scenario):
    template = f"""
    Generate a short story based on a simple narrative,
    the story should be no more than 30 words. 
    
    NARRATIVE: {scenario}
    STORY: 
    """
    
    payload = {"inputs": template}
    
    response = requests.post(GENERATE_STORY_URL, headers=headers, json=payload)
    result = response.json()
    return result[0]["generated_text"].split("STORY:")[1].strip()

# Generate a speech from the text
def text2Speech(text):
    elevenlabs.set_api_key(eleven_labs) 
    audio = elevenlabs.generate(text = text, voice="Gigi")
    elevenlabs.save(audio, "test.mp3")
    return audio
    
def main():
    st.set_page_config(page_title="Image to Audio Story", page_icon="🗣️")
    st.header("Turn Image into Audio Story")
    
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data) 
    st.image(uploaded_file, caption = "Uploaded Image.", use_column_width=True)
    scenario = image2Text(uploaded_file.name)
    story = generate_story(scenario)
    text2Speech(story)
    
    with st.expander("scenario"):
        st.write(scenario)
    with st.expander("story"):
        st.write(story)
        
    st.audio("test.mp3", format="audio/mp3")
    
if __name__ == "__main__":
    main()