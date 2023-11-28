import gradio as gr
import tempfile
import os
import whisper
import openai

openai.api_key = os.getenv('openai_api_key')

# Load Whisper model
whisper_model = whisper.load_model("base")

# Load CSS from a file
with open("/Users/blake/Downloads/style.css", "r") as f:
    custom_css = f.read()
def format_as_study_guide(study_guide, video_file):
    formatted_guide = f"Study Guide for {video_file}\n"
    formatted_guide += "=================================\n\n"
    formatted_guide += study_guide
    return formatted_guide

def write_study_guide_to_file(study_guide, video_file):
    filename = f"{video_file} Study Guide.txt"
    with open(filename, 'w') as file:
        file.write(study_guide)

def transcribe_audio_with_whisper(video_path):
    result = whisper_model.transcribe(video_path)
    return result["text"]

def summarize_text_with_gpt(text):
    # Updated prompt focusing on key points, questions, and answers
    prompt = (
        "Create a comprehensive study guide based on the following transcript. Use only complete sentences. The study guide should follow this format strictly."
        "The study guide should include:\n\n"
        "1. A brief summary of the main content, in complete sentences.\n"
        "2. Detailed key points and important concepts, each expressed as a complete sentence.\n"
        "3. Important questions related to the content, phrased as complete sentences, followed by their corresponding answers.\n\n"
        "Ensure all elements of the study guide are clearly structured. Here's an example of the study guide format:\n"
        "Title: [Title of the Content]\n"
        "Summary: [A brief, clear summary of the main content in complete sentences]\n"
        "Key Points:\n"
        "  - Complete sentence describing Point 1\n"
        "  - Complete sentence describing Point 2\n"
        "  - Complete sentence describing Point 3\n"
        "Important Questions and Answers:\n"
        "  - Question 1: What is the significance of...?\n"
        "    Answer: The significance is...\n"
        "  - Question 2: How does... relate to...?\n"
        "    Answer: It relates by...\n\n"
        "Based on the above format and guidelines, please create a study guide for the following text:\n\n" + text
    )
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=1200  # Increased token limit for detailed guide with Q&A
    )
    study_guide = response.choices[0].text.strip()
    return study_guide

def process_video(video_content, custom_title):
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video_file:
        temp_video_file.write(video_content)
        temp_video_path = temp_video_file.name

    # Transcribe the video
    transcript = transcribe_audio_with_whisper(temp_video_path)

    # Generate the study guide
    study_guide = summarize_text_with_gpt(transcript)

    # Format and save the study guide with the custom title
    formatted_guide = format_as_study_guide(study_guide, custom_title)
    study_guide_filename = write_study_guide_to_file(formatted_guide, custom_title)

    # Clean up temporary video file
    os.remove(temp_video_path)

    return transcript, study_guide, f"Study guide saved as '{custom_title} Study Guide.txt'"

iface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.File(label="Upload Video", type="binary"),
        gr.Textbox(label="Custom Title for Study Guide", placeholder="Enter title here")
    ],
    outputs=[
        gr.Textbox(label="Transcript"),
        gr.Textbox(label="Study Guide"),
        gr.Textbox(label="Study Guide File")
    ],
    css=custom_css  # If you have custom CSS
)

iface.launch()
