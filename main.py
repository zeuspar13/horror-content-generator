# BACKUP OF YOUR ORIGINAL WORKING CODE
# DO NOT DELETE THIS FILE
# Date: Current working version before enhancements
# This is your fallback if anything goes wrong

from flask import Flask, request, jsonify
import os
import time
import openai
import requests
import base64

app = Flask(__name__)

# Configuration
SECRET_PASSWORD = "mysecret1303"
openai.api_key = os.getenv('OPENAI_API_KEY')

def check_password():
    """Check if the request has the right password"""
    password = request.headers.get('Authorization', '')
    return password == SECRET_PASSWORD

def log_what_happened():
    """Write down what request we got"""
    endpoint = request.args.get('endpoint', 'home')
    method = request.method
    print(f"Request: {method} /?endpoint={endpoint}")

def generate_real_story(theme, session_id):
    """Generate a real horror story using OpenAI - LONGER VERSION"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a master horror storyteller. Create atmospheric horror stories perfect for 60-90 second videos. Write in clear, dramatic sentences that work well with voice narration and subtitles. Focus on building suspense gradually."
                },
                {
                    "role": "user",
                    "content": f"Create a 60-90 second horror story about a {theme}. Write in short, clear sentences. Build tension gradually. Make it atmospheric and suspenseful. Keep sentences under 15 words each for clear narration."
                }
            ],
            max_tokens=400,  # Increased for longer stories
            temperature=0.8
        )
        
        ai_story = response.choices[0].message.content
        
        story = {
            'title': f'The Haunting of {theme.title()}',
            'text': ai_story,
            'scenes': [
                {'number': 1, 'description': f'Opening scene - {theme}', 'seconds': 20},
                {'number': 2, 'description': 'Building tension', 'seconds': 30},
                {'number': 3, 'description': 'The revelation', 'seconds': 20}
            ],
            'session_id': session_id,
            'ai_generated': True
        }
        
        return story
        
    except Exception as e:
        print(f"OpenAI Story Error: {e}")
        # Longer fallback story for testing
        fallback_story = f"""The old {theme} stands silent in the moonlight. 
        Its windows stare like empty eyes into the darkness. 
        Something moves behind the glass. 
        A shadow that shouldn't be there. 
        The door creaks open on rusted hinges. 
        Footsteps echo through empty halls. 
        Each step brings you closer to the truth. 
        The truth that some places should remain untouched. 
        Some secrets should stay buried. 
        But tonight, the {theme} remembers. 
        And it wants you to remember too."""
        
        return {
            'title': f'The Curse of the {theme.title()}',
            'text': fallback_story,
            'scenes': [
                {'number': 1, 'description': f'Approaching the {theme}', 'seconds': 20},
                {'number': 2, 'description': 'Strange sounds begin', 'second
