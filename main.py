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
    """Generate a real horror story using OpenAI"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a master horror storyteller. Create short, atmospheric horror stories perfect for 30-second videos. Focus on building dread and atmosphere."
                },
                {
                    "role": "user",
                    "content": f"Create a 30-second horror story about a {theme}. Include a title and divide it into 3 scenes (8, 12, and 10 seconds each). Make it atmospheric and creepy, not gory."
                }
            ],
            max_tokens=300,
            temperature=0.8
        )
        
        ai_story = response.choices[0].message.content
        
        story = {
            'title': f'The Haunting of {theme.title()}',
            'text': ai_story,
            'scenes': [
                {'number': 1, 'description': f'Opening scene - {theme}', 'seconds': 8},
                {'number': 2, 'description': 'Building tension', 'seconds': 12},
                {'number': 3, 'description': 'The revelation', 'seconds': 10}
            ],
            'session_id': session_id,
            'ai_generated': True
        }
        
        return story
        
    except Exception as e:
        print(f"OpenAI Story Error: {e}")
        return {
            'title': f'The Curse of the {theme.title()}',
            'text': f'Deep in the shadows of the {theme}, ancient evil stirs...',
            'scenes': [
                {'number': 1, 'description': f'Approaching the {theme}', 'seconds': 8},
                {'number': 2, 'description': 'Strange sounds begin', 'seconds': 12},
                {'number': 3, 'description': 'The horror is revealed', 'seconds': 10}
            ],
            'session_id': session_id,
            'ai_generated': False,
            'error': str(e)
        }

def generate_real_image(description, session_id):
    """Generate a real horror image using DALL-E"""
    try:
        # Create a concise but atmospheric prompt
        horror_prompt = f"Dark atmospheric horror scene: {description}. Cinematic lighting, shadows, eerie mood, high quality digital art"
        
        # Ensure prompt is under DALL-E limit (1000 characters)
        if len(horror_prompt) > 900:
            horror_prompt = f"Dark horror scene: {description[:500]}. Cinematic lighting, shadows, eerie mood"
        
        print(f"DALL-E prompt: {horror_prompt}")
        
        response = openai.Image.create(
            prompt=horror_prompt,
            n=1,
            size="1024x1024",
            response_format="url"
        )
        
        image_url = response['data'][0]['url']
        print(f"DALL-E success: {image_url}")
        
        return {
            'image_url': image_url,
            'description': description,
            'prompt_used': horror_prompt,
            'session_id': session_id,
            'ai_generated': True
        }
        
    except Exception as e:
        print(f"DALL-E Error: {e}")
        print(f"Error type: {type(e)}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response}")
        return {
            'image_url': f'/images/{session_id}_horror.jpg',
            'description': description,
            'session_id': session_id,
            'ai_generated': False,
            'error': str(e)
        }

def generate_real_voice(text, session_id):
    """Generate real horror voice using ElevenLabs API"""
    try:
        elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
        if not elevenlabs_api_key:
            raise Exception("ElevenLabs API key not found")
        
        voice_id = "ErXwobaYiN019PkySvjV"  # Antoni voice - deep and dramatic
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": elevenlabs_api_key
        }
        
        # Limit text length for voice generation
        if len(text) > 2500:
            text = text[:2500] + "..."
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.6,
                "similarity_boost": 0.8,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        print(f"ElevenLabs: Generating voice for {len(text)} characters")
        
        response = requests.post(url, json=data, headers=headers, timeout=60)
        
        if response.status_code == 200:
            # Convert audio to base64 for storage/transfer
            audio_base64 = base64.b64encode(response.content).decode('utf-8')
            
            print(f"ElevenLabs success: Generated {len(audio_base64)} chars of audio data")
            
            return {
                'audio_url': f"data:audio/mpeg;base64,{audio_base64}",
                'audio_data': audio_base64,
                'text': text,
                'text_length': len(text),
                'voice_id': voice_id,
                'session_id': session_id,
                'ai_generated': True,
                'duration_estimate': len(text) / 15,
                'status': 'Audio generated successfully'
            }
        else:
            raise Exception(f"ElevenLabs API error: {response.status_code} - {response.text}")
        
    except Exception as e:
        print(f"ElevenLabs Error: {e}")
        return {
            'audio_url': f'/audio/{session_id}_voice.mp3',
            'text': text,
            'text_length': len(text),
            'session_id': session_id,
            'ai_generated': False,
            'error': str(e)
        }

def create_real_video(session_id, image_url, audio_data):
    """Create real video using MoviePy"""
    try:
        from moviepy.editor import ImageClip, AudioFileClip
        import tempfile
        
        print(f"Video creation starting for session {session_id}")
        print(f"Image URL: {image_url}")
        print(f"Audio data length: {len(audio_data) if audio_data else 0}")
        
        # Validate image URL
        if not image_url or image_url.startswith('/images/'):
            raise Exception("Invalid image URL - AI image generation failed")
        
        # Download the image
        print("Downloading image...")
        image_response = requests.get(image_url, timeout=30)
        if image_response.status_code != 200:
            raise Exception(f"Failed to download image: {image_response.status_code}")
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as img_file:
            img_file.write(image_response.content)
            img_path = img_file.name
        
        print(f"Image saved to: {img_path}")
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as audio_file:
            if audio_data and len(audio_data) > 100:
                # Handle base64 audio (either with or without data: prefix)
                if audio_data.startswith('data:audio'):
                    audio_base64 = audio_data.split(',')[1]
                else:
                    audio_base64 = audio_data
                
                audio_bytes = base64.b64decode(audio_base64)
                audio_file.write(audio_bytes)
                audio_path = audio_file.name
                print(f"Audio saved to: {audio_path}")
            else:
                raise Exception("No valid audio data received")
        
        # Create video clip
        print("Creating video clip...")
        image_clip = ImageClip(img_path, duration=30)
        
        try:
            print("Processing audio...")
            audio_clip = AudioFileClip(audio_path)
            
            # Match video duration to audio duration (max 30 seconds)
            video_duration = min(audio_clip.duration, 30)
            image_clip = image_clip.set_duration(video_duration)
            
            print(f"Video duration: {video_duration} seconds")
            
            # Combine image and audio
            final_clip = image_clip.set_audio(audio_clip)
            
        except Exception as audio_error:
            print(f"Audio processing error: {audio_error}")
            # If audio fails, create silent video
            final_clip = image_clip.set_duration(30)
            video_duration = 30
        
        # Export video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as video_file:
            video_path = video_file.name
        
        print(f"Exporting video to: {video_path}")
        
        final_clip.write_videofile(
            video_path,
            fps=24,
            audio_codec='aac',
            codec='libx264',
            verbose=False,
            logger=None,
            temp_audiofile='temp-audio.m4a',
            remove_temp=True
        )
        
        print("Video export complete")
        
        # Clean up clips
        final_clip.close()
        if 'audio_clip' in locals():
            audio_clip.close()
        
        # Clean up temp files
        os.unlink(img_path)
        os.unlink(audio_path)
        
        # Convert video to base64
        print("Converting video to base64...")
        with open(video_path, 'rb') as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
        
        os.unlink(video_path)
        
        print(f"Video creation successful! Base64 length: {len(video_base64)}")
        
        return {
            'video_url': f"data:video/mp4;base64,{video_base64}",
            'video_data': video_base64,
            'session_id': session_id,
            'ai_generated': True,
            'duration': video_duration if 'video_duration' in locals() else 30,
            'status': 'Video created successfully',
            'size_bytes': len(video_base64) * 3 / 4  # Approximate file size
        }
        
    except Exception as e:
        print(f"Video Creation Error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return {
            'video_url': f'/videos/{session_id}_final.mp4',
            'session_id': session_id,
            'ai_generated': False,
            'error': str(e),
            'message': 'Video creation failed, returned mock URL'
        }

@app.route('/', methods=['GET', 'POST'])
def handle_everything():
    """This handles ALL requests to our website"""
    log_what_happened()
    endpoint = request.args.get('endpoint', 'home')
    
    # Home page
    if endpoint == 'home':
        return jsonify({
            'message': 'Horror Pipeline is Running!',
            'status': 'working',
            'version': '2.4 - COMPLETE AI PIPELINE',
            'time': time.time()
        })
    
    # Test page
    elif endpoint == 'test':
        return jsonify({
            'message': 'Test page works!',
            'endpoint': endpoint,
            'openai_configured': bool(os.getenv('OPENAI_API_KEY')),
            'elevenlabs_configured': bool(os.getenv('ELEVENLABS_API_KEY')),
            'moviepy_available': True,  # MoviePy is installed
            'time': time.time()
        })
    
    # Password check for other endpoints
    elif not check_password():
        return jsonify({
            'error': 'Wrong password!',
            'message': 'You need: Authorization: mysecret1303'
        }), 401
    
    # Create session
    elif endpoint == 'create-session':
        session_id = f"horror_{int(time.time())}"
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'New session created!'
        })
    
    # Create story (AI-powered)
    elif endpoint == 'create-story':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        theme = data.get('theme', 'haunted house')
        
        story = generate_real_story(theme, session_id)
        
        return jsonify({
            'success': True,
            'story': story
        })
    
    # Create image (AI-powered)
    elif endpoint == 'create-image':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        description = data.get('description', 'scary horror scene')
        
        image_result = generate_real_image(description, session_id)
        
        return jsonify({
            'success': True,
            **image_result
        })
    
    # Create voice (AI-powered)
    elif endpoint == 'create-voice':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        text = data.get('text', '')
        
        voice_result = generate_real_voice(text, session_id)
        
        return jsonify({
            'success': True,
            **voice_result
        })
    
    # Create video (AI-powered)
    elif endpoint == 'create-video':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        image_url = data.get('image_url', '')
        audio_data = data.get('audio_data', '')
        
        video_result = create_real_video(session_id, image_url, audio_data)
        
        return jsonify({
            'success': True,
            **video_result
        })
    
    # Unknown endpoint
    else:
        return jsonify({
            'error': 'Page not found',
            'endpoint': endpoint
        }), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
