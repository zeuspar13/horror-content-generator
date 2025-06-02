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
                    "content": "You are a master horror storyteller. Create short, atmospheric horror stories perfect for 30-second videos. Focus on building dread and atmosphere. Write in short, punchy sentences that work well with voice narration and subtitles."
                },
                {
                    "role": "user",
                    "content": f"Create a 30-second horror story about a {theme}. Write in short, dramatic sentences. Focus on atmosphere and suspense. Keep it under 200 words for clear narration."
                }
            ],
            max_tokens=200,
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

def generate_real_voice_with_timing(text, session_id):
    """Generate real horror voice with subtitle timing using ElevenLabs API"""
    try:
        elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
        if not elevenlabs_api_key:
            raise Exception("ElevenLabs API key not found")
        
        voice_id = "ErXwobaYiN019PkySvjV"  # Antoni voice - deep and dramatic
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"
        
        headers = {
            "Accept": "application/json",
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
            },
            "output_format": "mp3_22050_32"
        }
        
        print(f"ElevenLabs: Generating voice with timing for {len(text)} characters")
        
        response = requests.post(url, json=data, headers=headers, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract audio and timing data
            audio_base64 = result.get('audio_base64', '')
            alignment = result.get('alignment', {})
            
            # Create subtitle timing data
            characters = alignment.get('characters', [])
            character_start_times = alignment.get('character_start_times_seconds', [])
            character_end_times = alignment.get('character_end_times_seconds', [])
            
            # Process word timings for subtitles
            words = text.split()
            subtitle_data = []
            
            if character_start_times and character_end_times:
                char_index = 0
                for word in words:
                    word_start = None
                    word_end = None
                    
                    # Find start time of word
                    while char_index < len(characters) and characters[char_index] != word[0]:
                        char_index += 1
                    if char_index < len(character_start_times):
                        word_start = character_start_times[char_index]
                    
                    # Find end time of word
                    word_char_count = len(word)
                    end_char_index = min(char_index + word_char_count - 1, len(character_end_times) - 1)
                    if end_char_index >= 0:
                        word_end = character_end_times[end_char_index]
                    
                    if word_start is not None and word_end is not None:
                        subtitle_data.append({
                            'word': word,
                            'start': word_start,
                            'end': word_end
                        })
                    
                    char_index += word_char_count + 1  # +1 for space
            else:
                # Fallback: estimate timing based on word count
                total_duration = len(text) / 15  # Rough estimate
                words_per_second = len(words) / total_duration
                for i, word in enumerate(words):
                    start = i / words_per_second
                    end = (i + 1) / words_per_second
                    subtitle_data.append({
                        'word': word,
                        'start': start,
                        'end': end
                    })
            
            print(f"ElevenLabs success: Generated audio with {len(subtitle_data)} timed words")
            
            return {
                'audio_url': f"data:audio/mpeg;base64,{audio_base64}",
                'audio_data': audio_base64,
                'text': text,
                'text_length': len(text),
                'voice_id': voice_id,
                'session_id': session_id,
                'ai_generated': True,
                'duration_estimate': len(text) / 15,
                'subtitle_data': subtitle_data,
                'status': 'Audio with timing generated successfully'
            }
        else:
            raise Exception(f"ElevenLabs API error: {response.status_code} - {response.text}")
        
    except Exception as e:
        print(f"ElevenLabs Error: {e}")
        # Fallback to regular voice generation
        return generate_real_voice_fallback(text, session_id)

def generate_real_voice_fallback(text, session_id):
    """Fallback voice generation without timing"""
    try:
        elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
        voice_id = "ErXwobaYiN019PkySvjV"
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": elevenlabs_api_key
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.6,
                "similarity_boost": 0.8
            }
        }
        
        response = requests.post(url, json=data, headers=headers, timeout=60)
        
        if response.status_code == 200:
            audio_base64 = base64.b64encode(response.content).decode('utf-8')
            
            # Create estimated subtitle timing
            words = text.split()
            total_duration = len(text) / 15
            words_per_second = len(words) / total_duration if total_duration > 0 else 2
            
            subtitle_data = []
            for i, word in enumerate(words):
                start = i / words_per_second
                end = (i + 1) / words_per_second
                subtitle_data.append({
                    'word': word,
                    'start': start,
                    'end': end
                })
            
            return {
                'audio_url': f"data:audio/mpeg;base64,{audio_base64}",
                'audio_data': audio_base64,
                'text': text,
                'text_length': len(text),
                'session_id': session_id,
                'ai_generated': True,
                'subtitle_data': subtitle_data,
                'status': 'Fallback audio with estimated timing'
            }
        else:
            raise Exception(f"Fallback API error: {response.status_code}")
            
    except Exception as e:
        print(f"Fallback Error: {e}")
        return {
            'audio_url': f'/audio/{session_id}_voice.mp3',
            'text': text,
            'text_length': len(text),
            'session_id': session_id,
            'ai_generated': False,
            'error': str(e)
        }

def create_real_video_with_subtitles(session_id, image_url, audio_data, subtitle_data):
    """Create real video with synced subtitles using MoviePy"""
    try:
        from moviepy.editor import ImageClip, AudioFileClip, TextClip, CompositeVideoClip
        import tempfile
        
        print(f"Video creation starting for session {session_id}")
        print(f"Image URL: {image_url}")
        print(f"Audio data length: {len(audio_data) if audio_data else 0}")
        print(f"Subtitle words: {len(subtitle_data) if subtitle_data else 0}")
        
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
        
        # Create video clip with Ken Burns effect (slow zoom)
        print("Creating video clip with effects...")
        image_clip = ImageClip(img_path, duration=30)
        
        # Add Ken Burns effect (slow zoom in)
        image_clip = image_clip.resize(lambda t: 1 + 0.02*t).set_position(('center','center'))
        
        try:
            print("Processing audio...")
            audio_clip = AudioFileClip(audio_path)
            
            # Match video duration to audio duration (max 30 seconds)
            video_duration = min(audio_clip.duration, 30)
            image_clip = image_clip.set_duration(video_duration)
            
            print(f"Video duration: {video_duration} seconds")
            
            # Create subtitle clips
            subtitle_clips = []
            if subtitle_data:
                print("Creating subtitle clips...")
                
                # Group words into phrases for better readability
                phrases = []
                current_phrase = []
                current_start = None
                
                for word_data in subtitle_data:
                    if not current_phrase:
                        current_start = word_data['start']
                    
                    current_phrase.append(word_data['word'])
                    
                    # Create phrase every 3-4 words or at natural breaks
                    if (len(current_phrase) >= 4 or 
                        word_data['word'].endswith('.') or 
                        word_data['word'].endswith('!') or 
                        word_data['word'].endswith('?')):
                        
                        phrase_text = ' '.join(current_phrase)
                        phrases.append({
                            'text': phrase_text,
                            'start': current_start,
                            'end': word_data['end']
                        })
                        current_phrase = []
                
                # Add remaining words as final phrase
                if current_phrase:
                    phrase_text = ' '.join(current_phrase)
                    phrases.append({
                        'text': phrase_text,
                        'start': current_start,
                        'end': subtitle_data[-1]['end']
                    })
                
                # Create subtitle clips with horror styling
                for phrase in phrases:
                    if phrase['start'] < video_duration:
                        duration = min(phrase['end'] - phrase['start'], video_duration - phrase['start'])
                        
                        if duration > 0:
                            subtitle_clip = TextClip(
                                phrase['text'],
                                fontsize=24,
                                color='white',
                                font='Arial-Bold',
                                stroke_color='black',
                                stroke_width=2
                            ).set_position(('center', 'bottom')).set_start(phrase['start']).set_duration(duration)
                            
                            # Add fade in/out effects
                            subtitle_clip = subtitle_clip.crossfadein(0.2).crossfadeout(0.2)
                            subtitle_clips.append(subtitle_clip)
                
                print(f"Created {len(subtitle_clips)} subtitle clips")
            
            # Combine image, audio, and subtitles
            all_clips = [image_clip.set_audio(audio_clip)] + subtitle_clips
            final_clip = CompositeVideoClip(all_clips)
            
        except Exception as audio_error:
            print(f"Audio/subtitle processing error: {audio_error}")
            # If audio fails, create silent video with basic subtitles
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
            'subtitle_count': len(subtitle_clips) if 'subtitle_clips' in locals() else 0,
            'status': 'Video with subtitles created successfully',
            'size_bytes': len(video_base64) * 3 / 4,
            'features': ['ken_burns_effect', 'synced_subtitles', 'fade_effects']
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
            'version': '2.5 - AI PIPELINE WITH SUBTITLES',
            'time': time.time()
        })
    
    # Test page
    elif endpoint == 'test':
        return jsonify({
            'message': 'Test page works!',
            'endpoint': endpoint,
            'openai_configured': bool(os.getenv('OPENAI_API_KEY')),
            'elevenlabs_configured': bool(os.getenv('ELEVENLABS_API_KEY')),
            'moviepy_available': True,
            'subtitle_support': True,
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
    
    # Create voice with timing (AI-powered)
    elif endpoint == 'create-voice':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        text = data.get('text', '')
        
        voice_result = generate_real_voice_with_timing(text, session_id)
        
        return jsonify({
            'success': True,
            **voice_result
        })
    
    # Create video with subtitles (AI-powered)
    elif endpoint == 'create-video':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        image_url = data.get('image_url', '')
        audio_data = data.get('audio_data', '')
        subtitle_data = data.get('subtitle_data', [])
        
        video_result = create_real_video_with_subtitles(session_id, image_url, audio_data, subtitle_data)
        
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
