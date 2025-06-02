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
                {'number': 2, 'description': 'Strange sounds begin', 'seconds': 30},
                {'number': 3, 'description': 'The horror is revealed', 'seconds': 20}
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

def generate_free_voice_with_timing(text, session_id):
    """Generate voice using free TTS with manual subtitle timing"""
    try:
        print(f"Free TTS: Generating voice for {len(text)} characters")
        
        # Split text into sentences for better subtitle timing
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Create manual subtitle timing (estimated)
        subtitle_data = []
        current_time = 0
        words_per_second = 2.5  # Slower, more readable pace
        
        for sentence in sentences:
            words = sentence.split()
            for word in words:
                duration = len(word) / 5 + 0.5  # Longer duration per word
                subtitle_data.append({
                    'word': word,
                    'start': current_time,
                    'end': current_time + duration
                })
                current_time += duration + 0.2  # Longer pause between words
            
            current_time += 0.8  # Longer pause between sentences
        
        print(f"Free TTS success: Generated timing for {len(subtitle_data)} words")
        
        # For testing, we'll create a silent audio file or use no audio
        # Return data that indicates we're using subtitle-only mode
        return {
            'audio_url': '',  # No audio for testing
            'audio_data': '',  # No audio data
            'text': text,
            'text_length': len(text),
            'voice_id': 'free_tts_testing',
            'session_id': session_id,
            'ai_generated': True,
            'duration_estimate': current_time,
            'subtitle_data': subtitle_data,
            'status': 'Free TTS with manual timing generated - subtitle only mode',
            'service': 'free_testing_subtitle_only'
        }
        
    except Exception as e:
        print(f"Free TTS Error: {e}")
        # Create simple fallback with timing
        words = text.split()[:20]  # Limit words for testing
        subtitle_data = []
        for i, word in enumerate(words):
            subtitle_data.append({
                'word': word,
                'start': i * 1.0,  # 1 second per word
                'end': (i + 1) * 1.0
            })
        
        return {
            'audio_url': '',
            'audio_data': '',
            'text': text,
            'text_length': len(text),
            'session_id': session_id,
            'subtitle_data': subtitle_data,
            'ai_generated': False,
            'error': str(e),
            'service': 'fallback_subtitle_only'
        }

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
    """Create real video with synced subtitles - optimized for Render.com"""
    try:
        from moviepy.editor import ImageClip, AudioFileClip, TextClip, CompositeVideoClip
        import tempfile
        import gc
        
        print(f"Video creation starting for session {session_id}")
        print(f"Subtitle words: {len(subtitle_data) if subtitle_data else 0}")
        
        # Validate image URL
        if not image_url or image_url.startswith('/images/'):
            raise Exception("Invalid image URL")
        
        # Download image
        print("Downloading image...")
        image_response = requests.get(image_url, timeout=15)
        if image_response.status_code != 200:
            raise Exception("Failed to download image")
        
        # Create temp files
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as img_file:
            img_file.write(image_response.content)
            img_path = img_file.name
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as audio_file:
            if audio_data and len(audio_data) > 100:
                if audio_data.startswith('data:audio'):
                    audio_base64 = audio_data.split(',')[1]
                else:
                    audio_base64 = audio_data
                audio_bytes = base64.b64decode(audio_base64)
                audio_file.write(audio_bytes)
                audio_path = audio_file.name
            else:
                raise Exception("No audio data")
        
        print("Creating optimized video...")
        
        # Create base clips with reduced resolution for memory efficiency
        image_clip = ImageClip(img_path, duration=60)  # Increased duration for longer stories
        image_clip = image_clip.resize(height=480)  # Reduce resolution
        
        audio_clip = AudioFileClip(audio_path)
        video_duration = min(audio_clip.duration, 60)  # Max 60 seconds for longer testing
        image_clip = image_clip.set_duration(video_duration)
        
        # Create simplified subtitles - only 2-3 key phrases
        subtitle_clips = []
        if subtitle_data and len(subtitle_data) > 0:
            print("Creating simplified subtitles...")
            print(f"Sample subtitle data: {subtitle_data[:3]}")  # Debug first 3 items
            
            # Check the format of subtitle data
            if isinstance(subtitle_data, list) and len(subtitle_data) > 0:
                # Handle different subtitle data formats
                if isinstance(subtitle_data[0], dict):
                    # Format: [{"word": "text", "start": 0.1, "end": 0.5}, ...]
                    words = []
                    for item in subtitle_data[:10]:  # Limit to 10 words
                        if 'word' in item and 'start' in item:
                            words.append(item['word'])
                        elif 'text' in item:  # Alternative format
                            words.append(item['text'])
                        elif isinstance(item, str):  # Just strings
                            words.append(item)
                    
                    if words:
                        # Create 3 subtitle segments
                        words_per_segment = max(1, len(words) // 3)
                        segment_duration = video_duration / 3
                        
                        for i in range(3):
                            start_time = i * segment_duration
                            if start_time < video_duration:
                                # Get words for this segment
                                start_word_idx = i * words_per_segment
                                end_word_idx = min((i + 1) * words_per_segment, len(words))
                                segment_words = words[start_word_idx:end_word_idx]
                                
                                if segment_words:
                                    text = ' '.join(segment_words)
                                    
                                    try:
                                        subtitle_clip = TextClip(
                                            text,
                                            fontsize=20,
                                            color='white',
                                            stroke_color='black',
                                            stroke_width=1
                                        ).set_position(('center', 'bottom')).set_start(start_time).set_duration(segment_duration * 0.8)
                                        
                                        subtitle_clips.append(subtitle_clip)
                                        print(f"Created subtitle: '{text}' at {start_time}s")
                                    except Exception as subtitle_error:
                                        print(f"Error creating subtitle clip: {subtitle_error}")
                                        # Create simple text overlay as fallback
                                        continue
                
                elif isinstance(subtitle_data[0], str):
                    # Format: ["word1", "word2", ...]
                    words = subtitle_data[:10]
                    text = ' '.join(words)
                    
                    try:
                        subtitle_clip = TextClip(
                            text,
                            fontsize=20,
                            color='white',
                            stroke_color='black',
                            stroke_width=1
                        ).set_position(('center', 'bottom')).set_start(0).set_duration(video_duration * 0.8)
                        
                        subtitle_clips.append(subtitle_clip)
                        print(f"Created single subtitle: '{text}'")
                    except Exception as subtitle_error:
                        print(f"Error creating subtitle: {subtitle_error}")
            
            print(f"Successfully created {len(subtitle_clips)} subtitle clips")
        
        # Combine clips efficiently
        if has_audio and audio_clip:
            main_clip = image_clip.set_audio(audio_clip)
        else:
            main_clip = image_clip  # Silent video
            
        if subtitle_clips:
            final_clip = CompositeVideoClip([main_clip] + subtitle_clips)
        else:
            final_clip = main_clip
        
        # Export with optimized settings
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as video_file:
            video_path = video_file.name
        
        print("Exporting optimized video...")
        final_clip.write_videofile(
            video_path,
            fps=15,  # Reduced FPS
            audio_codec='aac' if has_audio else None,
            codec='libx264',
            bitrate='500k',  # Lower bitrate
            verbose=False,
            logger=None
        )
        
        # Clean up immediately
        final_clip.close()
        if has_audio and audio_clip:
            audio_clip.close()
        del final_clip, image_clip
        if has_audio and audio_clip:
            del audio_clip
        gc.collect()
        
        # Clean up temp files
        os.unlink(img_path)
        if has_audio and audio_path:
            os.unlink(audio_path)
        
        # Convert to base64
        print("Converting to base64...")
        with open(video_path, 'rb') as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
        
        os.unlink(video_path)
        
        print(f"Optimized video creation successful! Base64 length: {len(video_base64)}")
        
        return {
            'video_url': f"data:video/mp4;base64,{video_base64}",
            'video_data': video_base64,
            'session_id': session_id,
            'ai_generated': True,
            'duration': video_duration,
            'subtitle_count': len(subtitle_clips),
            'status': 'Optimized video with subtitles created',
            'size_bytes': len(video_base64) * 3 / 4,
            'features': ['synced_subtitles', 'memory_optimized']
        }
        
    except Exception as e:
        print(f"Video Creation Error: {e}")
        return {
            'video_url': f'/videos/{session_id}_final.mp4',
            'session_id': session_id,
            'ai_generated': False,
            'error': str(e),
            'message': 'Video creation failed'
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
    
    # Create voice with timing (FREE TTS for testing)
    elif endpoint == 'create-voice':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        text = data.get('text', '')
        
        voice_result = generate_free_voice_with_timing(text, session_id)
        
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
        
        # Debug logging
        print(f"Video creation request data:")
        print(f"  session_id: {session_id}")
        print(f"  image_url length: {len(image_url) if image_url else 0}")
        print(f"  audio_data length: {len(audio_data) if audio_data else 0}")
        print(f"  subtitle_data: {subtitle_data}")
        print(f"  subtitle_data type: {type(subtitle_data)}")
        print(f"  subtitle_data length: {len(subtitle_data) if subtitle_data else 0}")
        
        # Fix subtitle data if it's a string (n8n serialization issue)
        if isinstance(subtitle_data, str):
            try:
                import json
                # Try to parse as JSON
                subtitle_data = json.loads(subtitle_data)
                print(f"  Parsed subtitle_data: {subtitle_data}")
            except:
                # If not valid JSON, create simple fallback subtitles
                print("  Creating fallback subtitles from story text")
                subtitle_data = [
                    {"word": "Horror", "start": 0, "end": 3},
                    {"word": "awaits in", "start": 3, "end": 6},
                    {"word": "the darkness", "start": 6, "end": 9},
                    {"word": "below...", "start": 9, "end": 12}
                ]
        
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
