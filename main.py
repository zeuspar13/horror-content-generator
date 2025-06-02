from flask import Flask, request, jsonify
import os
import time
import openai
import requests
import base64
import json
import tempfile
import gc
import whisper
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter
from pydub.generators import Sine, WhiteNoise

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
    """Generate a real horror story using OpenAI - ENHANCED VERSION"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a master horror storyteller. Create atmospheric horror stories perfect for 60-90 second videos. Write in clear, dramatic sentences that work well with voice narration and subtitles. Focus on building suspense gradually. Use short sentences under 12 words each."
                },
                {
                    "role": "user",
                    "content": f"Create a 60-90 second horror story about a {theme}. Write in very short, punchy sentences. Each sentence should be under 12 words. Build tension gradually. Make it cinematic and atmospheric. Focus on visual descriptions that work well with AI-generated images."
                }
            ],
            max_tokens=450,
            temperature=0.8
        )
        
        ai_story = response.choices[0].message.content
        
        # Split story into sentences for better subtitle timing
        sentences = [s.strip() for s in ai_story.split('.') if s.strip()]
        
        story = {
            'title': f'The Haunting of {theme.title()}',
            'text': ai_story,
            'sentences': sentences,
            'scenes': [
                {'number': 1, 'description': f'Opening scene - {theme}', 'seconds': 25},
                {'number': 2, 'description': 'Building tension', 'seconds': 35},
                {'number': 3, 'description': 'The climax', 'seconds': 20}
            ],
            'session_id': session_id,
            'ai_generated': True,
            'word_count': len(ai_story.split())
        }
        
        return story
        
    except Exception as e:
        print(f"OpenAI Story Error: {e}")
        fallback_story = f"""The old {theme} stands in darkness. Windows stare like dead eyes. Something moves inside. A shadow that shouldn't exist. The door opens slowly. Footsteps echo in empty halls. Each step reveals the truth. Some places remember everything. Tonight the {theme} awakens. And it remembers you."""
        
        return {
            'title': f'The Curse of the {theme.title()}',
            'text': fallback_story,
            'sentences': fallback_story.split('.'),
            'scenes': [
                {'number': 1, 'description': f'Approaching the {theme}', 'seconds': 25},
                {'number': 2, 'description': 'Strange sounds begin', 'seconds': 35},
                {'number': 3, 'description': 'The horror revealed', 'seconds': 20}
            ],
            'session_id': session_id,
            'ai_generated': False,
            'error': str(e)
        }

def generate_real_image(description, session_id):
    """Generate a real horror image using DALL-E - ENHANCED"""
    try:
        # Enhanced horror prompt with cinematic elements
        horror_prompt = f"Cinematic horror scene: {description}. Dark atmospheric lighting, dramatic shadows, eerie fog, high contrast, professional cinematography, 4K quality, unsettling mood"
        
        if len(horror_prompt) > 900:
            horror_prompt = f"Dark cinematic horror: {description[:400]}. Dramatic lighting, shadows, fog, unsettling atmosphere"
        
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
        return {
            'image_url': f'/images/{session_id}_horror.jpg',
            'description': description,
            'session_id': session_id,
            'ai_generated': False,
            'error': str(e)
        }

def create_horror_soundscape(duration_seconds, session_id):
    """Create atmospheric horror background sounds"""
    try:
        print(f"Creating horror soundscape for {duration_seconds} seconds")
        
        # Create base ambient sound
        silence = AudioSegment.silent(duration=duration_seconds * 1000)
        
        # Generate horror elements
        # 1. Low frequency rumble
        rumble_freq = 40  # Very low frequency
        rumble = Sine(rumble_freq).to_audio_segment(duration=duration_seconds * 1000)
        rumble = rumble - 25  # Make it quieter
        
        # 2. Occasional whispers/wind (white noise filtered)
        wind_segments = []
        for i in range(0, int(duration_seconds), 8):  # Every 8 seconds
            wind_duration = min(3000, (duration_seconds - i) * 1000)  # 3 seconds max
            wind = WhiteNoise().to_audio_segment(duration=wind_duration)
            wind = low_pass_filter(wind, 200)  # Make it sound like wind
            wind = wind - 30  # Make it subtle
            wind_segments.append((wind, i * 1000))
        
        # 3. Heartbeat effect (low sine wave pulses)
        heartbeat_segments = []
        for beat_time in range(5, int(duration_seconds), 12):  # Every 12 seconds starting at 5s
            beat = Sine(60).to_audio_segment(duration=200)  # 200ms beat
            beat = beat.fade_in(50).fade_out(50) - 20
            heartbeat_segments.append((beat, beat_time * 1000))
        
        # Combine all elements
        soundscape = silence.overlay(rumble)
        
        for wind, start_time in wind_segments:
            soundscape = soundscape.overlay(wind, position=start_time)
        
        for beat, start_time in heartbeat_segments:
            soundscape = soundscape.overlay(beat, position=start_time)
        
        # Normalize and apply effects
        soundscape = normalize(soundscape)
        soundscape = soundscape - 15  # Keep it subtle
        
        # Export to base64
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            soundscape.export(temp_file.name, format='mp3', bitrate='128k')
            
            with open(temp_file.name, 'rb') as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
            
            os.unlink(temp_file.name)
        
        print(f"Horror soundscape created successfully")
        
        return {
            'soundscape_data': audio_base64,
            'duration': duration_seconds,
            'effects': ['rumble', 'wind', 'heartbeat'],
            'session_id': session_id
        }
        
    except Exception as e:
        print(f"Soundscape creation error: {e}")
        return {
            'soundscape_data': '',
            'error': str(e),
            'session_id': session_id
        }

def generate_voice_with_whisper_timing(text, session_id):
    """Generate voice with Whisper-based subtitle timing"""
    try:
        print(f"Generating voice with Whisper timing for {len(text)} characters")
        
        # First, generate audio with ElevenLabs
        elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
        if not elevenlabs_api_key:
            raise Exception("ElevenLabs API key not found")
            
        voice_id = "ErXwobaYiN019PkySvjV"  # Dramatic horror voice
        
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
                "stability": 0.7,
                "similarity_boost": 0.8,
                "style": 0.4,
                "use_speaker_boost": True
            }
        }
        
        print("Generating audio with ElevenLabs...")
        response = requests.post(url, json=data, headers=headers, timeout=60)
        
        if response.status_code != 200:
            raise Exception(f"ElevenLabs API error: {response.status_code}")
        
        # Save audio to temp file for Whisper processing
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
            temp_audio.write(response.content)
            audio_path = temp_audio.name
        
        print("Processing audio with Whisper for precise timing...")
        
        # Load Whisper model (using tiny model for Render.com memory limits)
        model = whisper.load_model("tiny")
        
        # Transcribe with word-level timestamps
        result = model.transcribe(
            audio_path, 
            word_timestamps=True,
            language='en'
        )
        
        # Extract word-level timing data
        subtitle_data = []
        if 'segments' in result:
            for segment in result['segments']:
                if 'words' in segment:
                    for word_info in segment['words']:
                        subtitle_data.append({
                            'word': word_info['word'].strip(),
                            'start': word_info['start'],
                            'end': word_info['end'],
                            'confidence': word_info.get('probability', 1.0)
                        })
        
        # Convert audio to base64
        audio_base64 = base64.b64encode(response.content).decode('utf-8')
        
        # Get audio duration
        audio_segment = AudioSegment.from_mp3(audio_path)
        duration = len(audio_segment) / 1000.0  # Convert to seconds
        
        # Clean up temp file
        os.unlink(audio_path)
        
        print(f"Whisper processing complete: {len(subtitle_data)} words with timestamps")
        
        return {
            'audio_url': f"data:audio/mpeg;base64,{audio_base64}",
            'audio_data': audio_base64,
            'text': text,
            'text_length': len(text),
            'voice_id': voice_id,
            'session_id': session_id,
            'ai_generated': True,
            'duration': duration,
            'subtitle_data': subtitle_data,
            'word_count': len(subtitle_data),
            'status': 'ElevenLabs + Whisper timing generated',
            'service': 'elevenlabs_whisper'
        }
        
    except Exception as e:
        print(f"Voice + Whisper Error: {e}")
        # Fallback to Google TTS with estimated timing
        return generate_free_voice_with_timing(text, session_id)

def generate_free_voice_with_timing(text, session_id):
    """Fallback: Generate voice using Google TTS with manual timing"""
    try:
        print(f"Google TTS fallback: Generating voice for {len(text)} characters")
        
        from gtts import gTTS
        
        # Create TTS audio
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
            tts.save(temp_audio.name)
            temp_audio_path = temp_audio.name
        
        # Read the audio file and convert to base64
        with open(temp_audio_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Get duration using pydub
        audio_segment = AudioSegment.from_mp3(temp_audio_path)
        duration = len(audio_segment) / 1000.0
        
        # Clean up temp file
        os.unlink(temp_audio_path)
        
        # Create manual subtitle timing
        words = text.split()
        subtitle_data = []
        words_per_second = len(words) / duration if duration > 0 else 2
        
        for i, word in enumerate(words):
            start_time = i / words_per_second
            end_time = (i + 1) / words_per_second
            
            subtitle_data.append({
                'word': word,
                'start': start_time,
                'end': min(end_time, duration),
                'confidence': 0.8
            })
        
        print(f"Google TTS fallback complete: {len(subtitle_data)} words")
        
        return {
            'audio_url': f"data:audio/mpeg;base64,{audio_base64}",
            'audio_data': audio_base64,
            'text': text,
            'text_length': len(text),
            'voice_id': 'google_tts_free',
            'session_id': session_id,
            'ai_generated': True,
            'duration': duration,
            'subtitle_data': subtitle_data,
            'word_count': len(subtitle_data),
            'status': 'Google TTS with estimated timing',
            'service': 'google_tts_fallback'
        }
        
    except Exception as e:
        print(f"Google TTS Error: {e}")
        return {
            'audio_url': '',
            'audio_data': '',
            'text': text,
            'session_id': session_id,
            'subtitle_data': [],
            'ai_generated': False,
            'error': str(e),
            'service': 'failed'
        }

def create_enhanced_video_with_effects(session_id, image_url, audio_data, subtitle_data, soundscape_data=None):
    """Create enhanced video with professional subtitles and effects"""
    try:
        from moviepy.editor import (
            ImageClip, AudioFileClip, TextClip, CompositeVideoClip, 
            concatenate_audioclips, CompositeAudioClip
        )
        
        print(f"Enhanced video creation starting for session {session_id}")
        print(f"Subtitle words: {len(subtitle_data) if subtitle_data else 0}")
        
        # Validate inputs
        if not image_url or image_url.startswith('/images/'):
            raise Exception("Invalid image URL")
        
        if not audio_data or len(audio_data) < 100:
            raise Exception("No valid audio data")
        
        # Download and prepare image
        print("Downloading image...")
        image_response = requests.get(image_url, timeout=15)
        if image_response.status_code != 200:
            raise Exception("Failed to download image")
        
        # Create temp files
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as img_file:
            img_file.write(image_response.content)
            img_path = img_file.name
        
        # Prepare audio
        if audio_data.startswith('data:audio'):
            audio_base64 = audio_data.split(',')[1]
        else:
            audio_base64 = audio_data
        
        audio_bytes = base64.b64decode(audio_base64)
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as audio_file:
            audio_file.write(audio_bytes)
            audio_path = audio_file.name
        
        print("Creating enhanced video components...")
        
        # Create main clips
        audio_clip = AudioFileClip(audio_path)
        video_duration = min(audio_clip.duration, 90)  # Max 90 seconds
        
        # Create image clip with Ken Burns effect (slow zoom)
        image_clip = ImageClip(img_path, duration=video_duration)
        image_clip = image_clip.resize(height=720)  # HD quality
        
        # Apply Ken Burns effect (slow zoom)
        image_clip = image_clip.resize(lambda t: 1 + 0.02 * t)  # Slow zoom in
        
        # Prepare audio with soundscape
        final_audio = audio_clip
        
        if soundscape_data and len(soundscape_data) > 100:
            try:
                # Add horror soundscape
                soundscape_bytes = base64.b64decode(soundscape_data)
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as soundscape_file:
                    soundscape_file.write(soundscape_bytes)
                    soundscape_path = soundscape_file.name
                
                soundscape_clip = AudioFileClip(soundscape_path)
                soundscape_clip = soundscape_clip.subclip(0, video_duration)
                
                # Mix audio: voice louder, soundscape subtle
                final_audio = CompositeAudioClip([
                    audio_clip.volumex(1.0),  # Keep voice at full volume
                    soundscape_clip.volumex(0.3)  # Make soundscape subtle
                ])
                
                os.unlink(soundscape_path)
                print("Added horror soundscape to audio")
                
            except Exception as soundscape_error:
                print(f"Soundscape error (continuing without): {soundscape_error}")
                final_audio = audio_clip
        
        # Create professional subtitles
        subtitle_clips = []
        
        if subtitle_data and len(subtitle_data) > 0:
            print("Creating professional subtitles...")
            
            # Group words into readable phrases (3-5 words each)
            phrases = []
            current_phrase = []
            current_start = 0
            
            for i, word_data in enumerate(subtitle_data):
                if isinstance(word_data, dict) and 'word' in word_data:
                    word = word_data['word']
                    start = word_data.get('start', i * 0.5)
                    end = word_data.get('end', (i + 1) * 0.5)
                    
                    if len(current_phrase) == 0:
                        current_start = start
                    
                    current_phrase.append(word)
                    
                    # Create phrase every 4 words or at sentence end
                    if (len(current_phrase) >= 4 or 
                        word.endswith('.') or word.endswith('!') or word.endswith('?') or
                        i == len(subtitle_data) - 1):
                        
                        phrase_text = ' '.join(current_phrase).strip()
                        if phrase_text:
                            phrases.append({
                                'text': phrase_text,
                                'start': current_start,
                                'end': end,
                                'duration': end - current_start
                            })
                        
                        current_phrase = []
            
            # Create subtitle clips for each phrase
            for phrase in phrases:
                if phrase['start'] < video_duration:
                    try:
                        # Professional subtitle styling
                        subtitle_clip = TextClip(
                            phrase['text'],
                            fontsize=32,
                            color='white',
                            font='Arial-Bold',
                            stroke_color='black',
                            stroke_width=2,
                            method='caption',
                            size=(image_clip.w * 0.8, None),
                            align='center'
                        ).set_position(('center', 0.85), relative=True).set_start(
                            phrase['start']
                        ).set_duration(
                            min(phrase['duration'], video_duration - phrase['start'])
                        )
                        
                        # Add fade in/out for smooth transitions
                        subtitle_clip = subtitle_clip.crossfadein(0.2).crossfadeout(0.2)
                        
                        subtitle_clips.append(subtitle_clip)
                        print(f"Created subtitle: '{phrase['text']}' at {phrase['start']:.1f}s")
                        
                    except Exception as subtitle_error:
                        print(f"Error creating subtitle clip: {subtitle_error}")
                        continue
            
            print(f"Successfully created {len(subtitle_clips)} professional subtitle clips")
        
        # Combine all elements
        main_clip = image_clip.set_audio(final_audio)
        
        if subtitle_clips:
            final_clip = CompositeVideoClip([main_clip] + subtitle_clips)
        else:
            final_clip = main_clip
        
        # Export with high quality settings
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as video_file:
            video_path = video_file.name
        
        print("Exporting enhanced video...")
        final_clip.write_videofile(
            video_path,
            fps=24,
            audio_codec='aac',
            codec='libx264',
            bitrate='1000k',  # Higher quality
            audio_bitrate='128k',
            preset='medium',
            verbose=False,
            logger=None
        )
        
        # Clean up clips
        final_clip.close()
        audio_clip.close()
        if final_audio != audio_clip:
            final_audio.close()
        del final_clip, image_clip, audio_clip
        gc.collect()
        
        # Clean up temp files
        os.unlink(img_path)
        os.unlink(audio_path)
        
        # Convert to base64
        print("Converting enhanced video to base64...")
        with open(video_path, 'rb') as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
        
        os.unlink(video_path)
        
        video_size_mb = len(video_base64) * 3 / 4 / (1024 * 1024)
        
        print(f"Enhanced video creation successful! Size: {video_size_mb:.1f}MB")
        
        return {
            'video_url': f"data:video/mp4;base64,{video_base64}",
            'video_data': video_base64,
            'session_id': session_id,
            'ai_generated': True,
            'duration': video_duration,
            'subtitle_count': len(subtitle_clips),
            'phrase_count': len(phrases) if 'phrases' in locals() else 0,
            'status': 'Enhanced video with professional subtitles created',
            'size_mb': video_size_mb,
            'features': ['whisper_subtitles', 'ken_burns_effect', 'horror_soundscape', 'professional_styling'],
            'quality': 'HD_720p'
        }
        
    except Exception as e:
        print(f"Enhanced Video Creation Error: {e}")
        return {
            'video_url': f'/videos/{session_id}_final.mp4',
            'session_id': session_id,
            'ai_generated': False,
            'error': str(e),
            'message': 'Enhanced video creation failed'
        }

# Keep your existing route handlers but update the endpoints

@app.route('/', methods=['GET', 'POST'])
def handle_everything():
    """Enhanced request handler with new features"""
    log_what_happened()
    endpoint = request.args.get('endpoint', 'home')
    
    # Home page
    if endpoint == 'home':
        return jsonify({
            'message': 'Enhanced Horror Pipeline is Running!',
            'status': 'working',
            'version': '3.0 - ENHANCED WITH WHISPER SUBTITLES & AUDIO EFFECTS',
            'features': ['whisper_timing', 'horror_soundscape', 'professional_subtitles', 'ken_burns_effect'],
            'time': time.time()
        })
    
    # Test page
    elif endpoint == 'test':
        return jsonify({
            'message': 'Enhanced test page works!',
            'endpoint': endpoint,
            'openai_configured': bool(os.getenv('OPENAI_API_KEY')),
            'elevenlabs_configured': bool(os.getenv('ELEVENLABS_API_KEY')),
            'whisper_available': True,
            'pydub_available': True,
            'moviepy_available': True,
            'subtitle_support': 'whisper_enhanced',
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
            'message': 'New enhanced session created!',
            'features': ['whisper_subtitles', 'horror_soundscape', 'ken_burns_effect']
        })
    
    # Create enhanced story
    elif endpoint == 'create-story':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        theme = data.get('theme', 'haunted house')
        
        story = generate_real_story(theme, session_id)
        
        return jsonify({
            'success': True,
            'story': story
        })
    
    # Create enhanced image
    elif endpoint == 'create-image':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        description = data.get('description', 'scary horror scene')
        
        image_result = generate_real_image(description, session_id)
        
        return jsonify({
            'success': True,
            **image_result
        })
    
    # Create voice with Whisper timing
    elif endpoint == 'create-voice':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        text = data.get('text', '')
        
        voice_result = generate_voice_with_whisper_timing(text, session_id)
        
        return jsonify({
            'success': True,
            **voice_result
        })
    
    # Create horror soundscape
    elif endpoint == 'create-soundscape':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        duration = data.get('duration', 60)
        
        soundscape_result = create_horror_soundscape(duration, session_id)
        
        return jsonify({
            'success': True,
            **soundscape_result
        })
    
    # Create enhanced video with all effects
    elif endpoint == 'create-video':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        image_url = data.get('image_url', '')
        audio_data = data.get('audio_data', '')
        subtitle_data = data.get('subtitle_data', [])
        soundscape_data = data.get('soundscape_data', '')
        
        # Debug logging
        print(f"Enhanced video creation request:")
        print(f"  session_id: {session_id}")
        print(f"  image_url length: {len(image_url) if image_url else 0}")
        print(f"  audio_data length: {len(audio_data) if audio_data else 0}")
        print(f"  subtitle_data length: {len(subtitle_data) if subtitle_data else 0}")
        print(f"  soundscape_data length: {len(soundscape_data) if soundscape_data else 0}")
        
        # Parse subtitle data if it's a string
        if isinstance(subtitle_data, str) and subtitle_data:
            try:
                subtitle_data = json.loads(subtitle_data)
                print(f"  Parsed subtitle_data: {len(subtitle_data)} items")
            except:
                print("  Failed to parse subtitle_data, using fallback")
                subtitle_data = []
        
        video_result = create_enhanced_video_with_effects(
            session_id, image_url, audio_data, subtitle_data, soundscape_data
        )
        
        return jsonify({
            'success': True,
            **video_result
        })
    
    # Unknown endpoint
    else:
        return jsonify({
            'error': 'Page not found',
            'endpoint': endpoint,
            'available_endpoints': [
                'home', 'test', 'create-session', 'create-story', 
                'create-image', 'create-voice', 'create-soundscape', 'create-video'
            ]
        }), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
