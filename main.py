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
            max_tokens=400,
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
        fallback_story = f"""The old {theme} stands silent in the moonlight. Its windows stare like empty eyes into the darkness. Something moves behind the glass. A shadow that shouldn't be there. The door creaks open on rusted hinges. Footsteps echo through empty halls. Each step brings you closer to the truth. The truth that some places should remain untouched. Some secrets should stay buried. But tonight, the {theme} remembers. And it wants you to remember too."""
        
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
        horror_prompt = f"Dark atmospheric horror scene: {description}. Cinematic lighting, shadows, eerie mood, high quality digital art"
        
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
        return {
            'image_url': f'/images/{session_id}_horror.jpg',
            'description': description,
            'session_id': session_id,
            'ai_generated': False,
            'error': str(e)
        }

def generate_free_voice_with_timing(text, session_id):
    """Generate voice using Google TTS (free) with manual subtitle timing"""
    try:
        print(f"Google TTS: Generating voice for {len(text)} characters")
        
        from gtts import gTTS
        import tempfile
        
        tts = gTTS(text=text, lang='en', slow=False)
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
            tts.save(temp_audio.name)
            temp_audio_path = temp_audio.name
        
        with open(temp_audio_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        os.unlink(temp_audio_path)
        
        words = text.split()
        subtitle_data = []
        current_time = 0
        words_per_second = 2.2
        
        for word in words:
            char_duration = len(word) * 0.08
            word_duration = max(char_duration, 0.4)
            
            subtitle_data.append({
                'word': word,
                'start': current_time,
                'end': current_time + word_duration
            })
            
            current_time += word_duration + 0.15
        
        sentences = text.split('.')
        if len(sentences) > 1:
            sentence_words_count = 0
            for sentence in sentences[:-1]:
                sentence_words = sentence.strip().split()
                sentence_words_count += len(sentence_words)
                if sentence_words_count < len(subtitle_data):
                    for i in range(sentence_words_count, len(subtitle_data)):
                        subtitle_data[i]['start'] += 0.3
                        subtitle_data[i]['end'] += 0.3
        
        print(f"Google TTS success: Generated audio with {len(subtitle_data)} timed words")
        
        return {
            'audio_url': f"data:audio/mpeg;base64,{audio_base64}",
            'audio_data': audio_base64,
            'text': text,
            'text_length': len(text),
            'voice_id': 'google_tts_free',
            'session_id': session_id,
            'ai_generated': True,
            'duration_estimate': current_time,
            'subtitle_data': subtitle_data,
            'status': 'Google TTS with manual timing generated',
            'service': 'google_tts_free'
        }
        
    except Exception as e:
        print(f"Google TTS Error: {e}")
        words = text.split()[:30]
        subtitle_data = []
        for i, word in enumerate(words):
            subtitle_data.append({
                'word': word,
                'start': i * 0.6,
                'end': (i + 1) * 0.6
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
            'service': 'fallback_timing_only'
        }

def generate_enhanced_voice_with_whisper_timing(text, session_id):
    """Enhanced voice generation with Whisper for precise subtitle timing"""
    try:
        print(f"Enhanced Voice: Generating for {len(text)} characters")
        
        # First, generate audio with Google TTS (your existing method)
        from gtts import gTTS
        import tempfile
        
        # Generate TTS audio
        tts = gTTS(text=text, lang='en', slow=False)
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
            tts.save(temp_audio.name)
            temp_audio_path = temp_audio.name
        
        # Use Whisper for precise timing (if available)
        try:
            print("Loading Whisper model for precise timing...")
            import whisper
            import torch
            import gc
            
            # Use tiny model for speed on Render.com
            model = whisper.load_model("tiny")
            
            # Transcribe with word-level timestamps
            result = model.transcribe(
                temp_audio_path, 
                word_timestamps=True,
                language='en'
            )
            
            # Extract precise word timings
            subtitle_data = []
            for segment in result.get('segments', []):
                for word_info in segment.get('words', []):
                    subtitle_data.append({
                        'word': word_info['word'].strip(),
                        'start': word_info['start'],
                        'end': word_info['end'],
                        'confidence': word_info.get('probability', 1.0)
                    })
            
            print(f"Whisper success: {len(subtitle_data)} precisely timed words")
            timing_method = "whisper_precise"
            
            # Clean up memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as whisper_error:
            print(f"Whisper failed, using manual timing: {whisper_error}")
            # Fallback to your existing manual timing
            subtitle_data = generate_manual_timing(text)
            timing_method = "manual_fallback"
        
        # Convert audio to base64
        with open(temp_audio_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        os.unlink(temp_audio_path)
        
        return {
            'audio_url': f"data:audio/mpeg;base64,{audio_base64}",
            'audio_data': audio_base64,
            'text': text,
            'text_length': len(text),
            'voice_id': f'google_tts_{timing_method}',
            'session_id': session_id,
            'ai_generated': True,
            'duration_estimate': subtitle_data[-1]['end'] if subtitle_data else 60,
            'subtitle_data': subtitle_data,
            'timing_method': timing_method,
            'status': f'Enhanced voice with {timing_method} timing',
            'service': 'google_tts_enhanced'
        }
        
    except Exception as e:
        print(f"Enhanced Voice Error: {e}")
        # Complete fallback to your existing method
        return generate_free_voice_with_timing(text, session_id)

def generate_manual_timing(text):
    """Manual timing logic as fallback"""
    words = text.split()
    subtitle_data = []
    current_time = 0
    
    for word in words:
        char_duration = len(word) * 0.08
        word_duration = max(char_duration, 0.4)
        
        subtitle_data.append({
            'word': word,
            'start': current_time,
            'end': current_time + word_duration,
            'confidence': 1.0  # Manual timing always "confident"
        })
        
        current_time += word_duration + 0.15
    
    return subtitle_data

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
        
        if not image_url or image_url.startswith('/images/'):
            raise Exception("Invalid image URL")
        
        print("Downloading image...")
        image_response = requests.get(image_url, timeout=15)
        if image_response.status_code != 200:
            raise Exception("Failed to download image")
        
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
                has_audio = True
            else:
                has_audio = False
                audio_path = None
        
        print("Creating optimized video...")
        
        image_clip = ImageClip(img_path, duration=60)
        image_clip = image_clip.resize(height=480)
        
        if has_audio and audio_path:
            audio_clip = AudioFileClip(audio_path)
            video_duration = min(audio_clip.duration, 60)
            image_clip = image_clip.set_duration(video_duration)
        else:
            video_duration = 60
            audio_clip = None
        
        subtitle_clips = []
        if subtitle_data and len(subtitle_data) > 0:
            print("Creating simplified subtitles...")
            
            if isinstance(subtitle_data, list) and len(subtitle_data) > 0:
                if isinstance(subtitle_data[0], dict):
                    words = []
                    for item in subtitle_data[:10]:
                        if 'word' in item:
                            words.append(item['word'])
                        elif 'text' in item:
                            words.append(item['text'])
                        elif isinstance(item, str):
                            words.append(item)
                    
                    if words:
                        words_per_segment = max(1, len(words) // 3)
                        segment_duration = video_duration / 3
                        
                        for i in range(3):
                            start_time = i * segment_duration
                            if start_time < video_duration:
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
                                        continue
                
                elif isinstance(subtitle_data[0], str):
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
        
        if has_audio and audio_clip:
            main_clip = image_clip.set_audio(audio_clip)
        else:
            main_clip = image_clip
            
        if subtitle_clips:
            final_clip = CompositeVideoClip([main_clip] + subtitle_clips)
        else:
            final_clip = main_clip
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as video_file:
            video_path = video_file.name
        
        print("Exporting optimized video...")
        final_clip.write_videofile(
            video_path,
            fps=15,
            audio_codec='aac' if has_audio else None,
            codec='libx264',
            bitrate='500k',
            verbose=False,
            logger=None
        )
        
        final_clip.close()
        if has_audio and audio_clip:
            audio_clip.close()
        del final_clip, image_clip
        if has_audio and audio_clip:
            del audio_clip
        gc.collect()
        
        os.unlink(img_path)
        if has_audio and audio_path:
            os.unlink(audio_path)
        
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
    
    if endpoint == 'home':
        return jsonify({
            'message': 'Horror Pipeline is Running!',
            'status': 'working',
            'version': '2.6 - AI PIPELINE WITH ENHANCED SUBTITLES',
            'time': time.time()
        })
    
    elif endpoint == 'test':
        return jsonify({
            'message': 'Test page works!',
            'endpoint': endpoint,
            'openai_configured': bool(os.getenv('OPENAI_API_KEY')),
            'elevenlabs_configured': bool(os.getenv('ELEVENLABS_API_KEY')),
            'moviepy_available': True,
            'subtitle_support': True,
            'whisper_available': False,  # Will be True when requirements updated
            'time': time.time()
        })
    
    elif not check_password():
        return jsonify({
            'error': 'Wrong password!',
            'message': 'You need: Authorization: mysecret1303'
        }), 401
    
    elif endpoint == 'create-session':
        session_id = f"horror_{int(time.time())}"
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'New session created!'
        })
    
    elif endpoint == 'create-story':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        theme = data.get('theme', 'haunted house')
        
        story = generate_real_story(theme, session_id)
        
        return jsonify({
            'success': True,
            'story': story
        })
    
    elif endpoint == 'create-image':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        description = data.get('description', 'scary horror scene')
        
        image_result = generate_real_image(description, session_id)
        
        return jsonify({
            'success': True,
            **image_result
        })
    
    elif endpoint == 'create-voice':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        text = data.get('text', '')
        
        voice_result = generate_free_voice_with_timing(text, session_id)
        
        return jsonify({
            'success': True,
            **voice_result
        })
    
    elif endpoint == 'create-voice-enhanced':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        text = data.get('text', '')
        
        print(f"Enhanced voice request: {session_id}, text length: {len(text)}")
        
        # Try enhanced version first, fallback to original if it fails
        try:
            voice_result = generate_enhanced_voice_with_whisper_timing(text, session_id)
            print(f"Enhanced voice successful: {voice_result.get('timing_method', 'unknown')}")
        except Exception as e:
            print(f"Enhanced voice failed, using original method: {e}")
            voice_result = generate_free_voice_with_timing(text, session_id)
            # Add flag to indicate fallback was used
            voice_result['enhanced_attempted'] = True
            voice_result['enhanced_error'] = str(e)
        
        return jsonify({
            'success': True,
            **voice_result
        })
    
    elif endpoint == 'create-video':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        image_url = data.get('image_url', '')
        audio_data = data.get('audio_data', '')
        subtitle_data = data.get('subtitle_data', [])
        
        print(f"Video creation request data:")
        print(f"  session_id: {session_id}")
        print(f"  image_url length: {len(image_url) if image_url else 0}")
        print(f"  audio_data length: {len(audio_data) if audio_data else 0}")
        print(f"  subtitle_data type: {type(subtitle_data)}")
        print(f"  subtitle_data length: {len(subtitle_data) if subtitle_data else 0}")
        
        if isinstance(subtitle_data, str):
            try:
                import json
                subtitle_data = json.loads(subtitle_data)
                print(f"  Parsed subtitle_data: {subtitle_data}")
            except:
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
    
    else:
        return jsonify({
            'error': 'Page not found',
            'endpoint': endpoint
        }), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
