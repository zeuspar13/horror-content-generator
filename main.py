from flask import Flask, request, jsonify
import os
import time
import openai
import requests
import base64
import random

# Fix PIL.Image.ANTIALIAS compatibility for newer Pillow versions
try:
    from PIL import Image
    if not hasattr(Image, 'ANTIALIAS'):
        Image.ANTIALIAS = Image.LANCZOS
except:
    pass

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

def generate_voice_with_retry_and_url(text, session_id):
    """Generate voice with rate limit handling and return HTTP URL"""
    try:
        print(f"Voice generation: Generating for {len(text)} characters")
        
        from gtts import gTTS
        import tempfile
        
        # Add random delay to avoid rate limits
        delay = random.uniform(2, 5)
        print(f"Adding {delay:.1f}s delay to avoid rate limits")
        time.sleep(delay)
        
        # Try with slower speech and better error handling
        try:
            tts = gTTS(text=text, lang='en', slow=False)
        except Exception as fast_error:
            print(f"Fast TTS failed: {fast_error}, trying slow...")
            time.sleep(3)
            try:
                tts = gTTS(text=text, lang='en', slow=True)
            except Exception as slow_error:
                print(f"Slow TTS also failed: {slow_error}")
                raise slow_error
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
            tts.save(temp_audio.name)
            temp_audio_path = temp_audio.name
        
        # Read the audio file
        with open(temp_audio_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            
        # Clean up temp file
        os.unlink(temp_audio_path)
        
        # Convert to base64 data URL
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Generate manual timing since we have successful audio
        words = text.split()
        subtitle_data = []
        current_time = 0
        
        for word in words:
            char_duration = len(word) * 0.08
            word_duration = max(char_duration, 0.4)
            
            subtitle_data.append({
                'word': word,
                'start': current_time,
                'end': current_time + word_duration
            })
            
            current_time += word_duration + 0.15
        
        # Add pauses for punctuation
        sentences = text.split('.')
        if len(sentences) > 1:
            sentence_words_count = 0
            for sentence in sentences[:-1]:
                sentence_words = sentence.strip().split()
                sentence_words_count += len(sentence_words)
                if sentence_words_count < len(subtitle_data):
                    # Add pause after sentence
                    for i in range(sentence_words_count, len(subtitle_data)):
                        subtitle_data[i]['start'] += 0.3
                        subtitle_data[i]['end'] += 0.3
        
        print(f"Voice generation successful: {len(subtitle_data)} timed words")
        
        return {
            'audio_url': f"data:audio/mp3;base64,{audio_base64}",
            'audio_data': audio_base64,
            'text': text,
            'text_length': len(text),
            'session_id': session_id,
            'ai_generated': True,
            'duration_estimate': current_time,
            'subtitle_data': subtitle_data,
            'status': 'Voice generated with rate limit handling',
            'service': 'google_tts_retry'
        }
        
    except Exception as e:
        print(f"Voice generation error: {e}")
        
        # Enhanced fallback: return timing-only data with better estimates
        words = text.split()[:30]  # Limit to prevent long processing
        subtitle_data = []
        for i, word in enumerate(words):
            start_time = i * 0.6
            end_time = (i + 1) * 0.6
            subtitle_data.append({
                'word': word,
                'start': start_time,
                'end': end_time
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
            'service': 'fallback_timing_only',
            'status': 'Audio generation failed, returning timing data only'
        }

def generate_enhanced_voice_with_whisper_timing(text, session_id):
    """Enhanced voice generation with Whisper for precise subtitle timing"""
    try:
        print(f"Enhanced Voice: Generating for {len(text)} characters")
        
        # First, try our improved TTS method
        voice_result = generate_voice_with_retry_and_url(text, session_id)
        
        # If TTS was successful, try to enhance with Whisper
        if voice_result.get('ai_generated', False) and voice_result.get('audio_data'):
            try:
                print("Attempting Whisper enhancement...")
                import whisper
                import tempfile
                
                # Save audio to temp file for Whisper
                audio_bytes = base64.b64decode(voice_result['audio_data'])
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
                    temp_audio.write(audio_bytes)
                    temp_audio_path = temp_audio.name
                
                # Use tiny model for speed
                model = whisper.load_model("tiny")
                result = model.transcribe(temp_audio_path, word_timestamps=True, language='en')
                
                # Extract precise word timings
                enhanced_subtitle_data = []
                for segment in result.get('segments', []):
                    for word_info in segment.get('words', []):
                        enhanced_subtitle_data.append({
                            'word': word_info['word'].strip(),
                            'start': word_info['start'],
                            'end': word_info['end'],
                            'confidence': word_info.get('probability', 1.0)
                        })
                
                # Clean up
                os.unlink(temp_audio_path)
                del model
                
                if enhanced_subtitle_data:
                    voice_result['subtitle_data'] = enhanced_subtitle_data
                    voice_result['timing_method'] = 'whisper_enhanced'
                    print(f"Whisper enhancement successful: {len(enhanced_subtitle_data)} precisely timed words")
                
            except Exception as whisper_error:
                print(f"Whisper enhancement failed: {whisper_error}")
                voice_result['timing_method'] = 'manual_fallback'
        
        return voice_result
        
    except Exception as e:
        print(f"Enhanced Voice Error: {e}")
        return generate_voice_with_retry_and_url(text, session_id)

@app.route('/', methods=['GET', 'POST'])
def handle_everything():
    """This handles ALL requests to our website"""
    log_what_happened()
    endpoint = request.args.get('endpoint', 'home')
    
    if endpoint == 'home':
        return jsonify({
            'message': 'Horror Pipeline is Running!',
            'status': 'working',
            'version': '2.7 - SHOTSTACK INTEGRATION WITH IMPROVED VOICE',
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
            'whisper_available': True,
            'shotstack_ready': True,
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
            **story
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
        
        voice_result = generate_voice_with_retry_and_url(text, session_id)
        
        return jsonify({
            'success': True,
            **voice_result
        })
    
    elif endpoint == 'create-voice-enhanced':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        text = data.get('text', '')
        
        print(f"Enhanced voice request: {session_id}, text length: {len(text)}")
        
        voice_result = generate_enhanced_voice_with_whisper_timing(text, session_id)
        
        return jsonify({
            'success': True,
            **voice_result
        })
    
    else:
        return jsonify({
            'error': 'Page not found',
            'endpoint': endpoint,
            'available_endpoints': [
                'home', 'test', 'create-session', 'create-story', 
                'create-image', 'create-voice', 'create-voice-enhanced'
            ]
        }), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
