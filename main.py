from flask import Flask, request, jsonify
import os
import time
import openai

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
        response = openai.chat.completions.create(
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
        
        # Parse the AI response into our format
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
        print(f"OpenAI Error: {e}")
        # Fallback to mock story if OpenAI fails
        return {
            'title': f'The Curse of the {theme.title()}',
            'text': f'Deep in the shadows of the {theme}, ancient evil stirs. The walls whisper secrets of the damned, and every footstep echoes with the screams of the forgotten. Tonight, the darkness comes alive...',
            'scenes': [
                {'number': 1, 'description': f'Approaching the {theme}', 'seconds': 8},
                {'number': 2, 'description': 'Strange sounds begin', 'seconds': 12},
                {'number': 3, 'description': 'The horror is revealed', 'seconds': 10}
            ],
            'session_id': session_id,
            'ai_generated': False,
            'error': str(e)
        }

@app.route('/', methods=['GET', 'POST'])
def handle_everything():
    """This handles ALL requests to our website"""
    log_what_happened()
    endpoint = request.args.get('endpoint', 'home')
    
    # Home page - anyone can visit
    if endpoint == 'home':
        return jsonify({
            'message': 'Horror Pipeline is Running!',
            'status': 'working',
            'version': '2.0 - AI Powered',
            'time': time.time()
        })
    
    # Test page - anyone can visit  
    elif endpoint == 'test':
        return jsonify({
            'message': 'Test page works!',
            'endpoint': endpoint,
            'openai_configured': bool(os.getenv('OPENAI_API_KEY')),
            'time': time.time()
        })
    
    # All other pages need the password
    elif not check_password():
        return jsonify({
            'error': 'Wrong password!',
            'message': 'You need: Authorization: mysecret1303'
        }), 401
    
    # Create a new session
    elif endpoint == 'create-session':
        session_id = f"horror_{int(time.time())}"
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'New session created!'
        })
    
    # Create a horror story - NOW WITH REAL AI!
    elif endpoint == 'create-story':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        theme = data.get('theme', 'haunted house')
        
        # Generate real AI story
        story = generate_real_story(theme, session_id)
        
        return jsonify({
            'success': True,
            'story': story
        })
    
    # Create a horror image (still mock for now)
    elif endpoint == 'create-image':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        description = data.get('description', 'scary scene')
        
        return jsonify({
            'success': True,
            'image_url': f'/images/{session_id}_horror.jpg',
            'description': description
        })
    
    # Create a horror voice (still mock for now)
    elif endpoint == 'create-voice':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        text = data.get('text', '')
        
        return jsonify({
            'success': True,
            'audio_url': f'/audio/{session_id}_voice.mp3',
            'text_length': len(text)
        })
    
    # Create the final video (still mock for now)
    elif endpoint == 'create-video':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        
        return jsonify({
            'success': True,
            'video_url': f'/videos/{session_id}_final.mp4',
            'message': 'Video created successfully!'
        })
    
    # Unknown page
    else:
        return jsonify({
            'error': 'Page not found',
            'endpoint': endpoint
        }), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
