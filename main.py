from flask import Flask, request, jsonify
import os
import time

app = Flask(__name__)

# Secret password for security
SECRET_PASSWORD = "mysecret1303"

def check_password():
    """Check if the request has the right password"""
    password = request.headers.get('Authorization', '')
    print(f"Password check: got '{password}', need '{SECRET_PASSWORD}'")
    return password == SECRET_PASSWORD

def log_what_happened():
    """Write down what request we got"""
    endpoint = request.args.get('endpoint', 'home')
    method = request.method
    password = request.headers.get('Authorization', 'None')
    print(f"Request: {method} /?endpoint={endpoint}, Password: {password}")

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
            'time': time.time()
        })
    
    # Test page - anyone can visit  
    elif endpoint == 'test':
        return jsonify({
            'message': 'Test page works!',
            'endpoint': endpoint,
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
    
    # Create a horror story
    elif endpoint == 'create-story':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        theme = data.get('theme', 'haunted house')
        
        # This is fake for now - we'll make it real later
        story = {
            'title': f'The Curse of the {theme.title()}',
            'text': f'Deep in the shadows of the {theme}, ancient evil stirs. The walls whisper secrets of the damned, and every footstep echoes with the screams of the forgotten. Tonight, the darkness comes alive...',
            'scenes': [
                {'number': 1, 'description': f'Approaching the {theme}', 'seconds': 8},
                {'number': 2, 'description': 'Strange sounds begin', 'seconds': 12},
                {'number': 3, 'description': 'The horror is revealed', 'seconds': 10}
            ],
            'session_id': session_id
        }
        
        return jsonify({
            'success': True,
            'story': story
        })
    
    # Create a horror image
    elif endpoint == 'create-image':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        description = data.get('description', 'scary scene')
        
        # This is fake for now - we'll make it real later
        return jsonify({
            'success': True,
            'image_url': f'/images/{session_id}_horror.jpg',
            'description': description
        })
    
    # Create a horror voice
    elif endpoint == 'create-voice':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        text = data.get('text', '')
        
        # This is fake for now - we'll make it real later
        return jsonify({
            'success': True,
            'audio_url': f'/audio/{session_id}_voice.mp3',
            'text_length': len(text)
        })
    
    # Create the final video
    elif endpoint == 'create-video':
        data = request.get_json() or {}
        session_id = data.get('session_id', 'unknown')
        
        # This is fake for now - we'll make it real later
        return jsonify({
            'success': True,
            'video_url': f'/videos/{session_id}_final.mp4',
            'message': 'Video created successfully!'
        })
    
    # Unknown page
    else:
        return jsonify({
            'error': 'Page not found',
            'endpoint': endpoint,
            'available_pages': ['home', 'test', 'create-session', 'create-story', 'create-image', 'create-voice', 'create-video']
        }), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
