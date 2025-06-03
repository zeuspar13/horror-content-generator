# Horror Content Generator – Project Setup

This repository includes everything needed to:

1. **Phase 1**: Fix your n8n HTTP nodes so that the current pipeline (Story → Image → Voice → Video) works again.
2. **Phase 2**: Add Whisper-based subtitle timing (while preserving backward compatibility), test locally, deploy to Render staging, and then switch n8n to the new enhanced endpoint.

---

## Phase 1: Fix n8n HTTP Nodes

Follow these steps to make sure each n8n node sends exactly the JSON your Flask API expects.

### 1.1 – Session Creation Node

- **Method**: `POST`
- **URL**: `https://horror-content-generator.onrender.com/?endpoint=create-session`
- **Send Headers**: ON
  - `Authorization: mysecret1303`
  - `Content-Type: application/json`
- **Send Body**: OFF (no JSON needed for this endpoint)

**Test**:
> Click **Execute Step**. You should see something like:
> ```json
> {
>   "success": true,
>   "session_id": "horror_1748903823"
> }
> ```

### 1.2 – Story Creation Node

- **Method**: `POST`
- **URL**: `https://horror-content-generator.onrender.com/?endpoint=create-story`
- **Send Headers**: ON
  - `Authorization: mysecret1303`
  - `Content-Type: application/json`
- **Send Body**: ON
  - **Body Content Type**: `JSON`
  - **JSON Body**:
    ```json
    {
      "session_id": "={{ $json[\"session_id\"] }}"
    }
    ```

**Test**:
> Click **Execute Step**. You should see a response like:
> ```json
> {
>   "success": true,
>   "session_id": "horror_1748903823",
>   "story": "Once upon a midnight dreary…"
> }
> ```

### 1.3 – Image Creation Node

- **Method**: `POST`
- **URL**: `https://horror-content-generator.onrender.com/?endpoint=create-image`
- **Send Headers**: ON
  - `Authorization: mysecret1303`
  - `Content-Type: application/json`
- **Send Body**: ON
  - **Body Content Type**: `JSON`
  - **JSON Body**:
    ```json
    {
      "session_id": "={{ $json[\"session_id\"] }}",
      "story": "={{ $json[\"story\"] }}"
    }
    ```

**Test**:
> Click **Execute Step**. You should see:
> ```json
> {
>   "success": true,
>   "session_id": "horror_1748903823",
>   "image_path": "/tmp/abc.png"
> }
> ```

### 1.4 – Voice Creation Node

- **Method**: `POST`
- **URL**: `https://horror-content-generator.onrender.com/?endpoint=create-voice`
- **Send Headers**: ON
  - `Authorization: mysecret1303`
  - `Content-Type: application/json`
- **Send Body**: ON
  - **Body Content Type**: `JSON`
  - **JSON Body**:
    ```json
    {
      "session_id": "={{ $json[\"session_id\"] }}",
      "text": "={{ $json[\"story\"] }}"
    }
    ```

**Test**:
> Click **Execute Step**. You should see something like:
> ```json
> {
>   "success": true,
>   "audio_path": "/tmp/…123.mp3",
>   "subtitle_data": [ … ],
>   "timing_method": "manual"
> }
> ```

### 1.5 – Video Creation Node

- **Method**: `POST`
- **URL**: `https://horror-content-generator.onrender.com/?endpoint=create-video`
- **Send Headers**: ON
  - `Authorization: mysecret1303`
  - `Content-Type: application/json`
- **Send Body**: ON
  - **Body Content Type**: `JSON`
  - **JSON Body**:
    ```json
    {
      "session_id": "={{ $json[\"session_id\"] }}",
      "image_path": "={{ $json[\"image_path\"] }}",
      "audio_path": "={{ $json[\"audio_path\"] }}",
      "subtitle_data": "={{ $json[\"subtitle_data\"] }}"
    }
    ```

**Test**:
> Click **Execute Step**. You should see:
> ```json
> {
>   "success": true,
>   "video_path": "/tmp/video.mp4"
> }
> ```

### 1.6 – Execute Full Workflow

- Click **Execute Workflow** in n8n.  
- If each step returns a successful response (Story, Image, Voice, Video), you have completed Phase 1.

---

## Phase 2: Add Whisper-based Subtitles

> **Important**: Only begin Phase 2 once Phase 1 is fully working.

### 2.1 – Create a Git Branch & Update Dependencies

1. In your local project folder:
   ```bash
   git checkout -b subtitle/whisper-timing
