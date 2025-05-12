from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Fetch and configure Google API key
gemini_api_key = os.getenv("GOOGLE_API_KEY")
if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
        logger.info("Google Gemini API key configured successfully.")
    except Exception as config_err:
        logger.error(f"API configuration failed: {config_err}")
        gemini_api_key = None
else:
    logger.error("GOOGLE_API_KEY not set in environment.")

# Choose a generative model
SELECTED_MODEL = "gemini-1.5-flash-latest"
model_instance = None
if gemini_api_key:
    try:
        model_instance = genai.GenerativeModel(SELECTED_MODEL)
        logger.info(f"Model '{SELECTED_MODEL}' initialized.")
    except Exception as model_err:
        logger.error(f"Error initializing model: {model_err}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_story():
    if not gemini_api_key or not model_instance:
        return jsonify({"error": "Service is not properly configured."}), 500

    try:
        input_data = request.get_json()
        prompt_text = input_data.get('prompt', '').strip()

        if not prompt_text:
            return jsonify({"error": "No prompt provided."}), 400
        if len(prompt_text) > 2000:
            return jsonify({"error": "Prompt too long. Limit is 2000 characters."}), 400

        crafted_prompt = f"""You're a talented story writer.
Here's a starting line: "{prompt_text}"
Continue the story in a fun and creative way (150-200 words)."""

        gen_config = genai.types.GenerationConfig(
            max_output_tokens=300,
            temperature=0.85,
            top_p=0.9,
            top_k=50
        )

        try:
            ai_response = model_instance.generate_content(
                crafted_prompt,
                generation_config=gen_config
            )

            if not ai_response.candidates:
                feedback = ai_response.prompt_feedback
                block_reason = feedback.block_reason.name if feedback else "Unknown"
                logger.warning(f"Response blocked: {block_reason}")
                return jsonify({"error": f"Content blocked: {block_reason}"}), 400

            story = ai_response.text.strip()
            return jsonify({"story": story if story else "The AI couldn't continue the story. Try another prompt."})

        except Exception as api_exc:
            logger.error(f"Gemini API error: {api_exc}", exc_info=True)
            return jsonify({"error": "An error occurred while contacting the Gemini API."}), 500

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({"error": "A server error occurred."}), 500

if __name__ == '__main__':
    if not gemini_api_key:
        logger.warning("GOOGLE_API_KEY is not set. API features will be disabled.")
    if not model_instance:
        logger.warning(f"Model '{SELECTED_MODEL}' not initialized.")

    app.run(host='0.0.0.0', port=5000, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
