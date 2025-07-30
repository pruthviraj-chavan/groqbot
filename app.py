import os
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather
from groq import Groq

app = Flask(__name__)

# API credentials - Using environment variables for production security
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")

# Initialize Groq client with error handling
try:
    client = Groq(api_key=GROQ_API_KEY)
    print("✅ Groq client initialized successfully")
except Exception as e:
    print(f"❌ Error initializing Groq client: {e}")
    client = None

# Conversation context storage
conversation_contexts = {}

def get_conversation_context(caller_id):
    """Get or create conversation context for a caller"""
    if caller_id not in conversation_contexts:
        conversation_contexts[caller_id] = [
            {
                "role": "system", 
                "content": "आप एक सहायक हिंदी AI असिस्टेंट हैं। आप मुख्यतः हिंदी में जवाब देते हैं लेकिन अंग्रेजी भी समझ सकते हैं। अपने जवाब संक्षिप्त, स्पष्ट और मददगार रखें। फोन कॉल के लिए 30-40 शब्दों में जवाब दें।"
            }
        ]
    return conversation_contexts[caller_id]

def generate_response(prompt, caller_id):
    """Generate AI response using Groq API with enhanced error handling"""
    try:
        # Validate API key and client
        if not GROQ_API_KEY:
            print("ERROR: GROQ_API_KEY not found")
            return "तकनीकी समस्या है। कृपया बाद में कोशिश करें।"
        
        if not client:
            print("ERROR: Groq client not initialized")
            return "तकनीकी समस्या है। कृपया बाद में कोशिश करें।"
            
        # Validate input
        if not prompt or prompt.strip() == "":
            return "मुझे आपका सवाल समझ नहीं आया। कृपया दोबारा बोलें।"
            
        messages = get_conversation_context(caller_id)
        messages.append({"role": "user", "content": prompt})
        
        print(f"Sending to Groq: {prompt}")  # Debug log
        
        # Generate response using Groq
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=100,
            temperature=0.7,
            top_p=1,
            stream=False
        )
        
        response_text = completion.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": response_text})
        
        print(f"Groq response: {response_text}")  # Debug log
        return response_text
        
    except Exception as e:
        print(f"Error generating response: {e}")
        print(f"Error type: {type(e)}")
        return "मुझे खुशी होगी आपकी सहायता करने में। कृपया फिर से कोशिश करें।"

@app.route("/voice", methods=['POST'])
def voice():
    """Handle incoming voice calls with comprehensive error handling"""
    try:
        response = VoiceResponse()
        caller_id = request.form.get('From', 'unknown')
        
        print(f"Incoming call from: {caller_id}")
        print(f"Request form data: {dict(request.form)}")  # Debug log
        
        # Welcome message
        response.say(
            "नमस्ते! मैं आपका AI सहायक हूं। मैं हिंदी और अंग्रेजी में आपकी सहायता कर सकता हूं।", 
            voice="Polly.Aditi", 
            language="hi-IN"
        )
        
        # Gather user input
        gather = Gather(
            input="speech", 
            action="/process_voice", 
            method="POST", 
            language="hi-IN", 
            speech_timeout="auto",
            timeout=8,
            hints="हाँ,नहीं,धन्यवाद,नमस्ते,समस्या,सवाल,मदद"
        )
        
        gather.say(
            "कृपया अपना सवाल या समस्या बताएं।", 
            voice="Polly.Aditi", 
            language="hi-IN"
        )
        response.append(gather)
        
        # Fallback if no input
        response.say(
            "कोई आवाज़ नहीं सुनी गई। कॉल के लिए धन्यवाद!", 
            voice="Polly.Aditi", 
            language="hi-IN"
        )
        
        return Response(str(response), content_type='text/xml')
        
    except Exception as e:
        print(f"Error in voice route: {e}")
        response = VoiceResponse()
        response.say(
            "तकनीकी समस्या है। कृपया बाद में कोशिश करें।", 
            voice="Polly.Aditi", 
            language="hi-IN"
        )
        return Response(str(response), content_type='text/xml')

@app.route("/process_voice", methods=['POST'])
def process_voice():
    """Process user speech and generate AI response with error handling"""
    try:
        response = VoiceResponse()
        caller_id = request.form.get('From', 'unknown')
        
        # Get speech input from Twilio
        speech_text = request.form.get("SpeechResult", "")
        confidence = request.form.get("Confidence", "0")
        
        print(f"User said: '{speech_text}' (Confidence: {confidence})")
        print(f"Process voice form data: {dict(request.form)}")  # Debug log
        
        # Handle empty or unclear speech
        if not speech_text or speech_text.lower().strip() in ['', 'um', 'uh', 'hmm']:
            response.say(
                "मुझे आपकी आवाज़ साफ़ सुनाई नहीं दी। कृपया दोबारा बोलें।", 
                voice="Polly.Aditi", 
                language="hi-IN"
            )
            
            # Give another chance to speak
            gather = Gather(
                input="speech", 
                action="/process_voice", 
                method="POST", 
                language="hi-IN", 
                speech_timeout="auto",
                timeout=8,
                hints="हाँ,नहीं,धन्यवाद,नमस्ते,समस्या,सवाल,मदद"
            )
            gather.say("कृपया बोलें।", voice="Polly.Aditi", language="hi-IN")
            response.append(gather)
            
            # Final fallback
            response.say(
                "कॉल के लिए धन्यवाद। नमस्ते!", 
                voice="Polly.Aditi", 
                language="hi-IN"
            )
            return Response(str(response), content_type='text/xml')
        
        # Check for goodbye/end call phrases
        end_phrases = ['अलविदा', 'धन्यवाद', 'बाय', 'bye', 'goodbye', 'खत्म', 'समाप्त', 'बंद करो', 'रुको']
        if any(phrase in speech_text.lower() for phrase in end_phrases):
            response.say(
                "बातचीत के लिए धन्यवाद! आपका दिन शुभ हो।", 
                voice="Polly.Aditi", 
                language="hi-IN"
            )
            response.hangup()
            return Response(str(response), content_type='text/xml')
        
        # Generate AI response using Groq
        ai_response = generate_response(speech_text, caller_id)
        print(f"AI response: {ai_response}")
        
        # Speak the AI response
        response.say(ai_response, voice="Polly.Aditi", language="hi-IN")
        
        # Continue conversation
        gather = Gather(
            input="speech", 
            action="/process_voice", 
            method="POST", 
            language="hi-IN", 
            speech_timeout="auto",
            timeout=8,
            hints="हाँ,नहीं,और,अधिक,समझाओ,धन्यवाद,अलविदा"
        )
        
        gather.say(
            "क्या आपका कोई और सवाल है?", 
            voice="Polly.Aditi", 
            language="hi-IN"
        )
        response.append(gather)
        
        # End call gracefully if no more input
        response.say(
            "बातचीत के लिए धन्यवाद। नमस्ते!", 
            voice="Polly.Aditi", 
            language="hi-IN"
        )
        
        return Response(str(response), content_type='text/xml')
        
    except Exception as e:
        print(f"Error in process_voice route: {e}")
        response = VoiceResponse()
        response.say(
            "तकनीकी समस्या है। कृपया बाद में कोशिश करें।", 
            voice="Polly.Aditi", 
            language="hi-IN"
        )
        response.hangup()
        return Response(str(response), content_type='text/xml')

@app.route("/", methods=['GET'])
def home():
    return """
    <h1>🤖 Indian Voice Bot is Running!</h1>
    <p><strong>Status:</strong> Active and ready for calls</p>
    <p><strong>Groq API:</strong> Connected</p>
    <p><strong>Twilio:</strong> Configured</p>
    <p><strong>Language:</strong> Hindi (Primary), English (Secondary)</p>
    <p><strong>Webhook URL:</strong> /voice</p>
    <p><strong>Test Endpoint:</strong> <a href="/test">/test</a></p>
    <p><strong>Debug Endpoint:</strong> <a href="/debug">/debug</a></p>
    """

@app.route("/test", methods=['GET'])
def test():
    """Test endpoint to verify Groq API connection"""
    try:
        if not client:
            return "<h2>❌ Groq API Test Failed</h2><p><strong>Error:</strong> Groq client not initialized</p>"
        
        test_response = generate_response("नमस्ते, आप कैसे हैं?", "test_user")
        return f"<h2>✅ Groq API Test Successful!</h2><p><strong>Response:</strong> {test_response}</p>"
    except Exception as e:
        return f"<h2>❌ Groq API Test Failed</h2><p><strong>Error:</strong> {str(e)}</p>"

@app.route("/debug", methods=['GET'])
def debug():
    """Debug endpoint to check environment variables and system status"""
    return f"""
    <h2>🔧 System Debug Information</h2>
    <h3>Environment Variables:</h3>
    <p><strong>GROQ_API_KEY:</strong> {'✅ Set' if GROQ_API_KEY else '❌ Missing'}</p>
    <p><strong>TWILIO_ACCOUNT_SID:</strong> {'✅ Set' if TWILIO_ACCOUNT_SID else '❌ Missing'}</p>
    <p><strong>TWILIO_AUTH_TOKEN:</strong> {'✅ Set' if TWILIO_AUTH_TOKEN else '❌ Missing'}</p>
    
    <h3>System Status:</h3>
    <p><strong>Groq Client:</strong> {'✅ Initialized' if client else '❌ Failed'}</p>
    <p><strong>Active Conversations:</strong> {len(conversation_contexts)}</p>
    <p><strong>Flask Debug:</strong> {app.debug}</p>
    
    <h3>Webhook URLs:</h3>
    <p><strong>Voice:</strong> https://[your-app-url]/voice</p>
    <p><strong>Process:</strong> https://[your-app-url]/process_voice</p>
    """

@app.route("/health", methods=['GET'])
def health():
    """Health check endpoint for monitoring"""
    try:
        status = {
            "status": "healthy" if client else "unhealthy",
            "groq_client": "connected" if client else "disconnected",
            "active_conversations": len(conversation_contexts)
        }
        return status
    except Exception as e:
        return {"status": "error", "error": str(e)}, 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("🚀 Starting Indian Voice Bot...")
    print(f"✅ Groq API Key: {'Loaded' if GROQ_API_KEY else 'Missing'}")
    print(f"✅ Twilio Account SID: {'Loaded' if TWILIO_ACCOUNT_SID else 'Missing'}")
    print(f"✅ Twilio Auth Token: {'Loaded' if TWILIO_AUTH_TOKEN else 'Missing'}")
    print(f"🌐 Server starting on port {port}")
    print(f"🔧 Debug mode: {debug_mode}")
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
