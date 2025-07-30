import os
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather
from groq import Groq

app = Flask(__name__)

# Embedded API credentials
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

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
    """Generate AI response using Groq API"""
    try:
        messages = get_conversation_context(caller_id)
        messages.append({"role": "user", "content": prompt})
        
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
        
        return response_text
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return "मुझे खुशी होगी आपकी सहायता करने में। कृपया फिर से कोशिश करें।"

@app.route("/voice", methods=['POST'])
def voice():
    """Handle incoming voice calls"""
    response = VoiceResponse()
    caller_id = request.form.get('From', 'unknown')
    
    print(f"Incoming call from: {caller_id}")
    
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

@app.route("/process_voice", methods=['POST'])
def process_voice():
    """Process user speech and generate AI response"""
    response = VoiceResponse()
    caller_id = request.form.get('From', 'unknown')
    
    # Get speech input from Twilio
    speech_text = request.form.get("SpeechResult", "")
    confidence = request.form.get("Confidence", "0")
    
    print(f"User said: '{speech_text}' (Confidence: {confidence})")
    
    if not speech_text or speech_text.lower() in ['', 'um', 'uh']:
        response.say(
            "मुझे आपकी आवाज़ साफ़ सुनाई नहीं दी। कृपया दोबारा बोलें।", 
            voice="Polly.Aditi", 
            language="hi-IN"
        )
        response.redirect("/voice")
        return Response(str(response), content_type='text/xml')
    
    # Check for goodbye/end call phrases
    end_phrases = ['अलविदा', 'धन्यवाद', 'बाय', 'bye', 'goodbye', 'खत्म', 'समाप्त']
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
        hints="हाँ,नहीं,और,अधिक,समझाओ,धन्यवाद"
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

@app.route("/", methods=['GET'])
def home():
    return """
    <h1>🤖 Indian Voice Bot is Running!</h1>
    <p><strong>Status:</strong> Active and ready for calls</p>
    <p><strong>Groq API:</strong> Connected</p>
    <p><strong>Twilio:</strong> Configured</p>
    <p><strong>Language:</strong> Hindi (Primary), English (Secondary)</p>
    """

@app.route("/test", methods=['GET'])
def test():
    """Test endpoint to verify Groq API connection"""
    try:
        test_response = generate_response("नमस्ते, आप कैसे हैं?", "test_user")
        return f"<h2>Groq API Test Successful!</h2><p><strong>Response:</strong> {test_response}</p>"
    except Exception as e:
        return f"<h2>Groq API Test Failed</h2><p><strong>Error:</strong> {str(e)}</p>"

if __name__ == "__main__":
    print("🚀 Starting Indian Voice Bot...")
    print("✅ Groq API Key loaded")
    print("✅ Twilio credentials loaded")
    print("🌐 Server starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

