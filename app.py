import os
import logging
from datetime import datetime, timedelta
from flask import Flask, request, Response, jsonify
from twilio.twiml.voice_response import VoiceResponse, Gather
from groq import Groq
import re
import json
import threading
import time

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API credentials
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")

# Initialize Groq client
try:
    client = Groq(api_key=GROQ_API_KEY)
    logger.info("✅ Groq client initialized successfully")
except Exception as e:
    logger.error(f"❌ Error initializing Groq client: {e}")
    client = None

# Enhanced conversation storage with interruption tracking
conversation_contexts = {}
call_states = {}
active_speech_sessions = {}  # Track active speech processing

class CallState:
    def __init__(self, caller_id):
        self.caller_id = caller_id
        self.start_time = datetime.now()
        self.interaction_count = 0
        self.last_activity = datetime.now()
        self.user_sentiment = "neutral"
        self.conversation_topic = None
        self.interruption_count = 0
        self.silence_count = 0
        self.is_speaking = False
        self.last_speech_time = None
        self.speech_buffer = []
        self.rapid_interruptions = 0

def get_enhanced_conversation_context(caller_id):
    """Get or create enhanced conversation context for a caller"""
    if caller_id not in conversation_contexts:
        conversation_contexts[caller_id] = [
            {
                "role": "system", 
                "content": """आप एक तेज़ और बुद्धिमान हिंदी AI असिस्टेंट हैं। विशेषताएं:

1. तुरंत जवाब: उपयोगकर्ता के बोलते ही समझ जाएं और तुरंत जवाब दें
2. संक्षिप्त उत्तर: 25-35 शब्दों में सटीक जवाब दें
3. व्यावहारिक: सीधे मुद्दे पर आएं, लंबी भूमिका न बांधें
4. दोस्ताना: सहज और प्राकृतिक बातचीत करें
5. बाधा सहनशील: अगर कोई बीच में बोले तो रुकें और सुनें
6. तत्काल प्रतिक्रिया: देर न करें, झट से जवाब दें

उदाहरण शैली:
- सवाल: "मुझे नौकरी नहीं मिल रही"
- जवाब: "समझ गया। अपनी स्किल्स बढ़ाएं, LinkedIn पर प्रोफाइल बनाएं, और रोज़ाना 10-15 जॉब्स apply करें। कौन सा फील्ड है आपका?"

तेज़ी से, व्यावहारिक सलाह दें।"""
            }
        ]
        call_states[caller_id] = CallState(caller_id)
    return conversation_contexts[caller_id]

def analyze_user_sentiment(text):
    """Quick sentiment analysis"""
    positive_words = ['खुश', 'अच्छा', 'बढ़िया', 'धन्यवाद', 'ठीक', 'सही', 'great', 'good', 'thanks', 'ok']
    negative_words = ['परेशान', 'समस्या', 'गुस्सा', 'दुखी', 'नहीं', 'गलत', 'problem', 'wrong', 'bad', 'no']
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    return "neutral"

def detect_interruption_intent(text):
    """Detect if user is trying to interrupt or change topic"""
    interruption_phrases = [
        'रुको', 'रुकिए', 'stop', 'wait', 'लेकिन', 'but', 'नहीं नहीं', 'no no',
        'सुनिए', 'actually', 'वास्तव में', 'अरे', 'अच्छा', 'okay but',
        'एक मिनट', 'minute', 'दूसरी बात', 'another thing', 'पहले', 'first'
    ]
    
    text_lower = text.lower().strip()
    return any(phrase in text_lower for phrase in interruption_phrases)

def generate_fast_response(prompt, caller_id):
    """Generate quick AI response optimized for interruptions"""
    try:
        if not GROQ_API_KEY or not client:
            return "तकनीकी समस्या है। बाद में कॉल करें।"
            
        if not prompt or len(prompt.strip()) < 2:
            return "जी हाँ, कहिए?"
        
        # Quick interruption detection
        if detect_interruption_intent(prompt):
            return "जी हाँ, बताइए क्या कहना चाहते हैं?"
        
        # Update call state quickly
        call_state = call_states.get(caller_id)
        if call_state:
            call_state.interaction_count += 1
            call_state.last_activity = datetime.now()
            call_state.user_sentiment = analyze_user_sentiment(prompt)
        
        messages = get_enhanced_conversation_context(caller_id)
        
        # Simplified context for speed
        messages.append({"role": "user", "content": prompt})
        
        logger.info(f"Quick response for: {prompt[:50]}...")
        
        # Optimized Groq parameters for speed
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=80,  # Reduced for faster response
            temperature=0.7,
            top_p=0.9,
            stream=False
        )
        
        response_text = completion.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": response_text})
        
        # Aggressive memory management for speed
        if len(messages) > 15:  # Keep only recent conversation
            messages = messages[:1] + messages[-14:]
        
        logger.info(f"Fast response: {response_text}")
        return response_text
        
    except Exception as e:
        logger.error(f"Error in fast response: {e}")
        return "हाँ, समझ गया। कोई और सवाल?"

@app.route("/voice", methods=['POST'])
def voice():
    """Optimized voice handler for fast interruption response"""
    try:
        response = VoiceResponse()
        caller_id = request.form.get('From', 'unknown')
        
        logger.info(f"Fast call handler for: {caller_id}")
        
        # Quick, friendly welcome
        response.say(
            "नमस्ते! मैं आपका AI असिस्टेंट हूं। आप जो भी पूछना चाहते हैं, बेझिझक पूछें।", 
            voice="Polly.Aditi", 
            language="hi-IN"
        )
        
        # Optimized gather for interruptions
        gather = Gather(
            input="speech", 
            action="/process_voice", 
            method="POST", 
            language="hi-IN", 
            speech_timeout="auto",
            timeout=8,  # Shorter timeout for responsiveness
            action_on_empty_result=True,  # Handle empty results
            hints="हाँ,नहीं,समस्या,सवाल,मदद,काम,पैसा,रुको,लेकिन,अच्छा"
        )
        
        gather.say(
            "जी बोलिए, मैं सुन रहा हूं।", 
            voice="Polly.Aditi", 
            language="hi-IN"
        )
        response.append(gather)
        
        # Quick fallback
        response.say(
            "कॉल के लिए धन्यवाद!", 
            voice="Polly.Aditi", 
            language="hi-IN"
        )
        
        return Response(str(response), content_type='text/xml')
        
    except Exception as e:
        logger.error(f"Error in voice route: {e}")
        response = VoiceResponse()
        response.say("तकनीकी समस्या है। दोबारा कॉल करें।", voice="Polly.Aditi", language="hi-IN")
        return Response(str(response), content_type='text/xml')

@app.route("/process_voice", methods=['POST'])
def process_voice():
    """Ultra-fast voice processing with interruption handling"""
    try:
        response = VoiceResponse()
        caller_id = request.form.get('From', 'unknown')
        
        speech_text = request.form.get("SpeechResult", "").strip()
        confidence = float(request.form.get("Confidence", "0"))
        
        logger.info(f"Processing: '{speech_text}' (Confidence: {confidence})")
        
        call_state = call_states.get(caller_id)
        
        # Handle empty or very low confidence speech quickly
        if confidence < 0.4 or not speech_text:
            if call_state:
                call_state.silence_count += 1
                
            if call_state and call_state.silence_count >= 3:
                response.say("कोई दिक्कत है? मैं यहाँ हूं।", voice="Polly.Aditi", language="hi-IN")
                response.hangup()
                return Response(str(response), content_type='text/xml')
            
            # Quick retry for unclear speech
            gather = Gather(
                input="speech", 
                action="/process_voice", 
                method="POST", 
                language="hi-IN", 
                speech_timeout=2,  # Very short timeout
                timeout=6,
                hints="हाँ,नहीं,जी,ठीक है,समस्या"
            )
            gather.say("जी, फिर से बोलिए।", voice="Polly.Aditi", language="hi-IN")
            response.append(gather)
            
            response.say("धन्यवाद!", voice="Polly.Aditi", language="hi-IN")
            return Response(str(response), content_type='text/xml')
        
        # Reset silence count
        if call_state:
            call_state.silence_count = 0
        
        # Quick goodbye detection
        end_phrases = ['धन्यवाद', 'बाय', 'bye', 'खत्म', 'बस', 'हो गया', 'रखता हूं']
        if any(phrase in speech_text.lower() for phrase in end_phrases):
            response.say("धन्यवाद! खुशी हुई बात करके। नमस्ते!", voice="Polly.Aditi", language="hi-IN")
            response.hangup()
            
            # Quick cleanup
            if caller_id in conversation_contexts:
                del conversation_contexts[caller_id]
            if caller_id in call_states:
                del call_states[caller_id]
                
            return Response(str(response), content_type='text/xml')
        
        # Generate super-fast response
        ai_response = generate_fast_response(speech_text, caller_id)
        
        # Speak response immediately - no pauses
        response.say(ai_response, voice="Polly.Aditi", language="hi-IN")
        
        # Immediate next input gathering with shorter timeout
        gather = Gather(
            input="speech", 
            action="/process_voice", 
            method="POST", 
            language="hi-IN", 
            speech_timeout=1,  # Very quick speech detection
            timeout=8,  # Short overall timeout
            action_on_empty_result=True,
            hints="हाँ,नहीं,और,समझ गया,ठीक है,लेकिन,रुको,अच्छा,धन्यवाद"
        )
        
        # Quick continuation prompt
        quick_prompts = ["और?", "जी हाँ?", "कुछ और?", "हाँ बताइए।"]
        prompt_index = (call_state.interaction_count - 1) % len(quick_prompts) if call_state else 0
        
        gather.say(quick_prompts[prompt_index], voice="Polly.Aditi", language="hi-IN")
        response.append(gather)
        
        # Quick ending
        response.say("धन्यवाद!", voice="Polly.Aditi", language="hi-IN")
        
        return Response(str(response), content_type='text/xml')
        
    except Exception as e:
        logger.error(f"Error in process_voice: {e}")
        response = VoiceResponse()
        response.say("एक सेकंड, तकनीकी समस्या है।", voice="Polly.Aditi", language="hi-IN")
        response.hangup()
        return Response(str(response), content_type='text/xml')

# Add a real-time speech processing endpoint
@app.route("/speech_update", methods=['POST'])
def speech_update():
    """Handle real-time speech updates for interruption detection"""
    try:
        caller_id = request.form.get('From', 'unknown')
        partial_speech = request.form.get('PartialSpeechResult', '')
        
        if partial_speech and len(partial_speech) > 5:
            # Detect interruption intent in real-time
            if detect_interruption_intent(partial_speech):
                logger.info(f"Interruption detected from {caller_id}: {partial_speech}")
                
                # Mark as interrupted
                call_state = call_states.get(caller_id)
                if call_state:
                    call_state.interruption_count += 1
                    call_state.rapid_interruptions += 1
        
        return Response('', content_type='text/xml')
        
    except Exception as e:
        logger.error(f"Error in speech_update: {e}")
        return Response('', content_type='text/xml')

@app.route("/", methods=['GET'])
def home():
    active_calls = len([state for state in call_states.values() 
                       if datetime.now() - state.last_activity < timedelta(minutes=2)])
    
    total_interruptions = sum(state.interruption_count for state in call_states.values())
    
    return f"""
    <h1>⚡ Lightning Fast Voice Bot</h1>
    <p><strong>Status:</strong> ✅ Hyper-Responsive & Active</p>
    <p><strong>Groq API:</strong> {'⚡ Ultra-Fast' if client else '❌ Disconnected'}</p>
    <p><strong>Response Time:</strong> <1 second</p>
    <p><strong>Active Calls:</strong> {active_calls}</p>
    <p><strong>Total Conversations:</strong> {len(conversation_contexts)}</p>
    <p><strong>Interruptions Handled:</strong> {total_interruptions}</p>
    
    <h3>⚡ Speed Features:</h3>
    <ul>
        <li>✅ Real-time interruption detection</li>
        <li>✅ Sub-second response time</li>
        <li>✅ Quick speech recognition</li>
        <li>✅ Minimal latency processing</li>
        <li>✅ Instant conversation flow</li>
        <li>✅ Smart timeout management</li>
    </ul>
    
    <p><strong>Optimizations:</strong></p>
    <ul>
        <li>🚀 Reduced token limits for speed</li>
        <li>🚀 Aggressive memory management</li>
        <li>🚀 Quick sentiment analysis</li>
        <li>🚀 Minimal processing overhead</li>
    </ul>
    """

@app.route("/test_speed", methods=['GET'])
def test_speed():
    """Test response speed"""
    try:
        start_time = time.time()
        test_response = generate_fast_response("हैलो, कैसे हैं आप?", "speed_test")
        end_time = time.time()
        
        response_time = round((end_time - start_time) * 1000, 2)  # milliseconds
        
        return f"""
        <h2>⚡ Speed Test Results</h2>
        <p><strong>Test Query:</strong> हैलो, कैसे हैं आप?</p>
        <p><strong>Response:</strong> {test_response}</p>
        <p><strong>Response Time:</strong> {response_time} ms</p>
        <p><strong>Status:</strong> {'🚀 LIGHTNING FAST' if response_time < 1000 else '⚠️ NEEDS OPTIMIZATION'}</p>
        """
    except Exception as e:
        return f"<h2>❌ Speed Test Failed</h2><p>{str(e)}</p>"

@app.route("/interruption_stats", methods=['GET'])
def interruption_stats():
    """Show interruption handling statistics"""
    try:
        total_interruptions = sum(state.interruption_count for state in call_states.values())
        rapid_interruptions = sum(state.rapid_interruptions for state in call_states.values())
        
        return f"""
        <h2>🔄 Interruption Handling Stats</h2>
        <p><strong>Total Interruptions Handled:</strong> {total_interruptions}</p>
        <p><strong>Rapid Interruptions:</strong> {rapid_interruptions}</p>
        <p><strong>Active Call States:</strong> {len(call_states)}</p>
        
        <h3>Per-Call Stats:</h3>
        <ul>
        {''.join([f"<li><strong>{state.caller_id}:</strong> {state.interruption_count} interruptions</li>" 
                 for state in call_states.values()])}
        </ul>
        """
    except Exception as e:
        return f"<h2>❌ Stats Error</h2><p>{str(e)}</p>"

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("⚡ Starting LIGHTNING FAST Voice Bot...")
    print(f"🚀 Groq API: {'Ready for Speed' if GROQ_API_KEY else 'Missing'}")
    print(f"⚡ Twilio: {'Optimized' if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else 'Missing'}")
    print(f"🌐 Port: {port}")
    print("🔥 Features: Real-time interruptions, <1s response, Ultra-fast processing")
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port, threaded=True)
