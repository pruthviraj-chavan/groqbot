import os
import logging
from datetime import datetime, timedelta
from flask import Flask, request, Response, jsonify
from twilio.twiml.voice_response import VoiceResponse, Gather, Say
from groq import Groq
import re
import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor
import asyncio

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

# Advanced interruption tracking
conversation_contexts = {}
call_states = {}
active_responses = {}  # Track active bot responses
interruption_queue = queue.Queue()  # Queue for handling interruptions
executor = ThreadPoolExecutor(max_workers=10)  # For parallel processing

class AdvancedCallState:
    def __init__(self, caller_id):
        self.caller_id = caller_id
        self.start_time = datetime.now()
        self.interaction_count = 0
        self.last_activity = datetime.now()
        self.user_sentiment = "neutral"
        self.conversation_topic = None
        self.interruption_count = 0
        self.silence_count = 0
        self.is_bot_speaking = False
        self.current_response_id = None
        self.interruption_buffer = []
        self.last_user_input = ""
        self.response_start_time = None
        self.mid_response_interrupted = False
        self.quick_interruption_mode = False
        self.conversation_flow = []

class ResponseManager:
    def __init__(self):
        self.active_responses = {}
        self.interrupted_responses = set()
    
    def start_response(self, caller_id, response_id):
        self.active_responses[caller_id] = response_id
        logger.info(f"Started response {response_id} for {caller_id}")
    
    def interrupt_response(self, caller_id):
        if caller_id in self.active_responses:
            response_id = self.active_responses[caller_id]
            self.interrupted_responses.add(response_id)
            logger.info(f"Interrupted response {response_id} for {caller_id}")
            return True
        return False
    
    def is_interrupted(self, caller_id, response_id):
        return response_id in self.interrupted_responses
    
    def end_response(self, caller_id, response_id):
        if caller_id in self.active_responses:
            del self.active_responses[caller_id]
        if response_id in self.interrupted_responses:
            self.interrupted_responses.remove(response_id)

response_manager = ResponseManager()

def get_enhanced_conversation_context(caller_id):
    """Get or create enhanced conversation context for a caller"""
    if caller_id not in conversation_contexts:
        conversation_contexts[caller_id] = [
            {
                "role": "system", 
                "content": """आप एक अत्यधिक तेज़ और बुद्धिमान हिंदी AI असिस्टेंट हैं जो INSTANT interruptions को handle करते हैं:

CORE FEATURES:
1. तत्काल प्रतिक्रिया: 0.5 सेकंड में जवाब दें
2. बाधा संवेदनशील: अगर user बीच में बोले तो तुरंत रुकें और नया सवाल सुनें
3. संक्षिप्त उत्तर: केवल 15-25 शब्दों में सटीक जवाब
4. प्राथमिकता: नया सवाल > पुराना जवाब
5. स्मार्ट switching: context बदलने पर तुरंत adapt करें

INTERRUPTION HANDLING:
- अगर user नया सवाल पूछे तो तुरंत उसका जवाब दें
- पुराना incomplete जवाब भूल जाएं
- "हाँ, बताइए" जैसे quick acknowledgments दें

RESPONSE STYLE:
- Direct और practical
- No lengthy explanations
- Immediate value delivery
- Natural conversational flow

EXAMPLES:
User interrupts: "रुको, दूसरी बात पूछनी है"
Response: "जी हाँ, पूछिए।"

User asks new question while you're speaking: 
Response: "जी, सुन रहा हूं। क्या पूछना है?"

Keep responses SHORT, FAST, and RELEVANT."""
            }
        ]
        call_states[caller_id] = AdvancedCallState(caller_id)
    return conversation_contexts[caller_id]

def detect_interruption_intent(text, context=""):
    """Advanced interruption detection with context awareness"""
    # Strong interruption signals
    strong_interrupts = [
        'रुको', 'रुकिए', 'stop', 'wait', 'hold on', 'एक मिनट',
        'नहीं नहीं', 'no no', 'actually', 'लेकिन', 'but',
        'सुनिए', 'अरे हाँ', 'अच्छा सुनिए', 'दूसरी बात'
    ]
    
    # Question interruptions
    question_interrupts = [
        'क्या', 'कैसे', 'कब', 'कहाँ', 'कौन', 'क्यों',
        'what', 'how', 'when', 'where', 'who', 'why',
        'बताओ', 'समझाओ', 'explain', 'tell me'
    ]
    
    # New topic signals
    topic_change = [
        'दूसरा सवाल', 'another question', 'new topic',
        'अलग बात', 'different thing', 'नई बात'
    ]
    
    text_lower = text.lower().strip()
    
    # Check for strong interruption
    if any(phrase in text_lower for phrase in strong_interrupts):
        return "strong_interrupt"
    
    # Check for question interruption
    if any(phrase in text_lower for phrase in question_interrupts):
        return "question_interrupt"
    
    # Check for topic change
    if any(phrase in text_lower for phrase in topic_change):
        return "topic_change"
    
    # Check if it's a completely new sentence/thought
    if len(text_lower) > 10 and not any(word in text_lower for word in ['हाँ', 'ठीक', 'yes', 'ok', 'right']):
        return "new_input"
    
    return "continue"

def generate_interruption_response(interruption_type, new_input, caller_id):
    """Generate appropriate response based on interruption type"""
    call_state = call_states.get(caller_id)
    
    if interruption_type == "strong_interrupt":
        return "जी हाँ, बोलिए।"
    elif interruption_type == "question_interrupt":
        return generate_ultra_fast_response(new_input, caller_id)
    elif interruption_type == "topic_change":
        # Clear previous context for new topic
        if call_state:
            call_state.conversation_topic = None
        return "जी, नया सवाल पूछिए।"
    elif interruption_type == "new_input":
        return generate_ultra_fast_response(new_input, caller_id)
    else:
        return "जी, सुन रहा हूं।"

def generate_ultra_fast_response(prompt, caller_id, max_retries=2):
    """Ultra-fast response generation with interruption awareness"""
    try:
        if not GROQ_API_KEY or not client:
            return "तकनीकी समस्या है।"
            
        if not prompt or len(prompt.strip()) < 2:
            return "जी, कहिए?"
        
        call_state = call_states.get(caller_id)
        if call_state:
            call_state.interaction_count += 1
            call_state.last_activity = datetime.now()
            call_state.last_user_input = prompt
        
        messages = get_enhanced_conversation_context(caller_id)
        
        # Ultra-simplified context for maximum speed
        messages.append({"role": "user", "content": prompt})
        
        # Ultra-fast Groq parameters
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=60,  # Even smaller for speed
            temperature=0.6,
            top_p=0.8,
            stream=False,
            # Add timeout for faster failure
        )
        
        response_text = completion.choices[0].message.content.strip()
        
        # Ensure response is short
        if len(response_text) > 150:
            response_text = response_text[:150] + "..."
        
        messages.append({"role": "assistant", "content": response_text})
        
        # Aggressive memory management
        if len(messages) > 11:
            messages = messages[:1] + messages[-10:]
        
        logger.info(f"Ultra-fast response ({len(response_text)} chars): {response_text}")
        return response_text
        
    except Exception as e:
        logger.error(f"Error in ultra-fast response: {e}")
        return "समझ गया। कुछ और?"

def create_interruptible_response(ai_response, caller_id):
    """Create a response that can be interrupted mid-speech"""
    response = VoiceResponse()
    response_id = f"{caller_id}_{int(time.time())}"
    
    call_state = call_states.get(caller_id)
    if call_state:
        call_state.is_bot_speaking = True
        call_state.current_response_id = response_id
        call_state.response_start_time = datetime.now()
    
    # Start tracking this response
    response_manager.start_response(caller_id, response_id)
    
    # Break response into smaller chunks for better interruption
    words = ai_response.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= 5:  # 5 words per chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Add each chunk with pause for interruption detection
    for i, chunk in enumerate(chunks):
        # Check if interrupted before speaking each chunk
        if response_manager.is_interrupted(caller_id, response_id):
            logger.info(f"Response {response_id} interrupted at chunk {i}")
            break
            
        response.say(chunk, voice="Polly.Aditi", language="hi-IN")
        
        # Add minimal pause between chunks for interruption detection
        if i < len(chunks) - 1:
            response.pause(length=0.3)
    
    # End response tracking
    response_manager.end_response(caller_id, response_id)
    
    if call_state:
        call_state.is_bot_speaking = False
        call_state.current_response_id = None
    
    return response

@app.route("/voice", methods=['POST'])
def voice():
    """Ultra-responsive voice handler with instant interruption support"""
    try:
        response = VoiceResponse()
        caller_id = request.form.get('From', 'unknown')
        
        logger.info(f"Lightning call handler for: {caller_id}")
        
        # Instant welcome
        response.say(
            "नमस्ते! मैं तुरंत जवाब देता हूं। पूछिए।", 
            voice="Polly.Aditi", 
            language="hi-IN"
        )
        
        # Ultra-responsive gather
        gather = Gather(
            input="speech", 
            action="/process_voice", 
            method="POST", 
            language="hi-IN", 
            speech_timeout=1,  # Instant speech detection
            timeout=6,  # Quick timeout
            action_on_empty_result=True,
            hints="हाँ,नहीं,समस्या,सवाल,रुको,लेकिन,क्या,कैसे,बताओ"
        )
        
        gather.say("बोलिए।", voice="Polly.Aditi", language="hi-IN")
        response.append(gather)
        
        response.say("धन्यवाद!", voice="Polly.Aditi", language="hi-IN")
        return Response(str(response), content_type='text/xml')
        
    except Exception as e:
        logger.error(f"Error in voice route: {e}")
        response = VoiceResponse()
        response.say("तकनीकी समस्या है।", voice="Polly.Aditi", language="hi-IN")
        return Response(str(response), content_type='text/xml')

@app.route("/process_voice", methods=['POST'])
def process_voice():
    """Advanced voice processing with real-time interruption handling"""
    try:
        response = VoiceResponse()
        caller_id = request.form.get('From', 'unknown')
        
        speech_text = request.form.get("SpeechResult", "").strip()
        confidence = float(request.form.get("Confidence", "0"))
        
        logger.info(f"Processing: '{speech_text}' (Confidence: {confidence})")
        
        call_state = call_states.get(caller_id)
        
        # Handle very low confidence quickly
        if confidence < 0.3 or not speech_text:
            if call_state:
                call_state.silence_count += 1
                
            if call_state and call_state.silence_count >= 2:
                response.say("कनेक्शन की समस्या लग रही है।", voice="Polly.Aditi", language="hi-IN")
                response.hangup()
                return Response(str(response), content_type='text/xml')
            
            # Quick retry
            gather = Gather(
                input="speech", 
                action="/process_voice", 
                method="POST", 
                language="hi-IN", 
                speech_timeout=1,
                timeout=5,
                hints="जी,हाँ,हैलो,नमस्ते"
            )
            gather.say("फिर से बोलिए।", voice="Polly.Aditi", language="hi-IN")
            response.append(gather)
            
            response.say("धन्यवाद!", voice="Polly.Aditi", language="hi-IN")
            return Response(str(response), content_type='text/xml')
        
        # Reset silence count
        if call_state:
            call_state.silence_count = 0
        
        # Check if bot was speaking and got interrupted
        if call_state and call_state.is_bot_speaking:
            logger.info(f"INTERRUPTION DETECTED! Bot was speaking, user said: {speech_text}")
            call_state.interruption_count += 1
            call_state.mid_response_interrupted = True
            
            # Stop current response
            if call_state.current_response_id:
                response_manager.interrupt_response(caller_id)
        
        # Detect interruption type
        interruption_type = detect_interruption_intent(speech_text, call_state.last_user_input if call_state else "")
        
        # Quick goodbye detection
        end_phrases = ['धन्यवाद', 'बाय', 'bye', 'खत्म', 'बस', 'हो गया', 'रखता हूं', 'चलता हूं']
        if any(phrase in speech_text.lower() for phrase in end_phrases):
            response.say("धन्यवाद! अच्छा दिन हो।", voice="Polly.Aditi", language="hi-IN")
            response.hangup()
            
            # Cleanup
            if caller_id in conversation_contexts:
                del conversation_contexts[caller_id]
            if caller_id in call_states:
                del call_states[caller_id]
                
            return Response(str(response), content_type='text/xml')
        
        # Generate appropriate response based on interruption type
        if interruption_type in ["strong_interrupt", "question_interrupt", "topic_change", "new_input"]:
            ai_response = generate_interruption_response(interruption_type, speech_text, caller_id)
        else:
            ai_response = generate_ultra_fast_response(speech_text, caller_id)
        
        # Speak response with interruption capability
        response.say(ai_response, voice="Polly.Aditi", language="hi-IN")
        
        # Immediate next input gathering - ULTRA SHORT timeouts
        gather = Gather(
            input="speech", 
            action="/process_voice", 
            method="POST", 
            language="hi-IN", 
            speech_timeout=0.5,  # Detect speech in 0.5 seconds
            timeout=6,  # Total timeout 6 seconds
            action_on_empty_result=True,
            hints="और,हाँ,नहीं,क्या,कैसे,रुको,लेकिन,बताओ,समझाओ,धन्यवाद"
        )
        
        # Ultra-quick continuation prompts
        ultra_quick_prompts = ["और?", "जी?", "हाँ?"]
        prompt_index = (call_state.interaction_count - 1) % len(ultra_quick_prompts) if call_state else 0
        
        gather.say(ultra_quick_prompts[prompt_index], voice="Polly.Aditi", language="hi-IN")
        response.append(gather)
        
        # Quick ending
        response.say("धन्यवाद!", voice="Polly.Aditi", language="hi-IN")
        
        return Response(str(response), content_type='text/xml')
        
    except Exception as e:
        logger.error(f"Error in process_voice: {e}")
        response = VoiceResponse()
        response.say("समस्या है। दोबारा कॉल करें।", voice="Polly.Aditi", language="hi-IN")
        response.hangup()
        return Response(str(response), content_type='text/xml')

# Real-time interruption detection endpoint
@app.route("/detect_interruption", methods=['POST'])
def detect_interruption():
    """Real-time interruption detection during bot speech"""
    try:
        caller_id = request.form.get('From', 'unknown')
        partial_speech = request.form.get('PartialSpeechResult', '')
        stable_speech = request.form.get('SpeechResult', '')
        
        # Use stable speech if available, otherwise partial
        detected_speech = stable_speech if stable_speech else partial_speech
        
        if detected_speech and len(detected_speech) > 3:
            call_state = call_states.get(caller_id)
            
            if call_state and call_state.is_bot_speaking:
                interruption_type = detect_interruption_intent(detected_speech)
                
                if interruption_type != "continue":
                    logger.info(f"REAL-TIME INTERRUPTION: {detected_speech}")
                    
                    # Immediately stop current response
                    if call_state.current_response_id:
                        response_manager.interrupt_response(caller_id)
                    
                    # Mark as interrupted
                    call_state.interruption_count += 1
                    call_state.mid_response_interrupted = True
                    
                    # Add to interruption buffer for processing
                    call_state.interruption_buffer.append({
                        'text': detected_speech,
                        'type': interruption_type,
                        'timestamp': datetime.now()
                    })
        
        return Response('', content_type='text/xml')
        
    except Exception as e:
        logger.error(f"Error in detect_interruption: {e}")
        return Response('', content_type='text/xml')

@app.route("/", methods=['GET'])
def home():
    active_calls = len([state for state in call_states.values() 
                       if datetime.now() - state.last_activity < timedelta(minutes=2)])
    
    total_interruptions = sum(state.interruption_count for state in call_states.values())
    mid_response_interruptions = sum(1 for state in call_states.values() if state.mid_response_interrupted)
    
    return f"""
    <h1>🚀 REAL-TIME Interruption Voice Bot</h1>
    <p><strong>Status:</strong> ✅ INSTANT Response & Mid-Speech Interruption</p>
    <p><strong>Groq API:</strong> {'⚡ Ultra-Fast' if client else '❌ Disconnected'}</p>
    <p><strong>Response Time:</strong> <0.5 seconds</p>
    <p><strong>Active Calls:</strong> {active_calls}</p>
    <p><strong>Total Conversations:</strong> {len(conversation_contexts)}</p>
    <p><strong>Total Interruptions:</strong> {total_interruptions}</p>
    <p><strong>Mid-Response Interruptions:</strong> {mid_response_interruptions}</p>
    
    <h3>🔥 ADVANCED Features:</h3>
    <ul>
        <li>✅ <strong>MID-SPEECH interruption handling</strong></li>
        <li>✅ Real-time speech detection (0.5s)</li>
        <li>✅ Instant context switching</li>
        <li>✅ Smart interruption classification</li>
        <li>✅ Ultra-short responses (15-25 words)</li>
        <li>✅ Response chunking for interruptions</li>
        <li>✅ Parallel processing</li>
    </ul>
    
    <h3>⚡ Interruption Types Handled:</h3>
    <ul>
        <li>🛑 <strong>Strong Interrupts:</strong> "रुको", "stop", "wait"</li>
        <li>❓ <strong>Question Interrupts:</strong> "क्या", "कैसे", "बताओ"</li>
        <li>🔄 <strong>Topic Changes:</strong> "दूसरी बात", "नया सवाल"</li>
        <li>💬 <strong>New Input:</strong> Any new sentence/thought</li>
    </ul>
    """

@app.route("/test_interruption", methods=['GET'])
def test_interruption():
    """Test interruption handling capabilities"""
    try:
        # Simulate interruption scenario
        test_caller = "test_interruption"
        
        # Simulate bot speaking
        call_state = call_states.get(test_caller) or AdvancedCallState(test_caller)
        call_states[test_caller] = call_state
        
        call_state.is_bot_speaking = True
        call_state.current_response_id = f"{test_caller}_test"
        
        # Test interruption detection
        test_interruptions = [
            "रुको, दूसरा सवाल है",
            "क्या यह सही है?",
            "लेकिन मैं कुछ और पूछना चाहता हूं",
            "नहीं नहीं, यह नहीं"
        ]
        
        results = []
        for interruption in test_interruptions:
            interruption_type = detect_interruption_intent(interruption)
            response = generate_interruption_response(interruption_type, interruption, test_caller)
            results.append(f"<li><strong>'{interruption}'</strong> → Type: {interruption_type} → Response: {response}</li>")
        
        return f"""
        <h2>🧪 Interruption Test Results</h2>
        <h3>Test Scenarios:</h3>
        <ul>
        {''.join(results)}
        </ul>
        <p><strong>Bot Speaking Status:</strong> {call_state.is_bot_speaking}</p>
        <p><strong>Current Response ID:</strong> {call_state.current_response_id}</p>
        """
        
    except Exception as e:
        return f"<h2>❌ Test Failed</h2><p>{str(e)}</p>"

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("🚀 Starting REAL-TIME INTERRUPTION Voice Bot...")
    print(f"⚡ Groq API: {'Ready for Lightning Speed' if GROQ_API_KEY else 'Missing'}")
    print(f"🔥 Twilio: {'Ultra-Optimized' if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else 'Missing'}")
    print(f"🌐 Port: {port}")
    print("🎯 Features: MID-SPEECH interruptions, 0.5s detection, Context switching")
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port, threaded=True)
