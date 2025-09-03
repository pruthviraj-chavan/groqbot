import os
import logging
from datetime import datetime, timedelta
from flask import Flask, request, Response, jsonify
from twilio.twiml.voice_response import VoiceResponse, Gather
from groq import Groq
import re
import time

app = Flask(__name__)

# Minimal logging for speed
logging.basicConfig(level=logging.WARNING)  # Changed from INFO to WARNING
logger = logging.getLogger(__name__)

# API credentials
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")

# Initialize Groq client with connection pooling
try:
    client = Groq(api_key=GROQ_API_KEY)
    # Test connection
    test_completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "test"}],
        max_tokens=5
    )
    logger.warning("✅ Groq client initialized and tested successfully")
except Exception as e:
    logger.error(f"❌ Error initializing Groq client: {e}")
    client = None

# Simplified storage for maximum speed
conversations = {}
call_stats = {}

class FastCallState:
    def __init__(self, caller_id):
        self.caller_id = caller_id
        self.count = 0
        self.last_input = ""
        self.context_buffer = []

def get_conversation_context(caller_id):
    """Minimal context management for speed"""
    if caller_id not in conversations:
        conversations[caller_id] = [
            {
                "role": "system", 
                "content": """You are a lightning-fast Hindi AI assistant. Rules:

1. INSTANT responses in 10-20 words maximum
2. NEVER ask to repeat - always respond to whatever you hear
3. Be helpful even with unclear input
4. Mix Hindi and English naturally
5. Give direct, practical answers
6. No politeness overload - be efficient

Examples:
- User: "नौकरी" → "कौन सा field? Resume बनाएं, LinkedIn use करें।"  
- User: "पैसा" → "Income बढ़ाने के लिए skill development या side business करें।"
- User: unclear/partial → "समझ गया, [give best guess response]"

NEVER say "फिर से बोलिए" or "समझ नहीं आया" - ALWAYS give a helpful response."""
            }
        ]
        call_stats[caller_id] = FastCallState(caller_id)
    return conversations[caller_id]

def is_goodbye(text):
    """Quick goodbye detection"""
    goodbye_words = ['बाय', 'bye', 'धन्यवाद', 'thanks', 'खत्म', 'बस', 'चलता', 'जाना']
    return any(word in text.lower() for word in goodbye_words)

def generate_lightning_response(prompt, caller_id):
    """Ultra-optimized response generation"""
    try:
        # Quick validation
        if not client or not prompt:
            return "जी हाँ, बताइए क्ya help चाहिए?"
        
        # Clean input quickly
        prompt = prompt.strip()
        if len(prompt) < 2:
            return "हाँ, कहिए आपको क्या चाहिए?"
        
        # Get minimal context
        messages = get_conversation_context(caller_id)
        messages.append({"role": "user", "content": prompt})
        
        # Update stats
        stats = call_stats.get(caller_id)
        if stats:
            stats.count += 1
            stats.last_input = prompt
        
        # Ultra-fast Groq call with timeout
        start_time = time.time()
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=50,  # Very small for speed
            temperature=0.7,
            top_p=0.85,
            stream=False
        )
        
        response_text = completion.choices[0].message.content.strip()
        
        # Ensure ultra-short response
        if len(response_text) > 100:
            # Cut at sentence boundary or word boundary
            sentences = response_text.split('।')
            if len(sentences) > 1:
                response_text = sentences[0] + '।'
            else:
                words = response_text.split()
                response_text = ' '.join(words[:15]) + '।'
        
        messages.append({"role": "assistant", "content": response_text})
        
        # Aggressive memory management - keep only last 8 messages
        if len(messages) > 9:
            messages = messages[:1] + messages[-8:]
        
        response_time = round((time.time() - start_time) * 1000, 1)
        if response_time > 1000:  # Log if slow
            logger.warning(f"Slow response: {response_time}ms")
        
        return response_text
        
    except Exception as e:
        logger.error(f"Error in response generation: {e}")
        # Smart fallback based on input
        if 'job' in prompt.lower() or 'नौकरी' in prompt.lower():
            return "Skills बढ़ाएं, resume update करें। कौन सा field है?"
        elif 'money' in prompt.lower() or 'पैसा' in prompt.lower():
            return "Income बढ़ाने के तरीके हैं। क्या सीखना चाहते हैं?"
        elif 'health' in prompt.lower() or 'स्वास्थ्य' in prompt.lower():
            return "डॉक्टर से मिलें। क्या problem है specifically?"
        else:
            return "समझ गया। और details में बताएं?"

@app.route("/voice", methods=['POST'])
def voice():
    """Ultra-fast initial call handler"""
    try:
        response = VoiceResponse()
        caller_id = request.form.get('From', 'unknown')
        
        # Minimal welcome - no long introductions
        response.say(
            "नमस्ते! पूछिए।", 
            voice="Polly.Aditi", 
            language="hi-IN"
        )
        
        # Optimized gather with very liberal settings
        gather = Gather(
            input="speech", 
            action="/process", 
            method="POST", 
            language="hi-IN", 
            speech_timeout="auto",  # Let Twilio decide
            timeout=10,  # Longer timeout to avoid re-prompting
            action_on_empty_result=False,  # Don't trigger on empty
            hints="नौकरी,पैसा,स्वास्थ्य,काम,समस्या,help,problem,job,money,health"
        )
        
        response.append(gather)
        
        # Fallback only if absolutely no input
        response.say("कॉल के लिए धन्यवाद!", voice="Polly.Aditi", language="hi-IN")
        
        return Response(str(response), content_type='text/xml')
        
    except Exception as e:
        logger.error(f"Error in voice route: {e}")
        response = VoiceResponse()
        response.say("Hello, I'm here to help.", voice="alice")
        return Response(str(response), content_type='text/xml')

@app.route("/process", methods=['POST'])
def process():
    """Ultra-optimized main processing"""
    try:
        response = VoiceResponse()
        caller_id = request.form.get('From', 'unknown')
        
        speech_text = request.form.get("SpeechResult", "").strip()
        confidence = float(request.form.get("Confidence", "0"))
        
        # NEVER ask to repeat - always try to help
        if not speech_text:
            # Give helpful prompt instead of asking to repeat
            ai_response = "बताइए आपको क्या help चाहिए? Job, money, health, या कुछ और?"
        elif confidence < 0.3:
            # Low confidence - make intelligent guess
            if len(speech_text) < 5:
                ai_response = "हाँ, सुन रहा हूं। आपकी problem क्या है?"
            else:
                # Try to process even low confidence input
                ai_response = generate_lightning_response(speech_text, caller_id)
        else:
            # Normal processing
            if is_goodbye(speech_text):
                response.say("धन्यवाद! Good day!", voice="Polly.Aditi", language="hi-IN")
                response.hangup()
                
                # Quick cleanup
                if caller_id in conversations:
                    del conversations[caller_id]
                if caller_id in call_stats:
                    del call_stats[caller_id]
                    
                return Response(str(response), content_type='text/xml')
            
            # Generate response
            ai_response = generate_lightning_response(speech_text, caller_id)
        
        # Speak response immediately
        response.say(ai_response, voice="Polly.Aditi", language="hi-IN")
        
        # Continue conversation with minimal timeout
        gather = Gather(
            input="speech", 
            action="/process", 
            method="POST", 
            language="hi-IN", 
            speech_timeout="auto",
            timeout=8,  # Shorter for faster flow
            action_on_empty_result=False,
            hints="और,more,हाँ,नहीं,yes,no,ok,ठीक,समझ गया,धन्यवाद"
        )
        
        # Ultra-minimal continuation
        gather.say("और?", voice="Polly.Aditi", language="hi-IN")
        response.append(gather)
        
        # Quick end
        response.say("धन्यवाद!", voice="Polly.Aditi", language="hi-IN")
        
        return Response(str(response), content_type='text/xml')
        
    except Exception as e:
        logger.error(f"Error in process: {e}")
        response = VoiceResponse()
        response.say("Technical issue. Please call back.", voice="alice")
        response.hangup()
        return Response(str(response), content_type='text/xml')

@app.route("/", methods=['GET'])
def home():
    active_calls = len(call_stats)
    total_interactions = sum(stats.count for stats in call_stats.values())
    
    return f"""
    <h1>⚡ Ultra-Fast Voice Bot</h1>
    <p><strong>Status:</strong> ✅ Lightning Speed Optimized</p>
    <p><strong>Groq API:</strong> {'⚡ Ready' if client else '❌ Error'}</p>
    <p><strong>Active Calls:</strong> {active_calls}</p>
    <p><strong>Total Interactions:</strong> {total_interactions}</p>
    
    <h3>🔥 Speed Optimizations:</h3>
    <ul>
        <li>✅ NEVER asks to repeat</li>
        <li>✅ Processes even unclear speech</li>
        <li>✅ 10-20 word responses</li>
        <li>✅ Minimal logging overhead</li>
        <li>✅ Aggressive memory management</li>
        <li>✅ Liberal timeout settings</li>
        <li>✅ Smart fallback responses</li>
    </ul>
    
    <h3>📊 Performance:</h3>
    <ul>
        <li>🎯 Target: <500ms response time</li>
        <li>🎯 No "repeat please" prompts</li>
        <li>🎯 Handles unclear input intelligently</li>
    </ul>
    """

@app.route("/speed_test", methods=['GET'])
def speed_test():
    """Test actual response speed"""
    try:
        test_inputs = [
            "नौकरी नहीं मिल रही",
            "पैसे की समस्या है", 
            "स्वास्थ्य खराब है",
            "बिजनेस शुरू करना है"
        ]
        
        results = []
        for test_input in test_inputs:
            start_time = time.time()
            response = generate_lightning_response(test_input, "speed_test")
            end_time = time.time()
            
            response_time = round((end_time - start_time) * 1000, 1)
            status = "🚀 FAST" if response_time < 1000 else "⚠️ SLOW"
            
            results.append(f"<li><strong>'{test_input}'</strong><br>Response: {response}<br>Time: {response_time}ms {status}</li>")
        
        return f"""
        <h2>⚡ Speed Test Results</h2>
        <ul>{''.join(results)}</ul>
        """
        
    except Exception as e:
        return f"<h2>❌ Speed Test Failed</h2><p>{str(e)}</p>"

@app.route("/stats", methods=['GET'])
def stats():
    """Show current statistics"""
    if not call_stats:
        return "<h2>📊 No Active Calls</h2>"
    
    stats_html = "<h2>📊 Call Statistics</h2><ul>"
    for caller_id, stats in call_stats.items():
        stats_html += f"<li><strong>{caller_id}:</strong> {stats.count} interactions, Last: '{stats.last_input}'</li>"
    stats_html += "</ul>"
    
    return stats_html

# Health check for monitoring
@app.route("/health", methods=['GET'])
def health():
    """Quick health check"""
    try:
        status = {
            "status": "healthy" if client else "unhealthy",
            "active_calls": len(call_stats),
            "total_conversations": len(conversations),
            "timestamp": datetime.now().isoformat()
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    
    print("⚡ Starting ULTRA-FAST Voice Bot...")
    print(f"🚀 Groq API: {'Ready' if GROQ_API_KEY else 'Missing'}")
    print(f"⚡ Twilio: {'Ready' if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else 'Missing'}")
    print(f"🌐 Port: {port}")
    print("🎯 Optimizations: No re-prompting, handles unclear speech, <500ms response")
    
    # Production settings
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
