import os
import logging
from datetime import datetime, timedelta
from flask import Flask, request, Response, jsonify
from twilio.twiml.voice_response import VoiceResponse, Gather
from groq import Groq
import re
import json

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

# Enhanced conversation context storage with metadata
conversation_contexts = {}
call_states = {}

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
        
def get_enhanced_conversation_context(caller_id):
    """Get or create enhanced conversation context for a caller"""
    if caller_id not in conversation_contexts:
        conversation_contexts[caller_id] = [
            {
                "role": "system", 
                "content": """आप एक अत्यधिक बुद्धिमान और सहायक हिंदी AI असिस्टेंट हैं। आप निम्नलिखित विशेषताओं के साथ काम करते हैं:

1. भाषा: मुख्यतः हिंदी में जवाब दें, लेकिन अंग्रेजी भी समझें और उसमें भी जवाब दे सकते हैं
2. शैली: दोस्ताना, सहायक, और व्यावहारिक
3. उत्तर की लंबाई: फोन कॉल के लिए 40-60 शब्दों में विस्तृत लेकिन संक्षिप्त जवाब दें
4. समझदारी: उपयोगकर्ता के मूड और संदर्भ को समझें
5. निरंतरता: पिछली बातचीत को याद रखें
6. बाधाएं: यदि उपयोगकर्ता बीच में बोले तो धैर्य रखें और उन्हें पूरा बोलने दें
7. मदद: व्यावहारिक सुझाव और समाधान दें
8. भावना: उपयोगकर्ता की भावनाओं को समझें और उसके अनुसार जवाब दें

कुछ उदाहरण स्थितियां:
- अगर कोई परेशान है तो सांत्वना दें
- तकनीकी सवाल में व्यावहारिक मदद करें  
- व्यक्तिगत सलाह में संवेदनशीलता दिखाएं
- जटिल विषयों को सरल भाषा में समझाएं"""
            }
        ]
        call_states[caller_id] = CallState(caller_id)
    return conversation_contexts[caller_id]

def analyze_user_sentiment(text):
    """Analyze user sentiment from speech"""
    positive_words = ['खुश', 'अच्छा', 'बढ़िया', 'धन्यवाद', 'great', 'good', 'happy', 'thanks']
    negative_words = ['परेशान', 'समस्या', 'गुस्सा', 'दुखी', 'problem', 'angry', 'sad', 'upset']
    
    text_lower = text.lower()
    
    if any(word in text_lower for word in positive_words):
        return "positive"
    elif any(word in text_lower for word in negative_words):
        return "negative"
    else:
        return "neutral"

def extract_topic(text):
    """Extract conversation topic from user input"""
    topics = {
        'health': ['स्वास्थ्य', 'बीमारी', 'दवा', 'डॉक्टर', 'health', 'medicine', 'doctor'],
        'technology': ['कंप्यूटर', 'मोबाइल', 'इंटरनेट', 'ऐप', 'computer', 'mobile', 'app', 'tech'],
        'work': ['काम', 'नौकरी', 'ऑफिस', 'व्यापार', 'work', 'job', 'office', 'business'],
        'family': ['परिवार', 'बच्चे', 'पति', 'पत्नी', 'family', 'children', 'husband', 'wife'],
        'education': ['पढ़ाई', 'स्कूल', 'कॉलेज', 'शिक्षा', 'study', 'school', 'college', 'education']
    }
    
    text_lower = text.lower()
    for topic, keywords in topics.items():
        if any(keyword in text_lower for keyword in keywords):
            return topic
    return 'general'

def generate_enhanced_response(prompt, caller_id):
    """Generate enhanced AI response with context awareness"""
    try:
        if not GROQ_API_KEY or not client:
            return "तकनीकी समस्या है। कृपया बाद में कोशिश करें।"
            
        if not prompt or prompt.strip() == "":
            return "मुझे आपका सवाल समझ नहीं आया। कृपया दोबारा स्पष्ट रूप से बोलें।"
        
        # Update call state
        call_state = call_states.get(caller_id)
        if call_state:
            call_state.interaction_count += 1
            call_state.last_activity = datetime.now()
            call_state.user_sentiment = analyze_user_sentiment(prompt)
            if not call_state.conversation_topic:
                call_state.conversation_topic = extract_topic(prompt)
        
        messages = get_enhanced_conversation_context(caller_id)
        
        # Add context information to the prompt
        context_info = f"[संदर्भ: बातचीत #{call_state.interaction_count if call_state else 1}, भावना: {call_state.user_sentiment if call_state else 'neutral'}]"
        enhanced_prompt = f"{context_info}\nउपयोगकर्ता: {prompt}"
        
        messages.append({"role": "user", "content": enhanced_prompt})
        
        logger.info(f"Sending to Groq: {prompt} (Caller: {caller_id})")
        
        # Generate response with higher token limit for detailed answers
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=150,  # Increased for more detailed responses
            temperature=0.8,  # Slightly more creative
            top_p=0.9,
            stream=False
        )
        
        response_text = completion.choices[0].message.content.strip()
        
        # Clean up response if it contains context info
        response_text = re.sub(r'\[संदर्भ:.*?\]', '', response_text).strip()
        
        messages.append({"role": "assistant", "content": response_text})
        
        # Limit conversation history to last 10 exchanges to manage memory
        if len(messages) > 21:  # 1 system + 20 conversation messages
            messages = messages[:1] + messages[-20:]
        
        logger.info(f"AI response: {response_text}")
        return response_text
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "मुझे खुशी होगी आपकी सहायता करने में। कृपया अपना सवाल फिर से बताएं।"

@app.route("/voice", methods=['POST'])
def voice():
    """Enhanced voice handler with better interruption management"""
    try:
        response = VoiceResponse()
        caller_id = request.form.get('From', 'unknown')
        
        logger.info(f"Incoming call from: {caller_id}")
        
        # Enhanced welcome message
        response.say(
            "नमस्ते! मैं आपका व्यक्तिगत AI सहायक हूं। मैं आपकी किसी भी समस्या में मदद कर सकता हूं। आप हिंदी या अंग्रेजी में बात कर सकते हैं।", 
            voice="Polly.Aditi", 
            language="hi-IN"
        )
        
        # Enhanced gather with better parameters for interruption handling
        gather = Gather(
            input="speech", 
            action="/process_voice", 
            method="POST", 
            language="hi-IN", 
            speech_timeout="auto",
            timeout=10,  # Increased timeout
            partial_result_callback="/partial_result",  # For handling interruptions
            enhanced=True,  # Enable enhanced speech recognition
            hints="हाँ,नहीं,धन्यवाद,नमस्ते,समस्या,सवाल,मदद,काम,परिवार,स्वास्थ्य,पैसा,शिक्षा,तकनीक"
        )
        
        gather.say(
            "मैं आपकी बात सुन रहा हूं। आप अपनी समस्या या सवाल विस्तार से बताएं। आराम से बोलें, मैं धैर्य से सुनूंगा।", 
            voice="Polly.Aditi", 
            language="hi-IN"
        )
        response.append(gather)
        
        # Enhanced fallback
        response.say(
            "लगता है आपकी आवाज़ नहीं आई। कृपया अच्छे नेटवर्क के साथ दोबारा कॉल करें। धन्यवाद!", 
            voice="Polly.Aditi", 
            language="hi-IN"
        )
        
        return Response(str(response), content_type='text/xml')
        
    except Exception as e:
        logger.error(f"Error in voice route: {e}")
        response = VoiceResponse()
        response.say(
            "तकनीकी समस्या आई है। कृपया कुछ देर बाद कोशिश करें।", 
            voice="Polly.Aditi", 
            language="hi-IN"
        )
        return Response(str(response), content_type='text/xml')

@app.route("/partial_result", methods=['POST'])
def partial_result():
    """Handle partial speech results for better interruption management"""
    try:
        caller_id = request.form.get('From', 'unknown')
        partial_text = request.form.get('UnstableSpeechResult', '')
        
        logger.info(f"Partial result from {caller_id}: {partial_text}")
        
        call_state = call_states.get(caller_id)
        if call_state:
            call_state.last_activity = datetime.now()
        
        # Return empty response to continue listening
        return Response('', content_type='text/xml')
        
    except Exception as e:
        logger.error(f"Error in partial_result: {e}")
        return Response('', content_type='text/xml')

@app.route("/process_voice", methods=['POST'])
def process_voice():
    """Enhanced voice processing with better conversation flow"""
    try:
        response = VoiceResponse()
        caller_id = request.form.get('From', 'unknown')
        
        speech_text = request.form.get("SpeechResult", "").strip()
        confidence = float(request.form.get("Confidence", "0"))
        
        logger.info(f"User said: '{speech_text}' (Confidence: {confidence})")
        
        call_state = call_states.get(caller_id)
        
        # Handle low confidence or empty speech
        if confidence < 0.6 or not speech_text or len(speech_text) < 3:
            if call_state:
                call_state.silence_count += 1
                
            if call_state and call_state.silence_count >= 2:
                response.say(
                    "लगता है आपको बोलने में कोई दिक्कत हो रही है। क्या आप चाहेंगे कि मैं आपको कॉल बैक करूं?", 
                    voice="Polly.Aditi", 
                    language="hi-IN"
                )
                response.hangup()
                return Response(str(response), content_type='text/xml')
            
            response.say(
                "मुझे आपकी बात साफ़ सुनाई नहीं दी। कृपया थोड़ा धीरे और स्पष्ट रूप से बोलें।", 
                voice="Polly.Aditi", 
                language="hi-IN"
            )
            
            gather = Gather(
                input="speech", 
                action="/process_voice", 
                method="POST", 
                language="hi-IN", 
                speech_timeout="auto",
                timeout=12,
                enhanced=True,
                hints="हाँ,नहीं,ठीक है,समझ गया,दोबारा,फिर से"
            )
            gather.say("मैं फिर से सुन रहा हूं। कृपया बोलें।", voice="Polly.Aditi", language="hi-IN")
            response.append(gather)
            
            response.say("धन्यवाद। अच्छा दिन हो!", voice="Polly.Aditi", language="hi-IN")
            return Response(str(response), content_type='text/xml')
        
        # Reset silence count on successful speech
        if call_state:
            call_state.silence_count = 0
        
        # Enhanced goodbye detection
        end_phrases = [
            'अलविदा', 'धन्यवाद', 'बाय', 'bye', 'goodbye', 'खत्म', 'समाप्त', 
            'बंद करो', 'रुको', 'बस', 'हो गया', 'enough', 'stop', 'end call',
            'कॉल बंद', 'छोड़ो', 'जाना है', 'काम निपट गया'
        ]
        
        if any(phrase in speech_text.lower() for phrase in end_phrases):
            sentiment_response = "बहुत बढ़िया!" if call_state and call_state.user_sentiment == "positive" else "कोई बात नहीं।"
            
            response.say(
                f"{sentiment_response} बातचीत के लिए धन्यवाद! आपकी समस्या का समाधान हो गया हो तो खुशी की बात है। फिर कभी जरूरत हो तो कॉल करिएगा। नमस्ते!", 
                voice="Polly.Aditi", 
                language="hi-IN"
            )
            response.hangup()
            
            # Clean up conversation context
            if caller_id in conversation_contexts:
                del conversation_contexts[caller_id]
            if caller_id in call_states:
                del call_states[caller_id]
                
            return Response(str(response), content_type='text/xml')
        
        # Generate enhanced AI response
        ai_response = generate_enhanced_response(speech_text, caller_id)
        
        # Speak the AI response with appropriate pausing
        response.say(ai_response, voice="Polly.Aditi", language="hi-IN")
        
        # Add a brief pause
        response.pause(length=1)
        
        # Continue conversation with contextual prompting
        continuation_prompts = [
            "क्या इससे आपकी मदद हुई? कोई और सवाल है?",
            "क्या आपको और कुछ जानना है इस बारे में?",
            "कोई और समस्या है जिसमें मैं आपकी मदद कर सकूं?",
            "क्या आपको कुछ और पूछना है?"
        ]
        
        prompt_index = (call_state.interaction_count - 1) % len(continuation_prompts) if call_state else 0
        continuation_prompt = continuation_prompts[prompt_index]
        
        gather = Gather(
            input="speech", 
            action="/process_voice", 
            method="POST", 
            language="hi-IN", 
            speech_timeout="auto",
            timeout=15,  # Longer timeout for thinking
            enhanced=True,
            hints="हाँ,नहीं,और,अधिक,समझाओ,विस्तार,उदाहरण,धन्यवाद,अलविदा,बस,हो गया"
        )
        
        gather.say(continuation_prompt, voice="Polly.Aditi", language="hi-IN")
        response.append(gather)
        
        # Graceful ending
        response.say(
            "बातचीत के लिए बहुत-बहुत धन्यवाद। आपका दिन मंगलमय हो। नमस्ते!", 
            voice="Polly.Aditi", 
            language="hi-IN"
        )
        
        return Response(str(response), content_type='text/xml')
        
    except Exception as e:
        logger.error(f"Error in process_voice route: {e}")
        response = VoiceResponse()
        response.say(
            "मुझे तकनीकी समस्या आई है। कृपया दोबारा कॉल करें।", 
            voice="Polly.Aditi", 
            language="hi-IN"
        )
        response.hangup()
        return Response(str(response), content_type='text/xml')

@app.route("/", methods=['GET'])
def home():
    active_calls = len([state for state in call_states.values() 
                       if datetime.now() - state.last_activity < timedelta(minutes=5)])
    
    return f"""
    <h1>🤖 Advanced Indian Voice Bot</h1>
    <p><strong>Status:</strong> ✅ Active and Enhanced</p>
    <p><strong>Groq API:</strong> {'✅ Connected' if client else '❌ Disconnected'}</p>
    <p><strong>Features:</strong> Advanced conversation, interruption handling, sentiment analysis</p>
    <p><strong>Active Calls:</strong> {active_calls}</p>
    <p><strong>Total Conversations:</strong> {len(conversation_contexts)}</p>
    <p><strong>Language Support:</strong> हिंदी (Primary), English (Secondary)</p>
    
    <h3>🎯 New Features:</h3>
    <ul>
        <li>✅ Better interruption handling</li>
        <li>✅ Longer, more detailed responses</li>
        <li>✅ Sentiment analysis</li>
        <li>✅ Context awareness</li>
        <li>✅ Topic detection</li>
        <li>✅ Enhanced conversation flow</li>
    </ul>
    
    <p><strong>Endpoints:</strong></p>
    <ul>
        <li><a href="/test">Test Groq API</a></li>
        <li><a href="/debug">Debug Info</a></li>
        <li><a href="/analytics">Call Analytics</a></li>
    </ul>
    """

@app.route("/analytics", methods=['GET'])
def analytics():
    """Analytics dashboard for call insights"""
    try:
        total_calls = len(call_states)
        active_calls = len([state for state in call_states.values() 
                           if datetime.now() - state.last_activity < timedelta(minutes=5)])
        
        sentiment_stats = {}
        topic_stats = {}
        
        for state in call_states.values():
            # Sentiment stats
            sentiment = state.user_sentiment
            sentiment_stats[sentiment] = sentiment_stats.get(sentiment, 0) + 1
            
            # Topic stats
            topic = state.conversation_topic or 'general'
            topic_stats[topic] = topic_stats.get(topic, 0) + 1
        
        return f"""
        <h2>📊 Call Analytics Dashboard</h2>
        
        <h3>📈 Overview</h3>
        <p><strong>Total Calls:</strong> {total_calls}</p>
        <p><strong>Active Calls:</strong> {active_calls}</p>
        <p><strong>Total Conversations:</strong> {len(conversation_contexts)}</p>
        
        <h3>😊 Sentiment Analysis</h3>
        <ul>
        {''.join([f"<li><strong>{sentiment.title()}:</strong> {count}</li>" for sentiment, count in sentiment_stats.items()])}
        </ul>
        
        <h3>🏷️ Topic Distribution</h3>
        <ul>
        {''.join([f"<li><strong>{topic.title()}:</strong> {count}</li>" for topic, count in topic_stats.items()])}
        </ul>
        
        <p><em>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
        """
        
    except Exception as e:
        return f"<h2>❌ Analytics Error</h2><p>{str(e)}</p>"

@app.route("/test", methods=['GET'])
def test():
    """Enhanced test endpoint"""
    try:
        if not client:
            return "<h2>❌ Groq API Test Failed</h2><p><strong>Error:</strong> Groq client not initialized</p>"
        
        test_response = generate_enhanced_response("आप कैसे काम करते हैं?", "test_user")
        return f"""
        <h2>✅ Enhanced Groq API Test Successful!</h2>
        <p><strong>Test Query:</strong> आप कैसे काम करते हैं?</p>
        <p><strong>Response:</strong> {test_response}</p>
        <p><strong>Response Length:</strong> {len(test_response)} characters</p>
        """
    except Exception as e:
        return f"<h2>❌ Groq API Test Failed</h2><p><strong>Error:</strong> {str(e)}</p>"

@app.route("/health", methods=['GET'])
def health():
    """Enhanced health check"""
    try:
        active_calls = len([state for state in call_states.values() 
                           if datetime.now() - state.last_activity < timedelta(minutes=5)])
        
        status = {
            "status": "healthy" if client else "unhealthy",
            "groq_client": "connected" if client else "disconnected",
            "total_conversations": len(conversation_contexts),
            "active_calls": active_calls,
            "features": {
                "interruption_handling": True,
                "sentiment_analysis": True,
                "context_awareness": True,
                "topic_detection": True,
                "enhanced_responses": True
            },
            "timestamp": datetime.now().isoformat()
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("🚀 Starting Advanced Indian Voice Bot...")
    print(f"✅ Groq API Key: {'Loaded' if GROQ_API_KEY else 'Missing'}")
    print(f"✅ Twilio Credentials: {'Loaded' if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else 'Missing'}")
    print(f"🌐 Server starting on port {port}")
    print(f"🔧 Debug mode: {debug_mode}")
    print("🎯 Enhanced features: Interruption handling, Sentiment analysis, Context awareness")
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
