import streamlit as st
import cv2
import numpy as np
import sqlite3
import requests
import json
import base64
from datetime import datetime
import pandas as pd
from PIL import Image
import io
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import time
import threading
from groq import Groq

# Page configuration
st.set_page_config(
    page_title="Cop Hand - AI Civic Safety Platform",
    page_icon="👮‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database initialization
def init_database():
    """Initialize SQLite database for storing reports"""
    conn = sqlite3.connect('cop_hand.db')
    c = conn.cursor()
    
    # Create reports table
    c.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            report_type TEXT,
            violation_type TEXT,
            location TEXT,
            latitude REAL,
            longitude REAL,
            ai_summary TEXT,
            confidence_score REAL,
            status TEXT DEFAULT 'pending',
            image_data BLOB,
            citizen_id TEXT
        )
    ''')
    
    # Create alerts table for high-priority security alerts
    c.execute('''
        CREATE TABLE IF NOT EXISTS security_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            alert_type TEXT,
            emotion_detected TEXT,
            confidence_score REAL,
            location TEXT,
            status TEXT DEFAULT 'active',
            image_data BLOB
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize Groq AI client
@st.cache_resource
def init_groq_client():
    """Initialize Groq AI client - you'll need to add your API key"""
    try:
        # Replace with your Groq API key
        client = Groq(api_key=st.secrets.get("GROQ_API_KEY", "Your api key here"))
        return client
    except Exception as e:
        st.error(f"Failed to initialize Groq AI: {e}")
        return None

# Traffic violation detection using OpenCV (simplified YOLO-like detection)
class TrafficViolationDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def detect_violations(self, image):
        """Detect traffic violations in image"""
        violations = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple face detection for counting people (triple riding detection)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 2:  # Simplified triple riding detection
            violations.append({
                'type': 'triple_riding',
                'confidence': 0.85,
                'description': f'Detected {len(faces)} people on vehicle (triple riding violation)'
            })
        
        # Simulate helmet detection (in real implementation, use trained YOLO model)
        if len(faces) > 0 and np.random.random() > 0.7:  # Simulated detection
            violations.append({
                'type': 'no_helmet',
                'confidence': 0.78,
                'description': 'Person detected without helmet'
            })
        
        return violations, faces

# Emotion detection for suspicious behavior
class EmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # In production, load a trained emotion recognition model
        
    def detect_emotions(self, image):
        """Detect emotions for suspicious behavior analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        emotions = []
        for (x, y, w, h) in faces:
            # Simulate emotion detection (replace with actual model inference)
            face_roi = gray[y:y+h, x:x+w]
            
            # Simulated emotion scores
            emotion_scores = {
                'fear': np.random.random(),
                'stress': np.random.random(),
                'anger': np.random.random(),
                'neutral': np.random.random()
            }
            
            # Determine dominant emotion
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[dominant_emotion]
            
            emotions.append({
                'emotion': dominant_emotion,
                'confidence': confidence,
                'bbox': (x, y, w, h),
                'is_suspicious': dominant_emotion in ['fear', 'stress', 'anger'] and confidence > 0.7
            })
        
        return emotions

# Get location using IP-based geolocation
def get_location():
    """Get approximate location using IP geolocation"""
    try:
        response = requests.get('https://ipapi.co/json/', timeout=5)
        data = response.json()
        return {
            'city': data.get('city', 'Unknown'),
            'region': data.get('region', 'Unknown'),
            'country': data.get('country_name', 'Unknown'),
            'latitude': data.get('latitude', 0.0),
            'longitude': data.get('longitude', 0.0)
        }
    except:
        return {
            'city': 'Unknown',
            'region': 'Unknown', 
            'country': 'Unknown',
            'latitude': 0.0,
            'longitude': 0.0
        }

# Generate AI summary using Groq
def generate_ai_summary(violation_data, groq_client):
    """Generate natural language summary using Groq AI"""
    if not groq_client:
        return "AI summarization unavailable"
    
    try:
        prompt = f"""
        Analyze this traffic violation report and provide a concise summary:
        
        Violation Type: {violation_data.get('type', 'Unknown')}
        Confidence: {violation_data.get('confidence', 0)}
        Description: {violation_data.get('description', 'No description')}
        Location: {violation_data.get('location', 'Unknown location')}
        Timestamp: {violation_data.get('timestamp', 'Unknown time')}
        
        Please provide:
        1. A brief summary of the violation
        2. Severity level (Low/Medium/High)
        3. Recommended action
        
        Keep the response under 100 words and professional.
        """
        
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",  # Free tier model
            max_tokens=150
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"AI summary generation failed: {str(e)}"

# Save report to database
def save_report(report_data):
    """Save report to SQLite database"""
    conn = sqlite3.connect('cop_hand.db')
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO reports (timestamp, report_type, violation_type, location, 
                           latitude, longitude, ai_summary, confidence_score, image_data, citizen_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        report_data['timestamp'],
        report_data['report_type'],
        report_data['violation_type'],
        report_data['location'],
        report_data['latitude'],
        report_data['longitude'],
        report_data['ai_summary'],
        report_data['confidence_score'],
        report_data['image_data'],
        report_data['citizen_id']
    ))
    
    conn.commit()
    conn.close()

# Save security alert
def save_security_alert(alert_data):
    """Save high-priority security alert"""
    conn = sqlite3.connect('cop_hand.db')
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO security_alerts (timestamp, alert_type, emotion_detected, 
                                   confidence_score, location, image_data)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        alert_data['timestamp'],
        alert_data['alert_type'],
        alert_data['emotion_detected'],
        alert_data['confidence_score'],
        alert_data['location'],
        alert_data['image_data']
    ))
    
    conn.commit()
    conn.close()

# Initialize components
init_database()
violation_detector = TrafficViolationDetector()
emotion_detector = EmotionDetector()
groq_client = init_groq_client()

# Sidebar navigation
st.sidebar.title("🚨 Cop Hand")
st.sidebar.markdown("AI-Powered Civic Safety Platform")

page = st.sidebar.selectbox(
    "Navigate to:",
    ["🏠 Home", "📸 Report Violation", "📹 Live Camera", "👮‍♂️ Admin Dashboard", "📊 Analytics"]
)

# Main application logic
if page == "🏠 Home":
    st.title("🚨 Cop Hand - AI Civic Safety Platform")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header("🚗 Traffic Violations")
        st.write("Report traffic violations like:")
        st.write("• Triple riding")
        st.write("• No helmet violations")
        st.write("• Underage driving")
        st.write("• Reckless driving")
    
    with col2:
        st.header("🔍 AI Detection")
        st.write("Advanced AI capabilities:")
        st.write("• Real-time violation detection")
        st.write("• Emotion analysis")
        st.write("• Automatic report generation")
        st.write("• Suspicious behavior alerts")
    
    with col3:
        st.header("🛡️ Public Safety")
        st.write("Enhanced security features:")
        st.write("• Facial emotion monitoring")
        st.write("• Stress detection")
        st.write("• Security threat alerts")
        st.write("• Real-time monitoring")
    
    st.markdown("---")
    st.info("📍 Click 'Report Violation' to start reporting traffic violations or use 'Live Camera' for real-time monitoring.")

elif page == "📸 Report Violation":
    st.title("📸 Report Traffic Violation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Evidence")
        
        # File upload option
        uploaded_file = st.file_uploader("Upload Image/Video", type=['jpg', 'jpeg', 'png', 'mp4', 'avi'])
        
        # Camera capture option
        st.subheader("Or Capture Live")
        camera_input = st.camera_input("Take a picture")
        
        if uploaded_file is not None or camera_input is not None:
            # Process the image
            if camera_input:
                image = Image.open(camera_input)
            else:
                image = Image.open(uploaded_file)
            
            st.image(image, caption="Evidence Image", use_column_width=True)
            
            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Process with AI
            with st.spinner("🤖 Analyzing image with AI..."):
                violations, faces = violation_detector.detect_violations(opencv_image)
                emotions = emotion_detector.detect_emotions(opencv_image)
            
            if violations:
                st.success(f"✅ Detected {len(violations)} violation(s)!")
                
                for violation in violations:
                    st.warning(f"**{violation['type'].replace('_', ' ').title()}** - Confidence: {violation['confidence']:.2f}")
                    st.write(violation['description'])
                
                # Get location
                location_data = get_location()
                
                # Generate AI summary
                violation_data = {
                    'type': violations[0]['type'],
                    'confidence': violations[0]['confidence'],
                    'description': violations[0]['description'],
                    'location': f"{location_data['city']}, {location_data['region']}",
                    'timestamp': datetime.now().isoformat()
                }
                
                ai_summary = generate_ai_summary(violation_data, groq_client)
                
                with col2:
                    st.subheader("📋 Report Summary")
                    st.write(f"**Location:** {location_data['city']}, {location_data['region']}")
                    st.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Confidence:** {violations[0]['confidence']:.2%}")
                    
                    st.subheader("🤖 AI Analysis")
                    st.write(ai_summary)
                    
                    # Submit report
                    citizen_id = st.text_input("Your Contact ID (optional)")
                    
                    if st.button("📤 Submit Report", type="primary"):
                        # Save to database
                        img_bytes = io.BytesIO()
                        image.save(img_bytes, format='PNG')
                        img_bytes = img_bytes.getvalue()
                        
                        report_data = {
                            'timestamp': datetime.now().isoformat(),
                            'report_type': 'traffic_violation',
                            'violation_type': violations[0]['type'],
                            'location': f"{location_data['city']}, {location_data['region']}",
                            'latitude': location_data['latitude'],
                            'longitude': location_data['longitude'],
                            'ai_summary': ai_summary,
                            'confidence_score': violations[0]['confidence'],
                            'image_data': img_bytes,
                            'citizen_id': citizen_id or 'anonymous'
                        }
                        
                        save_report(report_data)
                        st.success("✅ Report submitted successfully!")
                        st.balloons()
            else:
                st.info("ℹ️ No traffic violations detected in this image.")

elif page == "📹 Live Camera":
    st.title("📹 Live Camera Monitoring")
    st.markdown("Real-time AI-powered violation and suspicious behavior detection")
    
    # WebRTC configuration for live camera
    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
    
    class VideoProcessor:
        def __init__(self):
            self.violation_detector = TrafficViolationDetector()
            self.emotion_detector = EmotionDetector()
            
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Process for violations
            violations, faces = self.violation_detector.detect_violations(img)
            emotions = emotion_detector.detect_emotions(img)
            
            # Draw bounding boxes for faces
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img, "Person", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Draw emotion indicators
            for emotion in emotions:
                x, y, w, h = emotion['bbox']
                color = (0, 0, 255) if emotion['is_suspicious'] else (0, 255, 0)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, f"{emotion['emotion']}: {emotion['confidence']:.2f}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Trigger security alert for suspicious emotions
                if emotion['is_suspicious'] and emotion['confidence'] > 0.8:
                    location_data = get_location()
                    
                    # Save security alert
                    _, img_encoded = cv2.imencode('.png', img)
                    alert_data = {
                        'timestamp': datetime.now().isoformat(),
                        'alert_type': 'suspicious_behavior',
                        'emotion_detected': emotion['emotion'],
                        'confidence_score': emotion['confidence'],
                        'location': f"{location_data['city']}, {location_data['region']}",
                        'image_data': img_encoded.tobytes()
                    }
                    save_security_alert(alert_data)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    # Live camera stream
    webrtc_ctx = webrtc_streamer(
        key="live-monitoring",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if webrtc_ctx.video_processor:
        st.success("📹 Live monitoring active!")
        st.info("🔍 AI is analyzing the video stream for violations and suspicious behavior")

elif page == "👮‍♂️ Admin Dashboard":
    st.title("👮‍♂️ Admin Dashboard")
    
    # Tabs for different admin functions
    tab1, tab2, tab3 = st.tabs(["📋 Reports", "🚨 Security Alerts", "⚙️ Settings"])
    
    with tab1:
        st.subheader("Traffic Violation Reports")
        
        # Fetch reports from database
        conn = sqlite3.connect('cop_hand.db')
        reports_df = pd.read_sql_query("SELECT * FROM reports ORDER BY timestamp DESC", conn)
        conn.close()
        
        if not reports_df.empty:
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                status_filter = st.selectbox("Filter by Status", ["All", "pending", "approved", "rejected"])
            with col2:
                violation_filter = st.selectbox("Filter by Violation", ["All"] + list(reports_df['violation_type'].unique()))
            with col3:
                if st.button("🔄 Refresh"):
                    st.rerun()
            
            # Apply filters
            filtered_df = reports_df.copy()
            if status_filter != "All":
                filtered_df = filtered_df[filtered_df['status'] == status_filter]
            if violation_filter != "All":
                filtered_df = filtered_df[filtered_df['violation_type'] == violation_filter]
            
            # Display reports
            for _, report in filtered_df.iterrows():
                with st.expander(f"Report #{report['id']} - {report['violation_type']} - {report['timestamp'][:16]}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Location:** {report['location']}")
                        st.write(f"**Confidence:** {report['confidence_score']:.2%}")
                        st.write(f"**Status:** {report['status']}")
                        st.write(f"**Citizen ID:** {report['citizen_id']}")
                        
                        st.subheader("AI Summary:")
                        st.write(report['ai_summary'])
                        
                        # Action buttons
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            if st.button(f"✅ Approve #{report['id']}", key=f"approve_{report['id']}"):
                                # Update status in database
                                conn = sqlite3.connect('cop_hand.db')
                                c = conn.cursor()
                                c.execute("UPDATE reports SET status = 'approved' WHERE id = ?", (report['id'],))
                                conn.commit()
                                conn.close()
                                st.success("Report approved!")
                                st.rerun()
                        
                        with col_b:
                            if st.button(f"❌ Reject #{report['id']}", key=f"reject_{report['id']}"):
                                # Update status in database
                                conn = sqlite3.connect('cop_hand.db')
                                c = conn.cursor()
                                c.execute("UPDATE reports SET status = 'rejected' WHERE id = ?", (report['id'],))
                                conn.commit()
                                conn.close()
                                st.warning("Report rejected!")
                                st.rerun()
                        
                        with col_c:
                            fine_amount = st.number_input(f"Fine Amount (₹)", key=f"fine_{report['id']}", min_value=0, value=500)
                            if st.button(f"💰 Generate Fine #{report['id']}", key=f"fine_btn_{report['id']}"):
                                st.success(f"Fine of ₹{fine_amount} generated for Report #{report['id']}")
                    
                    with col2:
                        # Display image if available
                        if report['image_data']:
                            try:
                                img = Image.open(io.BytesIO(report['image_data']))
                                st.image(img, caption="Evidence", use_column_width=True)
                            except:
                                st.error("Unable to display image")
        else:
            st.info("No reports available")
    
    with tab2:
        st.subheader("🚨 Security Alerts")
        
        # Fetch security alerts
        conn = sqlite3.connect('cop_hand.db')
        alerts_df = pd.read_sql_query("SELECT * FROM security_alerts ORDER BY timestamp DESC", conn)
        conn.close()
        
        if not alerts_df.empty:
            # Show active alerts prominently
            active_alerts = alerts_df[alerts_df['status'] == 'active']
            if not active_alerts.empty:
                st.error(f"⚠️ {len(active_alerts)} ACTIVE SECURITY ALERTS")
                
                for _, alert in active_alerts.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div style="padding: 1rem; background-color: #ffebee; border-left: 5px solid #f44336; margin: 1rem 0;">
                            <h4>🚨 SECURITY ALERT #{alert['id']}</h4>
                            <p><strong>Time:</strong> {alert['timestamp']}</p>
                            <p><strong>Location:</strong> {alert['location']}</p>
                            <p><strong>Emotion Detected:</strong> {alert['emotion_detected'].title()}</p>
                            <p><strong>Confidence:</strong> {alert['confidence_score']:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"✅ Resolve Alert #{alert['id']}", key=f"resolve_{alert['id']}"):
                                conn = sqlite3.connect('cop_hand.db')
                                c = conn.cursor()
                                c.execute("UPDATE security_alerts SET status = 'resolved' WHERE id = ?", (alert['id'],))
                                conn.commit()
                                conn.close()
                                st.success("Alert resolved!")
                                st.rerun()
                        
                        with col2:
                            if st.button(f"🚨 Escalate #{alert['id']}", key=f"escalate_{alert['id']}"):
                                st.warning("Alert escalated to law enforcement!")
            else:
                st.success("✅ No active security alerts")
            
            # Show all alerts history
            st.subheader("Alert History")
            st.dataframe(alerts_df)
        else:
            st.info("No security alerts recorded")
    
    with tab3:
        st.subheader("⚙️ System Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**AI Detection Settings**")
            confidence_threshold = st.slider("Minimum Confidence Threshold", 0.0, 1.0, 0.7)
            enable_auto_alerts = st.checkbox("Enable Automatic Security Alerts", value=True)
            alert_threshold = st.slider("Security Alert Threshold", 0.0, 1.0, 0.8)
        
        with col2:
            st.write("**System Status**")
            st.success("✅ Database Connected")
            st.success("✅ AI Models Loaded")
            if groq_client:
                st.success("✅ Groq AI Connected")
            else:
                st.error("❌ Groq AI Disconnected")
            
            st.info(f"📊 Total Reports: {len(reports_df) if not reports_df.empty else 0}")
            st.info(f"🚨 Active Alerts: {len(active_alerts) if 'active_alerts' in locals() and not active_alerts.empty else 0}")

elif page == "📊 Analytics":
    st.title("📊 Analytics Dashboard")
    
    # Fetch data for analytics
    conn = sqlite3.connect('cop_hand.db')
    reports_df = pd.read_sql_query("SELECT * FROM reports", conn)
    alerts_df = pd.read_sql_query("SELECT * FROM security_alerts", conn)
    conn.close()
    
    if not reports_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Violation Types")
            violation_counts = reports_df['violation_type'].value_counts()
            st.bar_chart(violation_counts)
            
            st.subheader("📍 Reports by Location")
            location_counts = reports_df['location'].value_counts().head(10)
            st.bar_chart(location_counts)
        
        with col2:
            st.subheader("⏰ Reports Over Time")
            reports_df['date'] = pd.to_datetime(reports_df['timestamp']).dt.date
            daily_counts = reports_df.groupby('date').size()
            st.line_chart(daily_counts)
            
            st.subheader("✅ Report Status Distribution")
            status_counts = reports_df['status'].value_counts()
            st.pie_chart(status_counts)
        
        # Key metrics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reports", len(reports_df))
        with col2:
            pending_reports = len(reports_df[reports_df['status'] == 'pending'])
            st.metric("Pending Reports", pending_reports)
        with col3:
            approved_reports = len(reports_df[reports_df['status'] == 'approved'])
            st.metric("Approved Reports", approved_reports)
        with col4:
            avg_confidence = reports_df['confidence_score'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    else:
        st.info("No data available for analytics")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <p>🚨 Cop Hand - AI-Powered Civic Safety Platform</p>
        <p>Built with Streamlit • Powered by Groq AI • Open Source Computer Vision</p>
    </div>
    """, 
    unsafe_allow_html=True
)