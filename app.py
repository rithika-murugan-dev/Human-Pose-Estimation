import streamlit as st
import cv2
import mediapipe.python.solutions as mp_solutions
import numpy as np
import pandas as pd
import tempfile
from deep_translator import GoogleTranslator
import streamlit.components.v1 as components
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import base64
import io


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Injury Risk Analysis System",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown("""
<style>
/* 1. Sidebar Background - matching the main page theme perfectly */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e0033 0%, #4c1d95 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 10px 0 30px rgba(0, 0, 0, 0.5);
}
/* CSS to make video player always take full width */
[data-testid="stVideo"] {
    width: 100% !important;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}

/* 2. Text & Labels Visibility - Essential for Dark Background */
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p {
    color: #f5f3ff !important; /* Soft White/Violet Text */
    text-shadow: 0px 2px 4px rgba(0,0,0,0.3);
}

/* 3. Input Fields - Brightening up for Contrast */
div[data-testid="stSidebar"] .stTextInput input, 
div[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"],
div[data-testid="stSidebar"] .stNumberInput input {
    background-color: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: white !important;
    border-radius: 12px !important;
}

/* 4. Arrow Fix - Making it stand out against Dark Sidebar */
[data-testid="stSidebarCollapseButton"] {
    background-color: rgba(255, 255, 255, 0.1) !important;
    color: #ddd6fe !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}

/* 5. Expander Makeover */
div[data-testid="stSidebar"] .st-emotion-cache-p4m0d0 {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 15px !important;
}
            .profile-circle {

    width: 60px;

    height: 60px;

    background: linear-gradient(90deg, #ff416c, #ff4b2b);

    border-radius: 50%;

    display: flex;

    align-items: center;

    justify-content: center;

    color: white;

    font-size: 24px;

    cursor: pointer;

    border: 2px solid white;

    margin: 10px auto;

}
</style>
""", unsafe_allow_html=True)
# -------------------------------------------------
# PREMIUM GLASS-FROST WHITE-VIOLET UI
# -------------------------------------------------
st.markdown("""
<style>
/* Remove top blank padding */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem;
}

/* Bright Radiant Background - Violet to White/Light Cyan */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top right, 
    #ffffff 0%, 
    #f5f3ff 20%, 
    #ddd6fe 50%, 
    #7c3aed 100%);
    background-attachment: fixed;
}
            

/* Glassmorphism White Cards - Center Aligned Content */
div[data-testid="stVerticalBlock"] > div > div > div > div.stMarkdown {
    background: rgba(255, 255, 255, 0.7); /* Bright White Glass */
    backdrop-filter: blur(20px);
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.5);
    padding: 30px;
    box-shadow: 0 10px 40px rgba(124, 58, 237, 0.1);
    text-align: center; /* centering title content */
}

/* Centering and Styling Hero Titles */
.hero {
    text-align: center;
    padding-bottom: 30px;
}

.hero h1 {
    font-size: 72px;
    font-weight: 900;
    color: #4c1d95; /* Deep Violet */
    margin-bottom: 10px;
    letter-spacing: -1.5px;
}

.hero p {
    font-size: 26px;
    color: #6d28d9;
    font-weight: 500;
    letter-spacing: 1px;
}

/* Neon-Violet Buttons */
.stButton>button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white !important;
    border-radius: 50px;
    padding: 18px 45px;
    font-weight: 700;
    border: none;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    box-shadow: 0 10px 20px rgba(124, 58, 237, 0.3);
}

.stButton>button:hover {
    box-shadow: 0 15px 30px rgba(124, 58, 237, 0.5);
    transform: translateY(-4px) scale(1.02);
    background: linear-gradient(135deg, #8b5cf6, #6366f1);
}

/* Metric Display Adjustments - Bright Cyan on White */
[data-testid="stMetricValue"] {
    color: #7c3aed !important;
    font-weight: 900;
}

[data-testid="stMetricLabel"] {
    color: #4c1d95 !important;
    font-weight: 600;
}
            /* st.info (Highest observed stress) line-ah white-ah matha */
div[data-testid="stNotification"] p {
    color: white !important;
    font-weight: 700 !important;
    font-size: 18px !important;
}

/* Alert box background-ah konjam transparent-ah matha matching theme-ku */
div[data-testid="stNotificationContent"] {
    background-color: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 12px;
}

/* Sidebar Styling - Bright Mode */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.9);
    border-right: 1px solid rgba(124, 58, 237, 0.1);
}

/* Language selector container fix */
.stSelectbox label {
    color: #4c1d95 !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LANGUAGE SELECTOR
# -------------------------------------------------

st.markdown("<div style='position:fixed; bottom:20px; left:20px;'>", unsafe_allow_html=True)

language = st.selectbox(
    "🌐 Language",
    ["English","Tamil","Hindi"]
)

st.markdown("</div>", unsafe_allow_html=True)
# -------------------------------------------------
# LANGUAGE TEXT DICTIONARY
# -------------------------------------------------

translations = {

"English":{
"title":"Injury Risk Analysis System",
"subtitle":"AI Training & Motion Symmetry Intelligence",
"upload":"Upload Training Video",
"drag":"Drag and drop video here",
"analyze":"Analyze Full Body Risk",
"complete":"Analysis Completed Successfully!",
"risktrend":"Risk Trend Over Time",
"final":"Final Risk Assessment",
"joint":"Joint Specific Risk",
"rec_start": "🔴 Start Recording",
"rec_stop": "⏹ Stop & Download",
"rec_instr": "Step 1: Click 'Start'. Step 2: Record. Step 3: Click 'Stop & Download'. Step 4: Upload the file on the left!",
"take_live": "Take Live Record",
"prof_title": "Athlete Profile",
    "prof_edit": "Edit Profile Details",
    "prof_name": "Full Name",
    "prof_save": "Save Profile",
    "ai_coach": " AI Coach: Personalized Recommendations",
    "ai_perfect": " Form looks great! No adjustments needed.",
    "pdf_btn": "📄 Download Training Report (PDF)",
    "err_human": " ERROR: No Human detected in the video!",
    "prof_age": "Age",
    "prof_sport": "Sport",
    "prof_type": "Training Type",
    "sports_list": ["Football", "Basketball", "Weightlifting", "Cricket", "Athletics"],
    "train_list": ["Strength", "Endurance", "Recovery"]
},

"Tamil":{
"title":"காயம் அபாய பகுப்பாய்வு அமைப்பு",
"subtitle":"AI பயிற்சி மற்றும் இயக்க சமநிலை பகுப்பாய்வு",
"upload":"பயிற்சி வீடியோ பதிவேற்று",
"drag":"வீடியோவை இங்கே பதிவேற்றவும்",
"analyze":"முழு உடல் அபாயம் பகுப்பாய்வு",
"complete":"பகுப்பாய்வு வெற்றிகரமாக முடிந்தது!",
"risktrend":"அபாய போக்கு வரைபடம்",
"final":"இறுதி அபாய மதிப்பீடு",
"joint":"மூட்டு அபாயம்",
"rec_start": "🔴 பதிவை தொடங்கு",
"rec_stop": "⏹ நிறுத்தி தரவிறக்கவும்",
"rec_instr": "படி 1: 'Start' கிளிக் செய்யவும். படி 2: பதிவு செய்யவும். படி 3: 'Stop' கிளிக் செய்து தரவிறக்கவும். படி 4: கோப்பை பதிவேற்றவும்!",
"take_live": " நேரடி பதிவு",
"prof_title": "வீரர் விவரக்குறிப்பு",
    "prof_edit": "விவரங்களை மாற்றியமைக்க",
    "prof_name": "முழு பெயர்",
    "prof_save": "விவரங்களைச் சேமி",
    "ai_coach": " AI பயிற்சியாளர்: தனிப்பயனாக்கப்பட்ட பரிந்துரைகள்",
    "ai_perfect": " உங்கள் பயிற்சி சரியாக உள்ளது! மாற்றங்கள் தேவையில்லை.",
    "pdf_btn": " 📄 பயிற்சி அறிக்கையை பதிவிறக்கவும் (PDF)",
    "err_human": " பிழை: வீடியோவில் மனித உடல் கண்டறியப்படவில்லை!",
    "prof_age": "வயது",
    "prof_sport": "விளையாட்டு",
    "prof_type": "பயிற்சி வகை",
    "sports_list": ["கால்பந்து", "கூடைப்பந்து", "பளுதூக்குதல்", "கிரிக்கெட்", "தடகளம்"],
    "train_list": ["வலிமை", "தாங்குதிறன்", "மீட்பு"]
},

"Hindi":{
"title":"चोट जोखिम विश्लेषण प्रणाली",
"subtitle":"AI प्रशिक्षण और गति विश्लेषण",
"upload":"प्रशिक्षण वीडियो अपलोड करें",
"drag":"वीडियो यहाँ अपलोड करें",
"analyze":"पूर्ण शरीर जोखिम विश्लेषण",
"complete":"विश्लेषण सफलतापूर्वक पूरा हुआ!",
"risktrend":"जोखिम ग्राफ",
"final":"अंतिम जोखिम मूल्यांकन",
"joint":"जोड़ जोखिम",
"rec_start": "🔴 रिकॉर्डिंग शुरू करें",
"rec_stop": "⏹ रोकें और डाउनलोड करें",
"rec_instr": "चरण 1: 'Start' पर क्लिक करें। चरण 2: रिकॉर्ड करें। चरण 3: 'Stop' करके डाउनलोड करें। चरण 4: फाइल अपलोड करें!",
"take_live": " लाइव रिकॉर्ड लें",
"prof_title": "एथलीट प्रोफाइल",
    "prof_edit": "प्रोफ़ाइल विवरण संपादित करें",
    "prof_name": "पूरा नाम",
    "prof_save": "प्रोफ़ाइल सहेजें",
    "ai_coach": "AI कोच: व्यक्तिगत सिफारिशें",
    "ai_perfect": " फॉर्म बहुत अच्छा लग रहा है! किसी बदलाव की जरूरत नहीं है।",
    "pdf_btn": "📄 प्रशिक्षण रिपोर्ट डाउनलोड करें (PDF)",
    "err_human": " त्रुटि: वीडियो में कोई मानव नहीं पाया गया!",
    "prof_age": "आयु",
    "prof_sport": "खेल",
    "prof_type": "प्रशिक्षण प्रकार",
    "sports_list": ["फुटबॉल", "बास्केटबॉल", "भारोत्तोलन", "क्रिकेट", "एथलेटिक्स"],
    "train_list": ["शक्ति", "सहनशक्ति", "रिकवरी"]
}

}
T = translations[language]
# --- Helper Function for Circular Image ---
def get_image_base64(image_file):
    return base64.b64encode(image_file.getvalue()).decode()

# -------------------------------------------------
# ATHLETE PROFILE SYSTEM (SIDEBAR)
# -------------------------------------------------
with st.sidebar:
    # Handle Profile Picture logic
    if "profile_pic" not in st.session_state:
        st.session_state.profile_pic = None

    # Circular Profile Display
    if st.session_state.profile_pic:
        st.markdown(f"""
            <div style="display: flex; justify-content: center;">
                <img src="data:image/png;base64,{st.session_state.profile_pic}" 
                     style="width: 80px; height: 80px; border-radius: 50%; object-fit: cover; border: 3px solid #ff416c;">
            </div>
        """, unsafe_allow_html=True)
    else:
       st.markdown('<div class="profile-circle">👤</div>', unsafe_allow_html=True)
       st.markdown(f"<h3 style='text-align: center;'>{T['prof_title']}</h3>", unsafe_allow_html=True)


    
    if "athlete_data" not in st.session_state:
        st.session_state.athlete_data = {
            "name": "Guest Athlete",
            "age": 22,
            "sport": "None",
            "training": "General"
        }

    with st.expander(f"📝 {T['prof_edit']}"):
        # Photo Upload Input
        uploaded_img = st.file_uploader("Upload Profile Photo", type=["jpg", "png", "jpeg"])
        
        name = st.text_input(T["prof_name"], st.session_state.athlete_data["name"])
        age = st.number_input(T["prof_age"], value=st.session_state.athlete_data["age"])
        sport = st.selectbox(T["prof_sport"], T["sports_list"])
        training = st.selectbox(T["prof_type"], T["train_list"])
        
        if st.button(T["prof_save"]):
            # If new photo is uploaded, convert and save it
            if uploaded_img:
                st.session_state.profile_pic = get_image_base64(uploaded_img)
            
            st.session_state.athlete_data = {
                "name": name,
                "age": age,
                "sport": sport,
                "training": training
            }
            st.success("Updated!")
            st.rerun() # Refresh to show photo immediately

    # Display Current Profile Info
    st.markdown("---")
    st.markdown(f"**{T['prof_name']}:** {st.session_state.athlete_data['name']}")
    st.markdown(f"**{T['prof_age']}:** {st.session_state.athlete_data['age']}")
    st.markdown(f"**{T['prof_sport']}:** {st.session_state.athlete_data['sport']}")
    st.markdown(f"**{T['prof_type']}:** {st.session_state.athlete_data['training']}")
    
# -------------------------------------------------
# HERO
# -------------------------------------------------
st.markdown(f"""
<div class="hero">
<h1>{T['title']}</h1>
<p>{T['subtitle']}</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "risk_history" not in st.session_state:
    st.session_state.risk_history = []

if "joint_history" not in st.session_state:
    st.session_state.joint_history = []

if "output_path" not in st.session_state:
    st.session_state.output_path = None

def translate_text(text):

    if language == "English":
        return text

    if language == "Tamil":
        return GoogleTranslator(source='auto', target='ta').translate(text)

    if language == "Hindi":
        return GoogleTranslator(source='auto', target='hi').translate(text)
    
def generate_pdf_report(risk_history, joint_history, avg_risk):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    
    # Header
    p.setFont("Helvetica-Bold", 22)
    p.drawCentredString(300, 750, "Injury Risk Analysis Report")
    
    p.setFont("Helvetica", 12)
    p.drawString(50, 720, f"Overall Risk Assessment: {avg_risk}%")
    status = "LOW" if avg_risk < 25 else "MODERATE" if avg_risk < 55 else "HIGH"
    p.drawString(50, 705, f"Risk Status: {status}")

    # Create Graph Image for PDF
    plt.figure(figsize=(6, 3))
    plt.plot(risk_history, color='purple')
    plt.title("Risk Trend Over Time")
    plt.xlabel("Frames")
    plt.ylabel("Risk %")
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    
    # Save temp image for reportlab to read
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
        tmp_img.write(img_buffer.getvalue())
        p.drawImage(tmp_img.name, 50, 480, width=500, height=200)

    # Joint Data
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, 450, "Joint-Specific Average Stress:")
    p.setFont("Helvetica", 12)
    
    y = 430
    if joint_history:
        joint_df = pd.DataFrame(joint_history)
        joint_avg = joint_df.mean().astype(int)
        for joint, val in joint_avg.items():
            p.drawString(70, y, f"• {joint}: {val}%")
            y -= 20
            if y < 50: # Page break logic if needed
                p.showPage()
                y = 750

    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

# -------------------------------------------------
# ANGLE FUNCTION
# -------------------------------------------------
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

# -------------------------------------------------
# BODY ANGLES
# -------------------------------------------------
def get_body_angles(lm, mp_pose):
    
    return {
        "Left Knee": calculate_angle(lm[mp_pose.PoseLandmark.LEFT_HIP],
                                     lm[mp_pose.PoseLandmark.LEFT_KNEE],
                                     lm[mp_pose.PoseLandmark.LEFT_ANKLE]),

        "Right Knee": calculate_angle(lm[mp_pose.PoseLandmark.RIGHT_HIP],
                                      lm[mp_pose.PoseLandmark.RIGHT_KNEE],
                                      lm[mp_pose.PoseLandmark.RIGHT_ANKLE]),

        "Left Hip": calculate_angle(lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                    lm[mp_pose.PoseLandmark.LEFT_HIP],
                                    lm[mp_pose.PoseLandmark.LEFT_KNEE]),

        "Right Hip": calculate_angle(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                     lm[mp_pose.PoseLandmark.RIGHT_HIP],
                                     lm[mp_pose.PoseLandmark.RIGHT_KNEE]),

        "Left Shoulder": calculate_angle(lm[mp_pose.PoseLandmark.LEFT_ELBOW],
                                         lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                         lm[mp_pose.PoseLandmark.LEFT_HIP]),

        "Right Shoulder": calculate_angle(lm[mp_pose.PoseLandmark.RIGHT_ELBOW],
                                          lm[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                          lm[mp_pose.PoseLandmark.RIGHT_HIP]),

        "Left Elbow": calculate_angle(lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                      lm[mp_pose.PoseLandmark.LEFT_ELBOW],
                                      lm[mp_pose.PoseLandmark.LEFT_WRIST]),

        "Right Elbow": calculate_angle(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                       lm[mp_pose.PoseLandmark.RIGHT_ELBOW],
                                       lm[mp_pose.PoseLandmark.RIGHT_WRIST]),

        "Trunk": calculate_angle(lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                 lm[mp_pose.PoseLandmark.LEFT_HIP],
                                 lm[mp_pose.PoseLandmark.LEFT_KNEE]),

        "Neck": calculate_angle(lm[mp_pose.PoseLandmark.NOSE],
                                lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                lm[mp_pose.PoseLandmark.LEFT_HIP])
    }
def get_ai_recommendations(joint_avg, lang):
    recommendations = {}
    
    # Static Scientific Database with Manual Translations
    kb_content = {
        "Left Knee": {
            "critical": {
                "English": ["⚠️ High Meniscus Pressure: Reduce squat depth.", "Alignment Fix: Don't let knee cave inward."],
                "Tamil": ["⚠️ மெனிசகஸ் அழுத்தம்: குந்துதல் (squat) ஆழத்தை குறைக்கவும்.", "சீரமைப்பு: முழங்காலை உள்நோக்கி சாய்க்க வேண்டாம்."],
                "Hindi": ["⚠️ मेनिस्कस दबाव: स्क्वाट की गहराई कम करें।", "संरेखण: घुटने को अंदर की ओर न झुकने दें।"]
            },
            "general": {
                "English": ["💡 Maintenance: Focus on slow eccentric phase.", "Improve ankle mobility for better knee safety."],
                "Tamil": ["💡 பராமரிப்பு: மெதுவான இறக்க நிலையில் கவனம் செலுத்துங்கள்.", "கணுக்கால் அசைவுத்திறனை மேம்படுத்தவும்."],
                "Hindi": ["💡 रखरखाव: धीमे सनकी चरण पर ध्यान दें।", "घुटने की सुरक्षा के लिए टखने की गतिशीलता में सुधार करें।"]
            }
        },
        "Right Knee": {
            "critical": {
                "English": ["⚠️ Patellar Strain: Avoid locking knees at the top.", "Form Check: Keep knee-to-toe alignment straight."],
                "Tamil": ["⚠️ பட்டேலர் திரிபு: முழங்கால்களை முழுமையாக நீட்டி லாக் செய்ய வேண்டாம்.", "முழங்கால்-விரல் சீரமைப்பை நேராக வைக்கவும்."],
                "Hindi": ["⚠️ पटेला खिंचाव: घुटनों को ऊपर की ओर लॉक करने से बचें।", "फॉर्म चेक: घुटने से पैर के अंगूठे का संरेखण सीधा रखें।"]
            },
            "general": {
                "English": ["💡 Maintenance: Strengthen hamstrings.", "Apply proper warm-up."],
                "Tamil": ["💡 பராமரிப்பு: தொடைப் பின் தசைகளை வலுப்படுத்தவும்.", "முறையான வார்ம்-அப் செய்யவும்."],
                "Hindi": ["💡 रखरखाव: हैमस्ट्रिंग को मजबूत करें।", "उचित वार्म-अप करें।"]
            }
        },
        "Right Shoulder": {
            "critical": {
                "English": ["⚠️ Rotator Cuff Stress: Keep elbow tucked closer to torso.", "Reduce weight to maintain full range of motion."],
                "Tamil": ["⚠️ சுழற்சி தசை அழுத்தம்: முழங்கையை உடலுக்கு நெருக்கமாக வைக்கவும்.", "இயக்கத்தின் அளவை பராமரிக்க எடையை குறைக்கவும்."],
                "Hindi": ["⚠️ रोटेटर कफ तनाव: कोहनी को धड़ के करीब रखें।", "गति की पूरी सीमा बनाए रखने के लिए वजन कम करें।"]
            },
            "general": {
                "English": ["💡 Maintenance: Focus on scapular retraction.", "Perform face-pulls."],
                "Tamil": ["💡 பராமரிப்பு: தோள்பட்டை எலும்புகளை பின்னுக்கு இழுப்பதில் கவனம் செலுத்துங்கள்.", "பேஸ்-புல் (Face-pull) பயிற்சிகளை செய்யவும்."],
                "Hindi": ["💡 रखरखाव: स्कैपुलर रिट्रेक्शन पर ध्यान दें।", "फेस-पुल व्यायाम करें।"]
            }
        },
        "Left Shoulder": {
            "critical": {
                "English": ["⚠️ Impingement Risk: Avoid lifting arm behind torso plane.", "Stop if you feel sharp pain."],
                "Tamil": ["⚠️ இம்பிஞ்ச்மென்ட் அபாயம்: கையை உடலுக்கு பின்னால் அதிகமாக தூக்குவதை தவிர்க்கவும்.", "கடுமையான வலி தெரிந்தால் உடனடியாக நிறுத்தவும்."],
                "Hindi": ["⚠️ इम्पिंगमेंट जोखिम: हाथ को धड़ के पीछे उठाने से बचें।", "तेज दर्द होने पर रुक जाएं।"]
            },
            "general": {
                "English": ["💡 Maintenance: thoracic spine mobility.", "Warm up with rotations."],
                "Tamil": ["💡 பராமரிப்பு: முதுகுத்தண்டு அசைவுத்திறனை அதிகரிக்கவும்.", "சுழற்சி பயிற்சிகளுடன் வார்ம்-அப் செய்யவும்."],
                "Hindi": ["💡 रखरखाव: थोरैसिक रीढ़ की गतिशीलता।", "रोटेशन के साथ वार्म अप करें।"]
            }
        },
        "Trunk": {
            "critical": {
                "English": ["🚨 Spinal Integrity: Your lower back is rounding! Neutralize spine.", "Engagement: Use bracing method."],
                "Tamil": ["🚨 முதுகுத்தண்டு பாதுகாப்பு: உங்கள் கீழ் முதுகு வளைகிறது! முதுகை நேராக வைக்கவும்.", "வயிற்று தசைகளை இறுக்கமாக (Bracing) பிடிக்கவும்."],
                "Hindi": ["🚨 रीढ़ की अखंडता: आपकी निचली पीठ झुक रही है! रीढ़ को तटस्थ रखें।", "ब्रेसिंग विधि का प्रयोग करें।"]
            },
            "general": {
                "English": ["💡 Maintenance: Engage lats.", "Check your breathing."],
                "Tamil": ["💡 பராமரிப்பு: லாட்ஸ் (Lats) தசைகளை பயன்படுத்தவும்.", "சுவாசத்தை சீராக வைக்கவும்."],
                "Hindi": ["💡 रखरखाव: लैट्स को शामिल करें।", "अपनी सांस लेने की जांच करें।"]
            }
        },
        "Neck": {
            "critical": {
                "English": ["⚠️ Cervical Strain: Stop looking up or at the mirror.", "Fix: Neutral Neck (Gaze at floor)."],
                "Tamil": ["⚠️ கழுத்து திரிபு: மேலே அல்லது கண்ணாடியை பார்ப்பதை தவிர்க்கவும்.", "சரிசெய்தல்: தரையை பார்த்து பயிற்சியை செய்யவும் (Neutral Neck)."],
                "Hindi": ["⚠️ सर्वाइकल स्ट्रेन: ऊपर या आईने में देखना बंद करें।", "फिक्स: न्यूट्रल नेक (फर्श पर देखें)।"]
            },
            "general": {
                "English": ["💡 Maintenance: Relax your traps.", "Keep chin tucked slightly."],
                "Tamil": ["💡 பராமரிப்பு: கழுத்து தசைகளை தளர்வாக வைக்கவும்.", "தாடையை சற்றே உள்ளிழுத்து வைக்கவும்."],
                "Hindi": ["💡 रखरखाव: अपने ट्रैप्स को रिलैक्स रखें।", "ठुड्डी को थोड़ा अंदर रखें।"]
            }
        }
    }

    for joint, val in joint_avg.items():
        if joint in kb_content:
            if val > 20:
                recommendations[joint] = kb_content[joint]["critical"][lang]
            elif val > 5:
                recommendations[joint] = kb_content[joint]["general"][lang]
            
    return recommendations
# -------------------------------------------------
# RISK CALCULATION
# -------------------------------------------------
def calculate_risk(angles, prev_angles):
    joint_risk = {}

    for joint, angle in angles.items():
        score = 0

        if angle < 30 or angle > 175:
            score += 30

        if prev_angles:
            change = abs(angle - prev_angles[joint])
            if change > 40:
                score += 40

        joint_risk[joint] = min(score, 100)

    knee_diff = abs(angles["Left Knee"] - angles["Right Knee"])
    hip_diff = abs(angles["Left Hip"] - angles["Right Hip"])
    symmetry_risk = min((knee_diff + hip_diff) / 2, 40)
    trunk_risk = 30 if angles["Trunk"] < 100 else 0

    overall_risk = int(np.mean(list(joint_risk.values())) +
                       symmetry_risk +
                       trunk_risk)

    return min(overall_risk, 100), joint_risk

# -------------------------------------------------
# UPLOAD & RECORD SECTION (FIXED ALIGNMENT)
# -------------------------------------------------
st.markdown(f"""
<div style="background:black; padding:30px; border-radius:15px; box-shadow:0 10px 25px rgba(0,0,0,0.1); margin-top:20px;">
<h3 style='text-align:center'>{T['upload']}</h3>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<style>
div.stButton > button:first-child {
    background: linear-gradient(90deg, #9333ea, #4c1d95);
    border: 1px solid rgba(255,255,255,0.2);
    margin-top: 32px; /* This aligns it perfectly with the upload box */
    height: 45px;
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_video = st.file_uploader(
        T["drag"], 
        type=["mp4","avi","mov","mpeg4", "webm"]
    )

with col2:
    st.markdown("<div style='height:3px;'></div>", unsafe_allow_html=True)
    
    
    show_recorder = st.button(T["take_live"], use_container_width=True)


if show_recorder:
    st.info(f"💡 {T['rec_instr']}")
    
    
    
    components.html(
        f"""
        <div style="text-align: center; font-family: 'Segoe UI', sans-serif; color: white;">
            <video id="preview" width="100%" height="auto" autoplay muted 
                style="background: #000; border-radius: 15px; border: 2px solid #9333ea; max-height: 500px;">
            </video>
            <br><br>
            <div style="display: flex; justify-content: center; gap: 20px;">
                <button id="startBtn" style="background: linear-gradient(90deg,#28a745,#218838); color: white; padding: 15px 30px; border: none; border-radius: 10px; cursor: pointer; font-weight: bold; font-size: 16px;">
                    {T['rec_start']}
                </button>
                <button id="stopBtn" disabled style="background: linear-gradient(90deg,#dc3545,#c82333); color: white; padding: 15px 30px; border: none; border-radius: 10px; cursor: pointer; font-weight: bold; font-size: 16px;">
                    {T['rec_stop']}
                </button>
            </div>
        </div>

        <script>
            let stream, recorder, chunks = [];
            const preview = document.getElementById('preview');
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');

            async function setup() {{
                try {{
                    stream = await navigator.mediaDevices.getUserMedia({{ video: {{ width: 1280, height: 720 }}, audio: false }});
                    preview.srcObject = stream;
                }} catch (err) {{
                    console.error("Camera error:", err);
                }}
            }}
            setup();

            startBtn.onclick = () => {{
                chunks = [];
                recorder = new MediaRecorder(stream, {{ mimeType: 'video/webm' }});
                recorder.ondataavailable = e => chunks.push(e.data);
                recorder.onstop = () => {{
                    const blob = new Blob(chunks, {{ type: 'video/webm' }});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'training_capture.webm';
                    a.click();
                }};
                recorder.start();
                startBtn.disabled = true;
                startBtn.style.opacity = "0.5";
                stopBtn.disabled = false;
            }};

            stopBtn.onclick = () => {{
                recorder.stop();
                startBtn.disabled = false;
                startBtn.style.opacity = "1";
                stopBtn.disabled = true;
            }};
        </script>
        """,
        height=600,
    )
if uploaded_video:

    video_bytes = uploaded_video.getvalue()
    st.video(video_bytes)

    if st.button("▶ " + T["analyze"]):
     with st.spinner(" Analyzing Video... Please wait"):

        st.session_state.risk_history = []
        st.session_state.joint_history = []

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(video_bytes)
        temp_file.close()

        cap = cv2.VideoCapture(temp_file.name)
        # Progress bar setup
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)
        current_frame = 0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 25
        if fps == 0:
            fps = 25
        fps = fps / 2

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*'avc1') # Raw MP4 support for Browsers
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        mp_pose = mp_solutions.pose
        mp_draw = mp_solutions.drawing_utils
        prev_angles = None

        # STEP 1: Strict Validation Flags
        human_detected_count = 0
        total_frames_checked = 0
        MAX_INITIAL_CHECK = 15 

        with mp_pose.Pose() as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                current_frame += 1
                total_frames_checked += 1
                progress_bar.progress(min(current_frame/frame_count, 1.0))

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                # STEP 2: Logic trigger ONLY if actual human landmarks are found
                if results.pose_landmarks:
                    human_detected_count += 1
                    lm = results.pose_landmarks.landmark
                    angles = get_body_angles(lm, mp_pose)
                    risk, joint_risk = calculate_risk(angles, prev_angles)
                    prev_angles = angles.copy()

                    # Store ONLY if human is actually present
                    st.session_state.risk_history.append(risk)
                    st.session_state.joint_history.append(joint_risk)

                    # --- YOUR PRECIOUS UI LOGIC (DO NOT REMOVE) ---
                    mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (10, 10), (800, 620), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                    
                    y = 90
                    cv2.putText(frame, f"Overall Risk: {risk}%", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                    y += 80
                    for joint, value in joint_risk.items():
                        cv2.putText(frame, f"{joint}: {value}%", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        y += 38
                    highest_joint = max(joint_risk, key=joint_risk.get)
                    highest_value = joint_risk[highest_joint]
                    y += 25
                    cv2.putText(frame, f"Highest Stress: {highest_joint} ({highest_value}%)", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)
                
                else:
                    # Overlay warning if no human in this frame
                    cv2.putText(frame, "NO HUMAN DETECTED", (150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

                out.write(frame)

                # STEP 3: Early Exit Check 
                if total_frames_checked >= MAX_INITIAL_CHECK and human_detected_count < 2:
                    break 

        cap.release()
        out.release()
        progress_bar.empty()
        cv2.destroyAllWindows()

        # STEP 4: Final Lockdown 
        if human_detected_count < 2:
            
            st.session_state.risk_history = []
            st.session_state.joint_history = []
            st.session_state.output_path = None
            st.error(T["err_human"])
        else:
            st.session_state.output_path = output_path
            st.success("✅ " + T["complete"])

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# DISPLAY ANALYZED RESULT (COMPACT SINGLE VIEW)
# -------------------------------------------------
if st.session_state.output_path is not None:
    st.write("---")
    
    res_label = " Analyzed Result" if language == "English" else " பகுப்பாய்வு முடிவு"
    
    
    v_col1, v_col2, v_col3 = st.columns([1, 6, 1]) 
    
    with v_col2:
        st.markdown(f"<h3 style='text-align: center; color: #4c1d95;'>{res_label}</h3>", unsafe_allow_html=True)

        try:
            video_file = open(st.session_state.output_path, 'rb')
            v_bytes = video_file.read()
            
           
            st.markdown("""
                <style>
                .stVideo {
                    max-height: 450px !important;
                    width: auto !important;
                    margin: 0 auto;
                    display: block;
                }
                </style>
            """, unsafe_allow_html=True)
            
            st.video(v_bytes, format="video/mp4")

        except Exception as e:
            st.error(f"Error: {e}")

       
        st.download_button(
            label="⬇ Download Video", 
            data=v_bytes,
            file_name="Analyzed_Video.mp4",
            mime="video/mp4",
            use_container_width=True
        )

# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------
if len(st.session_state.risk_history) > 0:

    st.subheader(T["risktrend"])

    df = pd.DataFrame({
        "Frame": range(len(st.session_state.risk_history)),
        "Risk": st.session_state.risk_history
    })

    st.line_chart(df.set_index("Frame"))

    avg_risk = int(np.mean(st.session_state.risk_history))

    st.subheader(T["final"])

    if avg_risk < 25:
        st.success(f"LOW RISK ({avg_risk}%)")
    elif avg_risk < 55:
        st.warning(f"MODERATE RISK ({avg_risk}%)")
    else:
        st.error(f"HIGH RISK ({avg_risk}%)")

    if len(st.session_state.joint_history) > 0:
        st.subheader("Joint-Specific Risk")

        joint_df = pd.DataFrame(st.session_state.joint_history)
        joint_avg = joint_df.mean().astype(int)

        cols = st.columns(3)
        for i, joint in enumerate(joint_avg.index):
            cols[i % 3].metric(joint, f"{joint_avg[joint]}%")

        highest_joint = joint_avg.idxmax()
        highest_value = joint_avg.max()

        st.info(f"⚠ Highest observed stress: {highest_joint} ({highest_value}%)")
# --- DISPLAY LOGIC ---
        st.subheader(T["ai_coach"])
        
      
        recommendations = get_ai_recommendations(joint_avg, language)
        
        if recommendations:
            for joint, tips in recommendations.items():
                icon = "🚨" if joint_avg[joint] > 20 else "💡"
                
                
                ana_label = "Analysis" if language == "English" else "பகுப்பாய்வு" if language == "Tamil" else "विश्लेषण"
                str_label = "Stress" if language == "English" else "அழுத்தம்" if language == "Tamil" else "तनाव"
                
                # Translating Joint Name for display
                t_joint = translate_text(joint) 
                
                with st.expander(f"{icon} {t_joint} {ana_label} ({joint_avg[joint]}% {str_label})"):
                    for tip in tips:
                        st.write(f"• {tip}")
        else:
            st.success(T["ai_perfect"])
# --- PDF REPORT DOWNLOAD ---
        st.write("---")
        pdf_buffer = generate_pdf_report(
            st.session_state.risk_history, 
            st.session_state.joint_history, 
            avg_risk
        )
        
        st.download_button(
    label=T["pdf_btn"],
    data=pdf_buffer,
    file_name="Injury_Report.pdf",
    mime="application/pdf",
    use_container_width=True
)