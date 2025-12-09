import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
import csv
import pandas as pd
from PIL import Image
import time


class FaceAttendanceSystem:
    def __init__(self):
        self.known_face_names = []
        self.descriptors_db = {}
        self.attendance_file = "attendance.csv"
        self.registered_faces_dir = "registered_faces"

        if not os.path.exists(self.registered_faces_dir):
            os.makedirs(self.registered_faces_dir)

        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Subject', 'Date', 'Time'])

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.load_known_faces()

    def extract_face_descriptors(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_rects = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        if len(faces_rects) == 0:
            return None, None

        (x, y, w, h) = faces_rects[0]
        face_roi = gray[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (200, 200))

        keypoints, descriptors = self.orb.detectAndCompute(face_roi, None)
        return face_roi, descriptors

    def load_known_faces(self):
        self.known_face_names = []
        self.descriptors_db = {}

        for filename in os.listdir(self.registered_faces_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(self.registered_faces_dir, filename)

                img = cv2.imread(image_path)
                if img is None:
                    continue

                _, des = self.extract_face_descriptors(img)
                if des is None:
                    continue

                if name not in self.descriptors_db:
                    self.descriptors_db[name] = des
                    self.known_face_names.append(name)
                else:
                    self.descriptors_db[name] = np.vstack([self.descriptors_db[name], des])

    def mark_attendance(self, name, subject):
        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d")
        time_string = now.strftime("%H:%M:%S")

        marked_today = False
        try:
            with open(self.attendance_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) >= 4:
                        if row[0] == name and row[1] == subject and row[2] == date_string:
                            marked_today = True
                            break
        except FileNotFoundError:
            pass

        if not marked_today:
            with open(self.attendance_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name, subject, date_string, time_string])
            return True, time_string
        else:
            return False, None

    def recognize_face(self, face_img):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        gray = cv2.resize(gray, (200, 200))
        kp_q, des_q = self.orb.detectAndCompute(gray, None)

        if des_q is None or len(kp_q) == 0:
            return "Unknown"

        best_name = "Unknown"
        best_good_matches = 0
        best_ratio = 0.0

        for name, des_ref in self.descriptors_db.items():
            matches = self.matcher.match(des_ref, des_q)
            if not matches:
                continue

            matches = sorted(matches, key=lambda m: m.distance)
            good_matches = [m for m in matches if m.distance < 70]

            num_good = len(good_matches)
            ratio = num_good / max(len(matches), 1)

            if num_good > best_good_matches:
                best_good_matches = num_good
                best_ratio = ratio
                best_name = name

        if best_good_matches >= 8 and best_ratio >= 0.15:
            return best_name
        else:
            return "Unknown"


# ------------------ Streamlit App ------------------

# Initialize system
if 'system' not in st.session_state:
    st.session_state.system = FaceAttendanceSystem()
    st.session_state.camera_active = False

# Page config
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Single navigation state
if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠 Dashboard"

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    .stApp {
        background: transparent;
    }
    
    h1, h2, h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .stat-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.2);
    }
    
    .stat-number {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .stat-label {
        color: #666;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        animation: slideIn 0.5s ease;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        animation: slideIn 0.5s ease;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .dataframe {
        background: white;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR NAVIGATION (buttons) ----------
with st.sidebar:
    st.markdown("# 👤 Navigation")

    if st.button("🏠 Dashboard", use_container_width=True):
        st.session_state.current_page = "🏠 Dashboard"
    if st.button("➕ Register Face", use_container_width=True):
        st.session_state.current_page = "➕ Register Face"
    if st.button("✅ Mark Attendance", use_container_width=True):
        st.session_state.current_page = "✅ Mark Attendance"
    if st.button("📊 View Records", use_container_width=True):
        st.session_state.current_page = "📊 View Records"
    if st.button("👥 Registered Users", use_container_width=True):
        st.session_state.current_page = "👥 Registered Users"

    st.markdown("---")
    st.markdown("### 📱 System Info")
    st.info(f"**Total Users:** {len(st.session_state.system.known_face_names)}")

    # Count today's attendance
    today = datetime.now().strftime("%Y-%m-%d")
    today_count = 0
    try:
        with open(st.session_state.system.attendance_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 3 and row[2] == today:
                    today_count += 1
    except:
        pass

    st.success(f"**Today's Attendance:** {today_count}")

# Single source of truth for page
page = st.session_state.current_page

# =============== PAGES ===============

# DASHBOARD
if page == "🏠 Dashboard":
    st.markdown("# 🎯 Smart Face Recognition Attendance System")
    st.markdown("### Welcome to the future of attendance management!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-number">{len(st.session_state.system.known_face_names)}</p>
            <p class="stat-label">👥 Registered Users</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-number">{today_count}</p>
            <p class="stat-label">📅 Today's Attendance</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        # Total attendance records (simple row count minus header)
        total_records = 0
        try:
            with open(st.session_state.system.attendance_file, 'r') as f:
                total_records = max(sum(1 for line in f) - 1, 0)
        except:
            pass

        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-number">{total_records}</p>
            <p class="stat-label">📝 Total Records</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🚀 Quick Actions")
        if st.button("➕ Register New Face", use_container_width=True, key="qa_register"):
            st.session_state.current_page = "➕ Register Face"
            st.rerun()
        if st.button("✅ Mark Attendance", use_container_width=True, key="qa_attendance"):
            st.session_state.current_page = "✅ Mark Attendance"
            st.rerun()

    with col2:
        st.markdown("### 📊 Reports")
        if st.button("📋 View All Records", use_container_width=True, key="qa_records"):
            st.session_state.current_page = "📊 View Records"
            st.rerun()
        if st.button("👥 See Registered Users", use_container_width=True, key="qa_users"):
            st.session_state.current_page = "👥 Registered Users"
            st.rerun()

    st.markdown("---")
    st.markdown("### 🎨 Features")
    features_col1, features_col2 = st.columns(2)

    with features_col1:
        st.markdown("""
        - 🔒 **Secure Face Recognition** using OpenCV ORB  
        - 📸 **Real-time Detection** with browser webcam   
        - 💾 **Automatic Attendance Logging**
        """)

    with features_col2:
        st.markdown("""
        - 📊 **Detailed Reports & Analytics**  
        - 👥 **Multi-user Support**  
        - 🎯 **Subject-wise Attendance**
        """)

# REGISTER FACE
elif page == "➕ Register Face":
    st.markdown("# ➕ Register New Face")
    st.markdown("### Add a new person to the attendance system")

    name = st.text_input("👤 Enter Person's Name:", placeholder="e.g., John Doe")

    col1, col2 = st.columns([2, 1])

    with col1:
        img_file = st.camera_input("📸 Take a picture (browser camera)")

        if img_file is not None and name:
            if st.button("💾 Save & Register", use_container_width=True):
                existing = [
                    f for f in os.listdir(st.session_state.system.registered_faces_dir)
                    if os.path.splitext(f)[0].lower() == name.lower()
                ]

                if existing:
                    st.markdown(
                        f'<div class="warning-box">⚠️ {name} is already registered!</div>',
                        unsafe_allow_html=True
                    )
                else:
                    image = Image.open(img_file)
                    img_array = np.array(image)
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                    face_img, des = st.session_state.system.extract_face_descriptors(img_bgr)

                    if face_img is None or des is None:
                        st.markdown(
                            '<div class="warning-box">❌ No valid face detected! Please try again.</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        image_path = os.path.join(
                            st.session_state.system.registered_faces_dir,
                            f"{name}.jpg"
                        )
                        cv2.imwrite(image_path, face_img)

                        st.session_state.system.load_known_faces()

                        st.markdown(
                            f'<div class="success-box">✅ {name} registered successfully!</div>',
                            unsafe_allow_html=True
                        )
                        st.balloons()
                        time.sleep(2)
                        st.rerun()

    with col2:
        st.markdown("### 📝 Instructions")
        st.info("""
        1. Enter the person's name  
        2. Click 'Take a picture' (browser will ask for camera permission)  
        3. Position face clearly  
        4. Click capture  
        5. Save & Register  
        """)

# MARK ATTENDANCE
elif page == "✅ Mark Attendance":
    st.markdown("# ✅ Mark Attendance")
    st.markdown("### Start face recognition for attendance")

    if len(st.session_state.system.descriptors_db) == 0:
        st.markdown(
            '<div class="warning-box">⚠️ No registered faces found! Please register at least one face first.</div>',
            unsafe_allow_html=True
        )
    else:
        subject = st.text_input(
            "📚 Enter Subject Name:",
            placeholder="e.g., DBMS, COA, Mathematics"
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            img_file = st.camera_input("📸 Capture face for attendance (browser camera)")

            if img_file is not None and subject:
                image = Image.open(img_file)
                img_array = np.array(image)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                faces_rects = st.session_state.system.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
                )

                if len(faces_rects) > 0:
                    (x, y, w, h) = faces_rects[0]
                    face_roi = img_bgr[y:y + h, x:x + w]

                    name = st.session_state.system.recognize_face(face_roi)

                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 3)
                    cv2.putText(
                        img_bgr, name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
                    )

                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, caption="Recognition Result", use_container_width=True)

                    if name != "Unknown":
                        success, time_str = st.session_state.system.mark_attendance(name, subject)
                        if success:
                            st.markdown(
                                f'<div class="success-box">✅ Attendance marked for {name} in {subject} at {time_str}</div>',
                                unsafe_allow_html=True
                            )
                            st.balloons()
                        else:
                            st.markdown(
                                f'<div class="warning-box">ℹ️ {name} already marked present today for {subject}</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        st.markdown(
                            '<div class="warning-box">❌ Face not recognized! Please register first.</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown(
                        '<div class="warning-box">❌ No face detected in image!</div>',
                        unsafe_allow_html=True
                    )

        with col2:
            st.markdown("### 📝 Instructions")
            st.info("""
            1. Enter subject name  
            2. Capture your face using the browser camera  
            3. System will recognize  
            4. Attendance marked!  
            """)

# VIEW RECORDS + DELETE
elif page == "📊 View Records":
    st.markdown("# 📊 Attendance Records")
    st.markdown("### Complete attendance history")

    attendance_file = st.session_state.system.attendance_file

    try:
        df = pd.read_csv(attendance_file)

        if len(df) > 0:
            col1, col2, col3 = st.columns(3)

            with col1:
                date_filter = st.date_input("📅 Filter by Date (Optional)")
            with col2:
                names = ["All"] + sorted(df['Name'].unique().tolist())
                name_filter = st.selectbox("👤 Filter by Name", names)
            with col3:
                subjects = ["All"] + sorted(df['Subject'].unique().tolist())
                subject_filter = st.selectbox("📚 Filter by Subject", subjects)

            filtered_df = df.copy()

            if date_filter:
                date_str = date_filter.strftime("%Y-%m-%d")
                filtered_df = filtered_df[filtered_df['Date'] == date_str]

            if name_filter != "All":
                filtered_df = filtered_df[filtered_df['Name'] == name_filter]
            if subject_filter != "All":
                filtered_df = filtered_df[filtered_df['Subject'] == subject_filter]

            st.markdown(f"### 📋 Showing {len(filtered_df)} records")
            df_show = filtered_df.copy()
            df_show.index = df_show.index + 1   # 1, 2, 3, ...

            st.dataframe(df_show, use_container_width=True, height=300)

            # Delete individual record
            st.markdown("### 🗑 Delete an Individual Record")

            if len(filtered_df) > 0:
                temp_df = filtered_df.reset_index(drop=False)  # keeps original index
                options = [
                    f"{row['index']} | {row['Name']} | {row['Subject']} | {row['Date']} {row['Time']}"
                    for _, row in temp_df.iterrows()
                ]

                choice = st.selectbox(
                    "Select a record to delete:",
                    ["None"] + options
                )

                if choice != "None":
                    if st.button("🗑 Delete Selected Record"):
                        idx_str = choice.split("|")[0].strip()
                        try:
                            idx_to_delete = int(idx_str)
                            full_df = pd.read_csv(attendance_file)
                            if 0 <= idx_to_delete < len(full_df):
                                full_df = full_df.drop(index=idx_to_delete)
                                full_df.to_csv(attendance_file, index=False)
                                st.success("✅ Record deleted successfully.")
                                st.rerun()
                            else:
                                st.error("Invalid row index.")
                        except Exception as e:
                            st.error(f"Error while deleting record: {e}")
            else:
                st.info("No records match the current filter to delete.")

            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Records as CSV",
                data=csv_data,
                file_name=f"attendance_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No attendance records found yet.")
    except FileNotFoundError:
        st.info("No attendance file found. Start marking attendance to create records!")

# REGISTERED USERS + DELETE
elif page == "👥 Registered Users":
    st.markdown("# 👥 Registered Users")
    st.markdown("### All registered people in the system")

    system = st.session_state.system

    if len(system.known_face_names) == 0:
        st.info("No users registered yet. Register your first user!")
    else:
        cols = st.columns(4)

        for idx, name in enumerate(system.known_face_names):
            with cols[idx % 4]:
                image_path_jpg = os.path.join(system.registered_faces_dir, f"{name}.jpg")
                image_path_png = os.path.join(system.registered_faces_dir, f"{name}.png")
                image_path_jpeg = os.path.join(system.registered_faces_dir, f"{name}.jpeg")

                img_path = None
                for p in [image_path_jpg, image_path_png, image_path_jpeg]:
                    if os.path.exists(p):
                        img_path = p
                        break

                if img_path and os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, caption=name, use_container_width=True)
                else:
                    st.markdown(f"""
                    <div class="stat-card">
                        <p style="font-size: 2rem;">👤</p>
                        <p style="color: #666;">{name}</p>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🗑 Delete a Registered User")

        user_to_delete = st.selectbox(
            "Select a user to delete (face file will be removed):",
            ["None"] + system.known_face_names
        )

        if user_to_delete != "None":
            if st.button("🗑 Delete Selected User"):
                deleted_any = False
                for ext in [".jpg", ".jpeg", ".png"]:
                    path = os.path.join(system.registered_faces_dir, user_to_delete + ext)
                    if os.path.exists(path):
                        os.remove(path)
                        deleted_any = True

                if deleted_any:
                    system.load_known_faces()
                    st.success(f"✅ Deleted face data for user: {user_to_delete}")
                    st.rerun()
                else:
                    st.error("No image file found for this user.")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: white;'>Made with ❤️ using Streamlit & OpenCV</p>",
    unsafe_allow_html=True
)
