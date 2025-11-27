import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import math
import os
# Importing kagglehub, but it may require authentication to work reliably.
# The code includes a robust fallback to generated data.
try:
    import kagglehub
except ImportError:
    print("Warning: kagglehub library not found. Cannot download real data. Using sample data.")
    kagglehub = None

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
# Using an instance of Pose outside the class for global access
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

class PushUpTrainer:
    """Handles dataset loading, feature extraction, and model training/prediction."""
    def __init__(self):
        self.model = None
        self.dataset = []
        self.labels = []
        
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points (e.g., shoulder, elbow, wrist)."""
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        ba = a - b
        bc = c - b
        
        # Calculate dot product and norms
        dot_product = np.dot(ba, bc)
        norm_product = np.linalg.norm(ba) * np.linalg.norm(bc)
        
        # Avoid division by zero and handle tiny values
        if norm_product == 0:
            return 180.0
            
        cosine_angle = dot_product / norm_product
        
        # Clamp value to [-1, 1] to prevent math domain errors from floating point inaccuracy
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def extract_features_from_landmarks(self, landmarks):
        """Extract features (angles and body alignment ratios) from MediaPipe landmarks."""
        features = []
        
        # Get coordinates for key joints (Normalized x, y coordinates from MediaPipe)
        
        # Left side
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        
        # Right side
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        
        # Calculate angles
        left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_shoulder_angle = self.calculate_angle(left_elbow, left_shoulder, left_hip)
        right_shoulder_angle = self.calculate_angle(right_elbow, right_shoulder, right_hip)
        
        # Body alignment (Normalized distance between shoulder and hip)
        # 
        # FIX: Corrected the distance calculation using the proper squared difference (Pythagorean)
        left_shoulder_hip_dist = math.sqrt((left_shoulder[0]-left_hip[0])**2 + (left_shoulder[1]-left_hip[1])**2)
        right_shoulder_hip_dist = math.sqrt((right_shoulder[0]-right_hip[0])**2 + (right_shoulder[1]-right_hip[1])**2)
        
        features.extend([
            left_elbow_angle, right_elbow_angle,
            left_shoulder_angle, right_shoulder_angle,
            left_shoulder_hip_dist, right_shoulder_hip_dist
        ])
        
        return features
    
    def download_and_load_dataset(self):
        """Download and load the real push-up dataset from Kaggle or use fallback."""
        if not kagglehub:
            print("KaggleHub not available. Using sample data.")
            self.generate_sample_data()
            return
            
        print("Attempting to download push-up dataset from Kaggle...")
        
        try:
            # Download the dataset
            path = kagglehub.dataset_download("mohamadashrafsalama/pushup")
            print(f"Dataset downloaded to: {path}")
            
            # Look for CSV files in the downloaded directory
            csv_files = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
            
            if not csv_files:
                print("No CSV files found in the dataset. Generating sample data instead.")
                self.generate_sample_data()
                return
            
            print(f"Found {len(csv_files)} CSV files")
            
            all_data = []
            all_labels = []
            
            # Heuristic to load relevant data
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    
                    # Look for angle columns (features) and label column
                    angle_columns = [col for col in df.columns if 'angle' in col.lower() or 'elbow' in col.lower() or 'shoulder' in col.lower()]
                    label_columns = [col for col in df.columns if 'label' in col.lower() or 'posture' in col.lower() or 'correct' in col.lower() or 'class' in col.lower()]
                    
                    if angle_columns and label_columns:
                        # Use data only if it looks relevant (at least 6 feature columns and 1 label column)
                        if len(angle_columns) >= 4 and len(label_columns) >= 1:
                            # Use the intersection of our expected features with available columns
                            expected_cols = ['left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle', 'right_shoulder_angle']
                            final_features = [col for col in angle_columns if col.lower() in expected_cols]
                            
                            features = df[final_features].values
                            labels = df[label_columns[0]].values
                            
                            # Simple cleanup for label encoding (assuming 1=Correct, 0=Incorrect/Other)
                            unique_labels = np.unique(labels)
                            if len(unique_labels) > 2:
                                # Simplistic conversion: treat the most frequent label as "1" (Correct) and others as "0"
                                label_map = {unique_labels[i]: 1 if i == 0 else 0 for i in range(len(unique_labels))}
                                labels = np.array([label_map.get(l, 0) for l in labels])
                            
                            # Pad features if necessary (This assumes 4 main angles are the key features from the dataset)
                            # We will skip the body alignment features if the real dataset doesn't have them
                            if features.shape[1] == 4:
                                # For consistency, pad with a dummy value (e.g., mean hip distance from sample data)
                                # NOTE: The live extraction uses 6 features. Training on 4 features is a compromise.
                                print(f"Warning: Dataset has 4 features, skipping body alignment classification.")
                                features_6 = np.hstack([features, np.full((features.shape[0], 2), 0.15)]) # Pad with 0.15 for compatibility
                                features = features_6
                            
                            all_data.extend(features.tolist())
                            all_labels.extend(labels.tolist())
                            
                        else:
                            print(f"Skipping {csv_file}: Insufficient angle or label columns.")

                except Exception as e:
                    print(f"Error processing {csv_file}: {e}")
                    continue
            
            if all_data:
                self.dataset = all_data
                self.labels = all_labels
                print(f"Successfully loaded {len(self.dataset)} samples from real dataset.")
                # Ensure all samples have 6 features
                self.dataset = [d for d in self.dataset if len(d) == 6]
                self.labels = self.labels[:len(self.dataset)]
                print(f"Final loaded samples after filtering: {len(self.dataset)}")
            else:
                print("No valid data found in CSV files. Generating sample data.")
                self.generate_sample_data()
                
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Generating sample data instead.")
            self.generate_sample_data()
    
    def generate_sample_data(self):
        """Generate sample push-up dataset as robust fallback."""
        print("Generating sample push-up dataset...")
        
        # Correct push-up features (good form, low elbow angle, straight body)
        for _ in range(200):
            features = [
                np.random.normal(85, 10),    # Left elbow angle (low, near 90 or below) 
                np.random.normal(85, 10),    # Right elbow angle
                np.random.normal(175, 5),    # Left shoulder angle (straight back near 180)
                np.random.normal(175, 5),    # Right shoulder angle
                np.random.normal(0.15, 0.02), # Left shoulder-hip distance (small variation)
                np.random.normal(0.15, 0.02)  # Right shoulder-hip distance
            ]
            self.dataset.append(features)
            self.labels.append(1)  # 1 = correct posture
        
        # Incorrect push-up features (sagging hips, high body alignment distance)
        for _ in range(200):
            features = [
                np.random.normal(90, 15),
                np.random.normal(90, 15),
                np.random.normal(160, 10), # Slight hip sag/pike makes this angle smaller
                np.random.normal(160, 10),
                np.random.normal(0.2, 0.05), # Hip sag increases normalized shoulder-hip distance
                np.random.normal(0.2, 0.05)
            ]
            self.dataset.append(features)
            self.labels.append(0) # 0 = incorrect posture
            
        print(f"Generated {len(self.dataset)} sample samples (6 features each)")
    
    def train_model(self):
        """Train the posture classification model."""
        if not self.dataset:
            self.download_and_load_dataset()
        
        X = np.array(self.dataset)
        y = np.array(self.labels)
        
        if len(X) < 10 or len(np.unique(y)) < 2:
            print("Error: Not enough data or classes to train model.")
            return 0.0
            
        print(f"Training model with {len(X)} samples")
        print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train Random Forest classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        
        # Feature importance
        feature_names = ['Left_Elbow', 'Right_Elbow', 'Left_Shoulder', 'Right_Shoulder', 
                         'Left_Body_Align', 'Right_Body_Align']
        importances = self.model.feature_importances_
        print("Feature importances:")
        for name, importance in zip(feature_names, importances):
            print(f"  {name}: {importance:.3f}")
        
        return accuracy
    
    def save_model(self, filename='pushup_model.joblib'):
        """Save the trained model."""
        if self.model:
            joblib.dump(self.model, filename)
            print(f"Model saved as {filename}")
    
    def load_model(self, filename='pushup_model.joblib'):
        """Load a pre-trained model."""
        try:
            # Check if model file exists locally
            if not os.path.exists(filename):
                print(f"Model file {filename} not found.")
                return False
                
            self.model = joblib.load(filename)
            print(f"Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            print("No pre-trained model found. Please train a new model.")
            return False

class PushUpCounter:
    """Handles the state machine for counting reps based on angle thresholds."""
    # FIX: Angle calculator method is passed during initialization
    def __init__(self, angle_calculator):
        self.count = 0
        self.state = "up"  # "up" or "down"
        self.elbow_angle_threshold = 100 # Adjusted threshold for typical push-up depth
        self.angle_calculator = angle_calculator
        
    def update(self, landmarks):
        """Update push-up count based on elbow angles."""
        if not landmarks:
            return False
        
        # Get coordinates for angle calculation (Left side is sufficient for counting)
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        # Calculate elbow angle using the method passed from trainer
        elbow_angle = self.angle_calculator(left_shoulder, left_elbow, left_wrist)
        
        # State machine for counting (down-to-up transition is a completed rep)
        if self.state == "up" and elbow_angle < self.elbow_angle_threshold:
            self.state = "down"
        elif self.state == "down" and elbow_angle > (self.elbow_angle_threshold + 10): # Requires going back up sufficiently
            self.state = "up"
            self.count += 1
            return True
        
        return False

def main():
    """Main execution function for the push-up tracker application."""
    # Initialize components
    trainer = PushUpTrainer()
    # FIX: Pass the angle calculation method to the counter to avoid redundant object creation
    counter = PushUpCounter(trainer.calculate_angle) 
    
    # Try to load pre-trained model, else train new one with real data
    if not trainer.load_model():
        print("Training new model...")
        trainer.train_model()
        trainer.save_model()
    
    # Initialize webcam
    # Note: On some systems (e.g., Linux), higher index (1) might be needed, or you might need a video file path.
    cap = cv2.VideoCapture(0) 
    
    print("\n" + "="*50)
    print("PUSH-UP COUNTER WITH POSTURE DETECTION STARTING")
    print("="*50)
    print("Instructions:")
    print("- Ensure camera is facing sideways to see your form.")
    print("- Keep your back straight (detected as CORRECT POSTURE).")
    print("- Go low enough (elbows near 90°) for each rep to count.")
    print("- Press 'q' to quit")
    print("- Press 'r' to reset counter")
    print("="*50)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam (cv2.VideoCapture(0) failed).")
        print("Please check camera connection and permissions.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror view (optional, but standard for video demos)
        frame = cv2.flip(frame, 1)
        # Convert to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = pose.process(rgb_frame)
        
        # Default status messages
        posture_text = "Waiting for Pose..."
        posture_color = (255, 255, 255)
        elbow_angle = 0
        
        if results.pose_landmarks:
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
            
            landmarks_list = results.pose_landmarks.landmark
            
            # Extract features and classify posture
            features = trainer.extract_features_from_landmarks(landmarks_list)
            
            # Use the trainer's angle calculator for the display angle
            left_shoulder = [landmarks_list[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks_list[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks_list[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks_list[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks_list[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks_list[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            elbow_angle = trainer.calculate_angle(left_shoulder, left_elbow, left_wrist)

            
            # Ensure we have the right number of features to predict
            if trainer.model and len(features) == 6: 
                try:
                    # Reshape feature list into a 2D array for prediction
                    posture_prediction = trainer.model.predict([features])[0]
                    posture_confidence = np.max(trainer.model.predict_proba([features]))
                    
                    # Update push-up counter
                    rep_completed = counter.update(landmarks_list)
                    
                    # Display results
                    posture_text = "CORRECT POSTURE ✓" if posture_prediction == 1 else "INCORRECT POSTURE ✗"
                    posture_color = (0, 255, 0) if posture_prediction == 1 else (0, 0, 255)
                    
                    # Simple feedback on why posture might be incorrect (if needed)
                    if posture_prediction == 0 and features[2] < 170:
                        posture_text = "INCORRECT: Hips too low/Piking"
                        posture_color = (0, 165, 255) # Orange
                        
                except Exception as e:
                    posture_text = f"Prediction Error: {e}"
                    posture_color = (0, 0, 255)

        else:
            posture_text = "NO PERSON DETECTED"
            posture_color = (0, 0, 255)

        # --- Display Overlays ---
        
        # Main counter display (Yellow)
        cv2.putText(frame, f"PUSH-UPS: {counter.count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # Posture feedback
        cv2.putText(frame, posture_text, (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, posture_color, 2)
        
        # State and angle info
        state_text = f"State: {counter.state.upper()}"
        cv2.putText(frame, state_text, (20, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Show angle information
        angle_color = (0, 255, 0) if elbow_angle < counter.elbow_angle_threshold else (0, 165, 255)
        cv2.putText(frame, f"Elbow Angle: {elbow_angle:.1f}°", (20, 190), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, angle_color, 2)

        # Display instructions
        cv2.putText(frame, "Press 'q' to quit | 'r' to reset", (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display the video feed
        cv2.imshow('Push-Up Counter with Posture Detection', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            counter.count = 0
            counter.state = "up"
            print("Counter reset!")
    
    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal push-up count: {counter.count}")
    print("Thank you for using the Push-Up Counter!")

# FIX: Corrected typo from "_main_" to "__main__"
if __name__ == "__main__":
    main()