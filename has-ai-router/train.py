import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib

print("Loading DataSet...")
df = pd.read_csv("hostel_complaints_training_dataset_ML_optimized.csv")

print("Building the scikitlearn pipeline...")
nlp_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True)),
    ('clf', SVC(kernel='linear', probability=True))
])

print("Training the model")
nlp_pipeline.fit(df['complaint_text'], df['assigned_role'])

print('Testing model on a new complaint...')
test_complaint = ["My internet is completely dead and the router has no lights."]
predicted_role = nlp_pipeline.predict(test_complaint)[0]
confidence = nlp_pipeline.predict_proba(test_complaint).max()

print(f"\nTest Complaint: '{test_complaint[0]}'")
print(f"AI Assigned To: {predicted_role} (Confidence: {confidence:.2f})\n")

print("5. Saving the model...")
joblib.dump(nlp_pipeline, 'hostel_routing_model.pkl')
print("Model saved successfully as 'hostel_routing_model.pkl'!")