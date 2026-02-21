import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

print("1. Loading the trained AI model...")
# Load the brain you trained earlier
model = joblib.load('hostel_routing_model.pkl')

print("2. Loading the unseen test data...")
# Load the new test dataset
test_df = pd.read_csv('hostel_complaints_test_dataset.csv')

print("3. Making predictions...")
# Give the AI only the text, and ask it to predict the roles
predictions = model.predict(test_df['complaint_text'])

# 4. Grading the AI
actual_roles = test_df['expected_role']
accuracy = accuracy_score(actual_roles, predictions)

print("\n" + "="*50)
print(f"🌟 OVERALL AI ACCURACY: {accuracy * 100:.2f}%")
print("="*50 + "\n")

print("Detailed Performance Report by Role:")
# This generates a beautiful matrix showing how well it knows each category
report = classification_report(actual_roles, predictions)
print(report)

print("="*50)
print("Let's look at the actual vs predicted for the first 5 test cases:\n")
for i in range(5):
    text = test_df['complaint_text'].iloc[i]
    expected = test_df['expected_role'].iloc[i]
    predicted = predictions[i]
    
    # Calculate confidence for display
    confidence = model.predict_proba([text]).max() * 100
    
    print(f"Complaint: '{text}'")
    print(f"Expected: {expected} | AI Guessed: {predicted} ({confidence:.1f}% confident)\n")