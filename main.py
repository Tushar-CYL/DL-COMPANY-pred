import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Example Dataset (Replace with your dataset)
data = {
    'Company': ['Company A', 'Company B', 'Company C', 'Company D', 'Company E'],
    'Position': ['Software Engineer', 'Web Developer', 'Data Scientist', 'Web Developer', 'Software Engineer'],
    'Skills': ['Python, Java, SQL', 'HTML, CSS, JavaScript', 'Python, ML, Data Analysis', 'HTML, CSS, ReactJS', 'Java, Python, C++'],
    'Employees': [100, 50, 75, 30, 150]
}
df = pd.DataFrame(data)

# Preprocessing
# Encode Positions
label_encoder = LabelEncoder()
df['Position_Encoded'] = label_encoder.fit_transform(df['Position'])

# Vectorize Skills using TF-IDF
vectorizer = TfidfVectorizer()
skills_matrix = vectorizer.fit_transform(df['Skills']).toarray()

# Input Features: Skills and Employees
X = np.hstack((skills_matrix, np.log1p(df['Employees'].values).reshape(-1, 1)))  # Log-transform employees to scale

# Target: Encoded Position
y = df['Position_Encoded'].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert targets to one-hot encoding
num_classes = len(np.unique(y))
y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes)

# Build Neural Network Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Predict probabilities for each position
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(X_train, y_train_oh, validation_data=(X_test, y_test_oh), epochs=20, batch_size=16)

# Predict for User Input
def predict_user_input(user_skills, companies, num_employees):
    # Preprocess user skills
    user_vector = vectorizer.transform([user_skills]).toarray()  # Shape: (1, n_features)

    # Repeat user_vector for each company (to match num_employees size)
    user_vector_repeated = np.tile(user_vector, (len(num_employees), 1))  # Shape: (num_companies, n_features)

    # Log-transform employee count and reshape
    employee_feature = np.log1p(num_employees).reshape(-1, 1)  # Shape: (num_companies, 1)

    # Concatenate user_vector_repeated with employee_feature
    user_input = np.hstack((user_vector_repeated, employee_feature))  # Shape: (num_companies, n_features+1)

    # Predict probabilities
    predictions = model.predict(user_input)  # Shape: (num_companies, num_classes)

    # Combine predictions with company and position info
    results = []
    for i, company in enumerate(companies):
        for j, position in enumerate(label_encoder.classes_):
            score = predictions[i][j] * num_employees[i]
            results.append((company, position, score))
    
    # Rank results by score
    ranked_results = sorted(results, key=lambda x: x[2], reverse=True)
    return ranked_results


# Example User Input
user_skills = "Python, JavaScript, ReactJS"
companies = df['Company'].values
num_employees = df['Employees'].values

# Predict and Rank
ranked_results = predict_user_input(user_skills, companies, num_employees)

# Display Results
print("Ranked Results:")
for company, position, score in ranked_results[:5]:  # Top 5
    print(f"Company: {company}, Position: {position}, Score: {score:.2f}")
