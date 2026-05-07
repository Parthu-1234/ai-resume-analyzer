from flask import Flask, request, render_template
import PyPDF2

# ✅ AI Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 🔹 Skill database
skills_db = ["python", "java", "sql", "machine learning", "react", "node", "html", "css", "javascript"]

# 🔹 Job descriptions (for AI matching)
job_descriptions = {
    "Data Scientist": "python machine learning data analysis pandas numpy statistics",
    "Backend Developer": "python flask django api backend sql node server",
    "Frontend Developer": "html css javascript react ui ux frontend web"
}

# 🔹 Extract text from PDF
def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text.lower()

# 🔹 Extract skills (rule-based)
def extract_skills(text):
    found = []
    for skill in skills_db:
        if skill in text:
            found.append(skill)
    return found

# 🔹 AI Job Matching (TF-IDF)
def match_jobs(resume_text):
    documents = [resume_text]

    job_names = list(job_descriptions.keys())
    job_texts = list(job_descriptions.values())

    documents.extend(job_texts)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    results = {}
    for i in range(len(job_names)):
        score = round(similarity[i] * 100, 2)
        if score > 10:  # threshold
            results[job_names[i]] = score

    return results

# 🔹 Main Route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['resume']

        text = extract_text(file)

        skills = extract_skills(text)   # ✅ Skill extraction
        jobs = match_jobs(text)         # ✅ AI matching

        return render_template('result.html', skills=skills, jobs=jobs)

    return render_template('index.html')

# 🔹 Run app
if __name__ == '__main__':
    app.run(debug=True)